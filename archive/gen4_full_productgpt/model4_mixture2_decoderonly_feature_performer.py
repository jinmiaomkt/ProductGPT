"""
Generation 4 - mixture-head v2, feature-based embeddings.

Like the mixture variant, but the output layer is `UserMixtureOutputHead`:
H parallel output projections combined by per-user weights, mixed in either
logit or probability space, with `user` / `mean` / `uniform` gate modes so the
model can generalise to users unseen during training.

NOTE: this file keeps its own `ProjectionLayer`. It is the one component that
genuinely differs from the version shared by the other seven model modules, so
it was deliberately NOT replaced by `shared.layers.ProjectionLayer`.

The remaining shared blocks now live in shared/. Numerics are unchanged.
The module-level `pd.read_excel(...)` that ran on import has been removed; pass
the feature table in as `feature_tensor` via `shared.features.load_feature_tensor`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import CausalPerformer
from shared.decoder import Decoder, DecoderBlock
from shared.embeddings import SpecialPlusFeatureLookup
from shared.layers import (
    FeedForwardBlock,
    PositionalEncoding,
    gelu_approx,
)

class UserHeadGate(nn.Module):
    """
    gate(u) -> weights over heads: [B, H], sum_h gate=1
    Train: use per-user gates
    Val/Test: use mean gate across users
    """
    def __init__(self, num_users: int, num_heads: int):
        super().__init__()
        self.logits = nn.Embedding(num_users, num_heads)  # [U, H]
        nn.init.zeros_(self.logits.weight)               # start ~uniform after softmax

        self.register_buffer("mean_gate", torch.full((num_heads,), 1.0 / num_heads))
        self.use_mean_gate = False

    @torch.no_grad()
    def update_mean_gate(self):
        probs = torch.softmax(self.logits.weight, dim=-1)     # [U, H]
        self.mean_gate.copy_(probs.mean(dim=0))               # [H]

    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor:
        # user_ids: [B]
        if self.use_mean_gate:
            return self.mean_gate[None, :].expand(user_ids.size(0), -1)  # [B, H]
        return torch.softmax(self.logits(user_ids), dim=-1)              # [B, H]


##############################################################################
# 1. gelu_approx
##############################################################################

class UserMixtureOutputHead(nn.Module):
    """
    H output projections + user-specific mixture weights over H heads.

    Inputs
    ------
    x:        (B, T, D) decoder hidden states
    user_idx: (B,) integer user IDs in [0, num_users-1]  [required if gate_mode='user']

    Outputs
    -------
    If mix_space == "logit":
        agg_logits: (B, T, V)
    If mix_space == "prob":
        agg_prob:   (B, T, V)

    Gate modes
    ----------
    - "user"   : use per-user mixture weights alpha_u
    - "mean"   : use a shared mean gate (cached or computed)
    - "uniform": use equal weights over H heads
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_users: int,
        num_mix_heads: int,
        *,
        mix_space: str = "prob",     # "logit" or "prob"
        use_bias: bool = True,
        init_uniform_mix: bool = True,
    ):
        super().__init__()
        assert mix_space in {"logit", "prob"}, "mix_space must be 'logit' or 'prob'"

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.num_mix_heads = num_mix_heads
        self.mix_space = mix_space
        self.use_bias = use_bias

        H = num_mix_heads
        D = d_model
        V = vocab_size

        # H separate output projections, packed as one tensor
        # proj_weight[h]: (D, V)
        self.proj_weight = nn.Parameter(torch.empty(H, D, V))
        if use_bias:
            self.proj_bias = nn.Parameter(torch.zeros(H, V))
        else:
            self.register_parameter("proj_bias", None)

        # User-specific mixture logits over H heads
        self.user_mix_logits = nn.Embedding(num_users, H)

        # Optional cached mean alpha (e.g., computed from training users only)
        self.register_buffer("mean_alpha_buffer", torch.empty(0), persistent=True)

        self.reset_parameters(init_uniform_mix=init_uniform_mix)

    def reset_parameters(self, *, init_uniform_mix: bool = True):
        for h in range(self.num_mix_heads):
            nn.init.xavier_uniform_(self.proj_weight[h])

        if self.proj_bias is not None:
            nn.init.zeros_(self.proj_bias)

        # softmax(0,...,0) => uniform gate
        if init_uniform_mix:
            nn.init.zeros_(self.user_mix_logits.weight)
        else:
            nn.init.normal_(self.user_mix_logits.weight, mean=0.0, std=0.02)

    def _head_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns head_logits: (B, T, H, V)
        """
        # (B,T,D) x (H,D,V) -> (B,T,H,V)
        head_logits = torch.einsum("btd,hdv->bthv", x, self.proj_weight)
        if self.proj_bias is not None:
            head_logits = head_logits + self.proj_bias.unsqueeze(0).unsqueeze(0)  # (1,1,H,V)
        return head_logits

    def _user_alpha(self, user_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        user_idx: (B,) or (B,1)
        returns alpha: (B, H), rows sum to 1
        """
        if user_idx.dim() > 1:
            user_idx = user_idx.squeeze(-1)
        alpha_logits = self.user_mix_logits(user_idx.long())  # (B, H)
        alpha = F.softmax(alpha_logits, dim=-1).to(dtype=dtype)
        return alpha

    def _uniform_alpha(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((B, self.num_mix_heads), 1.0 / self.num_mix_heads, device=device, dtype=dtype)

    def _mean_alpha(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns shared mean alpha expanded to (B, H).
        Uses cached buffer if available; otherwise computes over all users (debug fallback).
        """
        if self.mean_alpha_buffer.numel() == self.num_mix_heads:
            mean_alpha = self.mean_alpha_buffer.to(device=device, dtype=dtype)
        else:
            # Fallback: mean over all users (okay for debugging; prefer train-only cache for eval)
            alpha_all = F.softmax(self.user_mix_logits.weight, dim=-1).to(device=device, dtype=dtype)  # (U,H)
            mean_alpha = alpha_all.mean(dim=0)  # (H,)
        return mean_alpha.unsqueeze(0).expand(B, -1)  # (B,H)

    @torch.no_grad()
    def set_mean_alpha_from_user_ids(self, user_ids) -> torch.Tensor:
        """
        Cache a mean gate from a specified set of user IDs (recommended: training users only).
        """
        if not torch.is_tensor(user_ids):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.user_mix_logits.weight.device)
        else:
            user_ids = user_ids.to(device=self.user_mix_logits.weight.device, dtype=torch.long)

        alpha = F.softmax(self.user_mix_logits(user_ids), dim=-1)  # (N,H)
        self.mean_alpha_buffer = alpha.mean(dim=0).detach()        # (H,)
        return self.mean_alpha_buffer

    def forward(
        self,
        x: torch.Tensor,                     # (B, T, D)
        user_idx: Optional[torch.Tensor],    # (B,) if gate_mode='user'
        *,
        gate_mode: str = "user",             # "user" | "mean" | "uniform"
        return_alpha: bool = False,
        return_head_logits: bool = False,
    ):
        assert gate_mode in {"user", "mean", "uniform"}, "gate_mode must be 'user', 'mean', or 'uniform'"

        B = x.size(0)
        head_logits = self._head_logits(x)  # (B,T,H,V)

        if gate_mode == "user":
            if user_idx is None:
                raise ValueError("user_idx is required when gate_mode='user'")
            alpha = self._user_alpha(user_idx, dtype=x.dtype)  # (B,H)
        elif gate_mode == "mean":
            alpha = self._mean_alpha(B=B, device=x.device, dtype=x.dtype)  # (B,H)
        else:  # uniform
            alpha = self._uniform_alpha(B=B, device=x.device, dtype=x.dtype)  # (B,H)

        # Broadcast alpha over time and vocab: (B,1,H,1)
        alpha_bt = alpha[:, None, :, None]

        if self.mix_space == "logit":
            # Convex combination in logit space
            out = torch.sum(alpha_bt * head_logits, dim=2)   # (B,T,V)
        else:
            # True mixture in probability space
            head_prob = F.softmax(head_logits, dim=-1)       # (B,T,H,V)
            out = torch.sum(alpha_bt * head_prob, dim=2)     # (B,T,V), valid prob dist

        extras = []
        if return_alpha:
            extras.append(alpha)
        if return_head_logits:
            extras.append(head_logits)

        if extras:
            return (out, *extras)
        return out


class ProjectionLayer(nn.Module):
    """
    Pure output projection wrapper.

    IMPORTANT:
    - This layer expects decoder hidden states x=(B,T,D)
    - It does NOT call the decoder itself
    """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_users: int,
        num_heads: int,              # tie mixture-head count to transformer attention heads
        *,
        mix_space: str = "prob",     # "prob" or "logit"
        use_bias: bool = True,
    ):
        super().__init__()
        self.mix_space = mix_space

        self.output_head = UserMixtureOutputHead(
            d_model=d_model,
            vocab_size=vocab_size,
            num_users=num_users,
            num_mix_heads=num_heads,
            mix_space=mix_space,
            use_bias=use_bias,
        )

    @torch.no_grad()
    def set_mean_gate_from_train_users(self, train_user_ids):
        return self.output_head.set_mean_alpha_from_user_ids(train_user_ids)

    def forward(
        self,
        x: torch.Tensor,                    # (B,T,D) decoder hidden states
        user_idx: Optional[torch.Tensor],   # (B,) if gate_mode='user'
        *,
        gate_mode: str = "user",
        return_alpha: bool = False,
        return_head_logits: bool = False,
    ):
        return self.output_head(
            x=x,
            user_idx=user_idx,
            gate_mode=gate_mode,
            return_alpha=return_alpha,
            return_head_logits=return_head_logits,
        )


##############################################################################
# 11. DecoderOnlyTransformer (GPT-style)
##############################################################################
class Transformer(nn.Module):
    """
    A single-stack (decoder-only) Transformer for next-token prediction.

    Notes
    -----
    - `self.gate` below is your EXISTING attention-head gate used by the decoder blocks.
    - `self.projection` is the NEW mixture output head over H output projections.
    - These are separate mechanisms.
    """
    def __init__(
        self,
        vocab_size_tgt: int,
        vocab_size_src: int,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_users: int,
        d_ff: int,
        nb_features: int,
        dropout: float,
        feature_tensor: torch.Tensor,
        special_token_ids: torch.Tensor,
        kernel_type: str = "exp",
        projection_mix_space: str = "prob",   # "prob" or "logit"
    ):
        super().__init__()

        self.vocab_size_tgt = vocab_size_tgt
        self.n_heads = n_heads
        self.num_users = num_users
        self.projection_mix_space = projection_mix_space

        self.token_embed = SpecialPlusFeatureLookup(
            d_model=d_model,
            feature_tensor=feature_tensor,                  # (59, 34)
            product_ids=list(range(13, 57)) + [59],        # 13 … 56 plus 59
            vocab_size_src=vocab_size_src                  # e.g. 59
        )

        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        # Your existing decoder attention gate (kept as-is)
        self.gate = UserHeadGate(num_users=num_users, num_heads=n_heads)

        # Build N decoder blocks
        blocks = []
        for _ in range(n_layers):
            performer = CausalPerformer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                kernel_type=kernel_type,
                nb_features=nb_features
            )
            ff_block = FeedForwardBlock(d_model, d_ff, dropout)
            blk = DecoderBlock(d_model, performer, ff_block, dropout)
            blocks.append(blk)

        self.decoder = Decoder(d_model, nn.ModuleList(blocks))

        # Final output projection (NEW mixture head)
        self.projection = ProjectionLayer(
            d_model=d_model,
            vocab_size=vocab_size_tgt,
            num_users=num_users,
            num_heads=n_heads,                 # H = number of attention heads
            mix_space=projection_mix_space,    # "prob" or "logit"
            use_bias=True,
        )

        # Optional extra head (kept from your original code)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 128)
        )

        # (Optional) param init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def set_projection_mean_gate_from_train_users(self, train_user_ids):
        """
        Call this before validation/test if you want a shared mean mixture gate
        computed from training users only.
        """
        return self.projection.set_mean_gate_from_train_users(train_user_ids)

    def forward(
        self,
        input_seq: torch.Tensor,
        user_ids: Optional[torch.LongTensor] = None,
        *,
        return_hidden: bool = False,
        return_proj_alpha: bool = False,
        projection_gate_mode: Optional[str] = None,  # None => auto ("user" if user_ids else "uniform")
    ):
        """
        Returns
        -------
        If projection_mix_space == "logit":
            output is logits (B,T,V) unless return_proj_alpha=True
        If projection_mix_space == "prob":
            output is probabilities (B,T,V) unless return_proj_alpha=True

        Auto gate behavior:
        - if user_ids is provided -> projection_gate_mode defaults to "user"
        - else                    -> projection_gate_mode defaults to "uniform"
        """
        x = self.token_embed(input_seq)
        x = self.pos_enc(x)

        # Existing decoder attention-head gate
        attn_gate = None
        if user_ids is not None:
            attn_gate = self.gate(user_ids)   # (B, H)

        x = self.decoder(x, gate=attn_gate)   # (B,T,D)

        if projection_gate_mode is None:
            projection_gate_mode = "user" if user_ids is not None else "uniform"

        # Output projection with user/mean/uniform mixture gate
        proj_out = self.projection(
            x,
            user_idx=user_ids,
            gate_mode=projection_gate_mode,
            return_alpha=return_proj_alpha,
        )

        if return_proj_alpha:
            out, proj_alpha = proj_out  # out: (B,T,V), proj_alpha: (B,H)
            if return_hidden:
                return out, x, proj_alpha
            return out, proj_alpha

        # Normal path
        out = proj_out
        if return_hidden:
            return out, x
        return out
    
##############################################################################
# 12. Build function
##############################################################################
def build_transformer(vocab_size_src: int,
                      vocab_size_tgt: int,
                      max_seq_len: int,
                      d_model: int,
                      n_layers: int,
                      num_users: int,
                      n_heads: int,
                      d_ff: int,
                      dropout: float,
                      nb_features: int,
                      feature_tensor: torch.Tensor,
                      special_token_ids: torch.Tensor,
                      kernel_type: str="exp",
                      projection_mix_space: str = "prob"):
    
    transformer = Transformer(
        vocab_size_src   = vocab_size_src,
        vocab_size_tgt   = vocab_size_tgt,
        max_seq_len  = max_seq_len,
        d_model      = d_model,
        n_layers     = n_layers,
        n_heads      = n_heads,
        num_users    = num_users,
        d_ff         = d_ff,
        dropout      = dropout,
        nb_features  = nb_features,
        feature_tensor = feature_tensor,
        special_token_ids = special_token_ids,
        kernel_type  = kernel_type,
        projection_mix_space = projection_mix_space,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
