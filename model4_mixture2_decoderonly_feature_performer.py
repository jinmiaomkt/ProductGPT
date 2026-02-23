from __future__ import annotations
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Tuple

df = pd.read_excel("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx", sheet_name=0)

## NewProductIndex3 is not a feature
feature_cols = [
    "Rarity", 
    "MaxLife", 
    "MaxOffense", 
    "MaxDefense",
    "WeaponTypeOneHandSword", 
    "WeaponTypeTwoHandSword", 
    "WeaponTypeArrow", 
    "WeaponTypeMagic", 
    "WeaponTypePolearm",
    "EthnicityIce", 
    "EthnicityRock", 
    "EthnicityWater", 
    "EthnicityFire", 
    # "EthnicityGrass", 
    "EthnicityThunder", 
    "EthnicityWind", 
    "GenderFemale", 
    "GenderMale",
    # "CountryFengDan", 
    "CountryRuiYue", 
    "CountryDaoQi", 
    "CountryZhiDong", 
    "CountryMengDe", 
    # "CountryXuMi",
    "type_figure",  # if it's numeric or one-hot encoded; else skip if it's a string
    "MinimumAttack", 
    "MaximumAttack",
    "MinSpecialEffect", 
    "MaxSpecialEffect",
    "SpecialEffectEfficiency", 
    "SpecialEffectExpertise",
    "SpecialEffectAttack", 
    "SpecialEffectSuper",
    "SpecialEffectRatio", 
    "SpecialEffectPhysical", 
    "SpecialEffectLife", 
    # "NewProductIndex3",  # only if you actually want it as a feature
    "LTO" 
]

# --------------------------------------------
# decision / product partition of the vocab
# --------------------------------------------
DECISION_IDS = torch.tensor([1,2,3,4,5,6,7,8,9])          # len = 9
PRODUCT_IDS  = torch.tensor(list(range(13, 57)))          # 13 … 56

## Determine max_token_id and the dimension
max_token_id =  68
# This is the count of numeric columns (your "feature_dim"). 
# For example, if you have 38 numeric columns (NOT counting the ID).
feature_dim = 34

feature_array = np.zeros((max_token_id + 1, feature_dim), dtype=np.float32)

for idx in range(len(df)):
    # The product ID for this row (should be in [1..65])
    token_id = int(df["NewProductIndex6"].iloc[idx])  
    
    # Gather the numeric features from the row. 
    # If you only want the columns in `feature_cols` (excluding "NewProductIndex3"), do:
    feats = df.loc[idx, feature_cols].values  # shape (34,) => [Rarity, MaxLife, MaxOffense, ...]

    feature_array[token_id, :] = feats

feature_tensor = torch.from_numpy(feature_array)  

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
def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate GeLU using the Tanh approximation.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))

class SpecialPlusFeatureLookup(nn.Module):
    def __init__(self, d_model: int,
                 feature_tensor: torch.Tensor,
                 product_ids: list[int],
                 vocab_size_src: int):

        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_tensor.size(1)

        # ── id and feature branches ─────────────────────────────
        self.id_embed  = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)

        # constant look‑up table (59,34)
        self.register_buffer("feat_tbl", feature_tensor, persistent=False)

        # mask: True for product tokens (13‥56 and 59)
        prod_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        prod_mask[product_ids] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)

        # learnable scale so the network can re‑weight the branches
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids: torch.Tensor,
                ext_features: torch.Tensor | None = None):
        """
        ids          : (B,T)  int64
        ext_features : (B,T,34) optional, used only for UNK products
        """
        ids_long = ids.long()                            # (B,T)

        # id branch
        id_vec = self.id_embed(ids_long)                 # (B,T,D)

        # feature branch
        raw_feat = self.feat_tbl[ids_long] if ext_features is None else ext_features
        feat_vec = self.feat_proj(raw_feat)              # (Batch, T 34, d_model 64)

        # zero‑out features for NON‑product tokens
        keep = self.prod_mask[ids_long]                  # (B,T) bool
        feat_vec = feat_vec * keep.unsqueeze(-1)         # broadcast

        # weighted sum
        return id_vec + self.gamma * feat_vec

##############################################################################
# 2. LayerNormalization
##############################################################################
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias  = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

##############################################################################
# 3. FeedForwardBlock
##############################################################################
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(
            self.dropout(gelu_approx(self.linear_1(x)))
        )

##############################################################################
# 4. InputEmbeddings
##############################################################################
class InputEmbeddings(nn.Module):
    """
    Learns a token embedding for an integer vocabulary,
    and optionally scales by sqrt(d_model).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len)
        # returns: (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

##############################################################################
# 5. PositionalEncoding
##############################################################################
class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding, up to a fixed max_seq_len.
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0)/d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape => (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def _extend_pe(self, new_len: int, d_model: int):
        device = self.pe.device
        pe = torch.zeros(1, new_len, d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            # dynamically expand positional encoding
            self._extend_pe(seq_len, x.size(-1))

        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)

##############################################################################
# 6. ResidualConnection
##############################################################################
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

##############################################################################
# 7. CausalPerformer (Self-Attention)
##############################################################################
class CausalPerformer(nn.Module):
    """
    Your blockwise causal performer from previous code, used for self-attention.
    """
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float=0.1,
                 kernel_type: str="exp",
                 nb_features: int=16,
                 block_size_h: int=1,
                 block_size_w: int=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        self.nb_features = nb_features
        self.kernel_type = kernel_type

        self.block_size_h = block_size_h
        self.block_size_w = block_size_w

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._create_feature_map()

    def _create_feature_map(self):
        omega = torch.randn(self.nb_features, self.d_k) / math.sqrt(self.d_k)
        # Typically we don't train omega in Performer random feature approach
        self.omega = nn.Parameter(omega, requires_grad=False)

    def _kernel_function(self, x: torch.Tensor):
        # "exp" or "gelu" kernel
        if self.kernel_type == "gelu":
            return gelu_approx(x) + 1e-6
        elif self.kernel_type == "exp":
            return torch.exp(-0.5 * (x**2))
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel_type}")

    def forward(self, q, k, v, gate: torch.Tensor | None = None):
        # q,k,v: (B, seq_len, d_model)
        B, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Project
        q = self.w_q(q).view(B, seq_len_q, self.n_heads, self.d_k)
        k = self.w_k(k).view(B, seq_len_k, self.n_heads, self.d_k)
        v = self.w_v(v).view(B, seq_len_k, self.n_heads, self.d_k)

        # Apply kernel
        q_prime = self._kernel_function(q @ self.omega.T)
        k_prime = self._kernel_function(k @ self.omega.T)

        # Normalize along feature dimension
        q_prime = q_prime / (q_prime.sum(dim=-1, keepdim=True) + 1e-6)
        k_prime = k_prime / (k_prime.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute prefix sums over k dimension
        K_cum = torch.cumsum(k_prime, dim=1)  # (B, seq_len_k, n_heads, r)
        KV_cum = torch.cumsum(k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=1)

        # Determine block boundaries (causal)
        # block i => last index is (i+1)*block_size_w - 1
        q_indices = torch.arange(seq_len_q, device=q.device)
        q_block_indices = q_indices // self.block_size_h
        key_indices = (q_block_indices + 1)*self.block_size_w - 1
        key_indices = key_indices.clamp(max=seq_len_k - 1)

        # Gather from prefix sums
        # K_cum_selected => shape (B, seq_len_q, n_heads, r)
        indices = key_indices.view(1, -1, 1, 1).expand(B, -1, self.n_heads, self.nb_features)
        K_cum_selected = K_cum.gather(dim=1, index=indices)

        # KV_cum_selected => shape (B, seq_len_q, n_heads, r, d_k)
        indices_kv = key_indices.view(1, -1, 1, 1, 1).expand(B, -1, self.n_heads, self.nb_features, self.d_k)
        KV_cum_selected = KV_cum.gather(dim=1, index=indices_kv)

        numerator   = torch.sum(q_prime.unsqueeze(-1) * KV_cum_selected, dim=-2)
        denominator = torch.sum(q_prime * K_cum_selected, dim=-1, keepdim=True)
        out = numerator / (denominator + 1e-6)

        # --- mixture-head gating (minimal surgery) ---
        # gate: (B, H) -> broadcast to (B, 1, H, 1)
        if gate is not None:
            out = out * gate[:, None, :, None]

        out = out.reshape(B, seq_len_q, self.d_model)
        out = self.w_o(out)
        return out

        # out = out.reshape(B, seq_len_q, self.d_model)
        # out = self.w_o(out)
        # return out

##############################################################################
# 8. DecoderBlock (Self-Attn + FeedForward)
##############################################################################
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int,
                 self_attention_block: CausalPerformer,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block

        self.residual_attn = ResidualConnection(d_model, dropout)
        self.residual_ff   = ResidualConnection(d_model, dropout)

    # def forward(self, x: torch.Tensor):
    #     # 1) Self-attn
    #     # x = self.residual_attn(x, self.self_attention_block)
    #     x = self.residual_attn(x, lambda x_norm:
    #         self.self_attention_block(x_norm, x_norm, x_norm))
    #     # 2) Feed-forward
    #     x = self.residual_ff(x, self.feed_forward_block)
    #     return x
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None):
        x = self.residual_attn(
            x,
            lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm, gate=gate)
        )
        x = self.residual_ff(x, self.feed_forward_block)
        return x

##############################################################################
# 9. Decoder (stack of blocks)
##############################################################################
class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(d_model)

    # def forward(self, x: torch.Tensor):
    #     for layer in self.layers:
    #         x = layer(x)
    #     return self.norm(x)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None):
        for layer in self.layers:
            x = layer(x, gate=gate)
        return self.norm(x)

##############################################################################
# 10. ProjectionLayer (mixture output head over H projections)
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
        mix_space: str = "logit",     # "logit" or "prob"
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
                      kernel_type: str="exp"):
    
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
        kernel_type  = kernel_type
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer