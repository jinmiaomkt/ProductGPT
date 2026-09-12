from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0
FIRST_PROD_ID = 13
LAST_PROD_ID = 56
UNK_PROD_ID = 59


class SpecialPlusFeatureLookup(nn.Module):
    """
    Token embedding + product-feature projection.

    Product tokens receive:
        id_embedding(token_id) + gamma * feature_projection(product_features)

    Non-product tokens receive:
        id_embedding(token_id)
    """

    def __init__(
        self,
        d_model: int,
        feature_tensor: torch.Tensor,
        product_ids: list[int],
        vocab_size_src: int,
        product_id_embed: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.feature_dim = int(feature_tensor.size(1))
        # product_id_embed=False removes the identity road for PRODUCT tokens:
        # they are represented by their attributes only, so two products with
        # the same features are indistinguishable. Special tokens (PAD, SOS,
        # decisions) have no features and keep their id embedding. This tests
        # whether the transformer's early holdout peak comes from memorising
        # campaign identity through the offer stream (EXPERIMENTS.md R18).
        self.product_id_embed = bool(product_id_embed)

        self.id_embed = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)

        self.register_buffer("feat_tbl", feature_tensor.float(), persistent=False)

        prod_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        for p in product_ids:
            if 0 <= int(p) < vocab_size_src:
                prod_mask[int(p)] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)

        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids:
            Any shape of integer token IDs, e.g. (B,S,4) or (B,S,10)

        returns:
            ids.shape + (D,)
        """
        ids = ids.long()
        id_vec = self.id_embed(ids)

        raw_feat = self.feat_tbl[ids]
        feat_vec = self.feat_proj(raw_feat)

        keep = self.prod_mask[ids]
        feat_vec = feat_vec * keep.unsqueeze(-1)
        if not self.product_id_embed:
            id_vec = id_vec * (~keep).unsqueeze(-1)

        return id_vec + self.gamma * feat_vec


class AttentionPool(nn.Module):
    """
    Learnable pooling over a small set of tokens inside one event.

    Used for:
        x_t:       4 LTO tokens
        o_{t-1}:  10 previous-obtained-product tokens
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:
            (B,S,K,D)

        mask:
            (B,S,K), True for valid non-PAD tokens

        returns:
            (B,S,D)
        """
        scores = self.score(x).squeeze(-1)          # (B,S,K)
        scores = scores.masked_fill(~mask, -1e9)

        all_pad = ~mask.any(dim=-1, keepdim=True)   # (B,S,1)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(all_pad, 0.0)

        return torch.sum(weights.unsqueeze(-1) * x, dim=2)


class OfferInventoryCrossAttention(nn.Module):
    """
    Current offer x_t attends to cumulative inventory-memory tokens H_{t-1}.

    offer_tok:
        (B,S,Lx,D), Lx=4

    inventory_tok:
        (B,S,Lo,D), Lo=10
        Row t contains o_{t-1}.

    inventory_mask:
        (B,S,Lo), True for valid non-PAD obtained-product tokens.

    offer_mask:
        (B,S,Lx), True for valid non-PAD offer tokens.

    Because row t contains o_{t-1}, when predicting y_t the query at event t is
    allowed to attend to memory rows <= t. This gives access to o_0,...,o_{t-1}
    but not future outcomes.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.offer_context_pool = AttentionPool(d_model, dropout)

    def forward(
        self,
        offer_tok: torch.Tensor,
        inventory_tok: torch.Tensor,
        inventory_mask: torch.Tensor,
        offer_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        returns:
            z_sat: (B,S,D)
            attn_mean if return_attention=True: (B,S,Lx,S*Lo)
        """
        B, S, Lx, D = offer_tok.shape
        B2, S2, Lo, D2 = inventory_tok.shape
        if (B, S, D) != (B2, S2, D2):
            raise ValueError(
                f"Shape mismatch: offer_tok={offer_tok.shape}, "
                f"inventory_tok={inventory_tok.shape}"
            )

        # Query: current offer tokens.
        q = self.q_proj(offer_tok)                             # (B,S,Lx,D)
        q = q.view(B, S, Lx, self.n_heads, self.d_head)
        q = q.permute(0, 3, 1, 2, 4).contiguous()               # (B,H,S,Lx,Dh)

        # Key/value: all inventory-memory tokens.
        M = S * Lo
        memory = inventory_tok.reshape(B, M, D)                 # (B,M,D)
        memory_mask = inventory_mask.reshape(B, M)              # (B,M)

        k = self.k_proj(memory).view(B, M, self.n_heads, self.d_head)
        v = self.v_proj(memory).view(B, M, self.n_heads, self.d_head)
        k = k.permute(0, 2, 1, 3).contiguous()                  # (B,H,M,Dh)
        v = v.permute(0, 2, 1, 3).contiguous()                  # (B,H,M,Dh)

        # Attention scores.
        logits = torch.einsum("bhsld,bhmd->bhslm", q, k)
        logits = logits / math.sqrt(self.d_head)                # (B,H,S,Lx,M)

        # Causal memory mask.
        device = offer_tok.device
        memory_event_index = torch.arange(S, device=device).repeat_interleave(Lo)  # (M,)
        query_event_index = torch.arange(S, device=device).unsqueeze(1)            # (S,1)
        causal_mask = memory_event_index.unsqueeze(0) <= query_event_index          # (S,M)
        valid_memory = causal_mask.unsqueeze(0) & memory_mask.unsqueeze(1)          # (B,S,M)

        logits = logits.masked_fill(~valid_memory[:, None, :, None, :], -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = attn.masked_fill(~valid_memory[:, None, :, None, :], 0.0)
        attn = self.dropout(attn)

        # Retrieve inventory context for each offer token.
        ctx = torch.einsum("bhslm,bhmd->bhsld", attn, v)          # (B,H,S,Lx,Dh)
        ctx = ctx.permute(0, 2, 3, 1, 4).contiguous()            # (B,S,Lx,H,Dh)
        ctx = ctx.view(B, S, Lx, D)
        ctx = self.out_proj(ctx)

        # Pool the offer-token-level inventory contexts into one satisfaction vector.
        z_sat = self.offer_context_pool(ctx, offer_mask)          # (B,S,D)

        if return_attention:
            return z_sat, attn.mean(dim=1)                       # (B,S,Lx,M)
        return z_sat


class CausalEventTransformer(nn.Module):
    """
    Causal Transformer over event representations r_1,...,r_t.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        recency_bias: bool = False,
    ):
        super().__init__()
        # ALiBi (Press et al. 2022): subtract slope_h * (i - j) from the
        # attention logit of query i on key j, with a fixed geometric slope per
        # head. No parameters. It gives attention a preference for RECENT
        # events that it otherwise lacks entirely -- this stack has no
        # positional encoding, so without it "one event ago" and "fifty events
        # ago" are indistinguishable except through the causal mask. A GRU has
        # recency built in; this is the cheapest way to hand it to attention.
        self.recency_bias = bool(recency_bias)
        self.n_heads = int(n_heads)
        if self.recency_bias:
            self.register_buffer("alibi_slopes", self._alibi_slopes(n_heads),
                                 persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _alibi_slopes(n_heads: int) -> torch.Tensor:
        """Geometric slopes 2^(-8/H), 2^(-16/H), ... as in the ALiBi paper."""
        def pow2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]
        if math.log2(n_heads).is_integer():
            return torch.tensor(pow2(n_heads))
        closest = 2 ** math.floor(math.log2(n_heads))
        extra = pow2(2 * closest)[0::2][: n_heads - closest]
        return torch.tensor(pow2(closest) + extra)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r: (B,S,D)
        returns: (B,S,D)
        """
        B, S = r.size(0), r.size(1)
        if not self.recency_bias:
            causal_mask = torch.triu(
                torch.ones(S, S, device=r.device, dtype=torch.bool),
                diagonal=1,
            )
            return self.norm(self.encoder(r, mask=causal_mask))

        # Additive float mask, one per (batch, head), batch-major as
        # nn.MultiheadAttention expects: index = b * n_heads + h.
        pos = torch.arange(S, device=r.device)
        dist = (pos[:, None] - pos[None, :]).clamp_min(0).to(r.dtype)   # (S,S)
        bias = -self.alibi_slopes.to(r.dtype)[:, None, None] * dist    # (H,S,S)
        bias = bias.masked_fill(pos[None, :] > pos[:, None], float("-inf"))
        mask = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.n_heads, S, S)
        return self.norm(self.encoder(r, mask=mask))


class UserMixtureOutputHead(nn.Module):
    """
    H output projections combined by per-customer mixture weights.

    This is Lu & Kannan's (JMR 2025) heterogeneous-mixture mechanism, ported
    from gen 4's model4_mixture2_*. Instead of one shared projection, the head
    holds H of them and each customer n gets weights

        alpha_n = softmax(user_mix_logits[n])        sum_h alpha_nh = 1

    so their output is a convex combination sum_h alpha_nh * logits_h. The
    weights are a soft membership over H behavioural patterns -- a continuous
    latent segmentation learned end to end, rather than the arbitrary latent
    vector a plain per-user embedding gives you. That is what makes it
    interpretable: alpha_n says WHICH pattern describes a customer, and the
    H projections say what each pattern predicts.

    OUT-OF-SAMPLE CUSTOMERS. Customers the model never trained on carry index
    0 and have no estimated alpha. They receive the population mean

        alpha_bar_h = (1/N) sum_n alpha_nh

    taken over the customers training actually updated -- the paper's "average
    head weight from the training population", and an empirical-Bayes prior
    mean. Note this averages the SOFTMAXED weights, not the logits; averaging
    logits and then softmaxing is a different (and wrong) quantity.

    Resolution is per sample rather than by a global mode, so a batch mixing
    in-sample and out-of-sample customers is handled correctly.

    Mixing happens in LOGIT space: the output stays logits, which is what the
    cross-entropy loss expects. Mixing in probability space would return a
    distribution and silently break the loss.
    """

    def __init__(self, d_model: int, vocab_size: int, num_users: int,
                 num_mix_heads: int):
        super().__init__()
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.num_mix_heads = int(num_mix_heads)

        self.proj_weight = nn.Parameter(
            torch.empty(self.num_mix_heads, d_model, vocab_size))
        self.proj_bias = nn.Parameter(torch.zeros(self.num_mix_heads, vocab_size))
        for h in range(self.num_mix_heads):
            nn.init.xavier_uniform_(self.proj_weight[h])

        # Zero init => softmax is uniform at the start, so every customer
        # begins as an average customer and heterogeneity has to be earned.
        self.user_mix_logits = nn.Embedding(num_users, self.num_mix_heads)
        nn.init.zeros_(self.user_mix_logits.weight)

        self.register_buffer("mean_alpha", torch.full((self.num_mix_heads,),
                                                      1.0 / self.num_mix_heads))

    @torch.no_grad()
    def refresh_mean_alpha(self, trained_user_indices) -> torch.Tensor:
        """Cache alpha_bar over the customers training actually updated."""
        if not len(trained_user_indices):
            return self.mean_alpha
        idx = torch.as_tensor(list(trained_user_indices), dtype=torch.long,
                              device=self.user_mix_logits.weight.device)
        alpha = F.softmax(self.user_mix_logits(idx), dim=-1)   # (N,H)
        self.mean_alpha.copy_(alpha.mean(dim=0).detach())
        return self.mean_alpha

    def alpha_for(self, user_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if user_idx.dim() > 1:
            user_idx = user_idx.squeeze(-1)
        alpha = F.softmax(self.user_mix_logits(user_idx.long()), dim=-1).to(dtype)
        unknown = (user_idx == 0).unsqueeze(-1)                # (B,1)
        mean = self.mean_alpha.to(dtype).unsqueeze(0).expand_as(alpha)
        return torch.where(unknown, mean, alpha)

    def forward(self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None,
                return_alpha: bool = False):
        B = x.size(0)
        head_logits = torch.einsum("btd,hdv->bthv", x, self.proj_weight)
        head_logits = head_logits + self.proj_bias[None, None]     # (B,T,H,V)

        if user_idx is None:
            alpha = self.mean_alpha.to(x.dtype).unsqueeze(0).expand(B, -1)
        else:
            alpha = self.alpha_for(user_idx, x.dtype)              # (B,H)

        out = torch.sum(alpha[:, None, :, None] * head_logits, dim=2)  # (B,T,V)
        return (out, alpha) if return_alpha else out


class MultiStreamStateSpaceTransformer(nn.Module):
    """
    Final agreed architecture.

    Raw event-level inputs:
        lto_ids:       x_t, shape (B,S,4)
        obtained_ids:  o_{t-1}, shape (B,S,10)
        prev_dec_ids:  y_{t-1}, shape (B,S)

    Core mechanism:
        x_t attends to cumulative inventory-memory tokens to produce z_sat_t.
        r_t = phi(z_x_t, z_sat_t, z_o_{t-1}, z_y_{t-1}, h^H_{t-1}).
        s_t = CausalTransformer(r_{<=t})_t.
        logits_t = Head(s_t).
    """

    def __init__(
        self,
        vocab_size_src: int,
        vocab_size_tgt: int,
        max_seq_len: int,
        ai_rate: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        feature_tensor: torch.Tensor,
        lto_len: int = 4,
        obtained_len: int = 10,
        prev_dec_len: int = 1,
        num_users: Optional[int] = None,
        use_user_embedding: bool = True,
        num_mix_heads: int = 0,
        return_attention_default: bool = False,
        encoder: str = "transformer",
        use_offer_inventory_attn: bool = True,
        attn_recency_bias: bool = False,
        product_id_embed: bool = True,
    ):
        super().__init__()
        # Component ablation switches (EXPERIMENTS.md R18). Together with the
        # separate RecurrentBaseline they form a 2x2 over
        #   {sequence encoder: attention, GRU} x {offer-inventory cross-attn: on, off}
        # plus a recency-bias arm and a features-only-product arm, so each
        # piece of the architecture can be credited or blamed on its own.
        encoder = str(encoder).lower()
        if encoder not in ("transformer", "gru"):
            raise ValueError(f"encoder must be 'transformer' or 'gru', got {encoder!r}")
        self.encoder_kind = encoder
        self.use_offer_inventory_attn = bool(use_offer_inventory_attn)

        if ai_rate != lto_len + obtained_len + prev_dec_len:
            raise ValueError(
                f"ai_rate must equal lto_len+obtained_len+prev_dec_len, got "
                f"{ai_rate} vs {lto_len}+{obtained_len}+{prev_dec_len}"
            )

        self.vocab_size_src = int(vocab_size_src)
        self.vocab_size_tgt = int(vocab_size_tgt)
        self.max_seq_len = int(max_seq_len)
        self.ai_rate = int(ai_rate)
        self.lto_len = int(lto_len)
        self.obtained_len = int(obtained_len)
        self.prev_dec_len = int(prev_dec_len)
        self.d_model = int(d_model)
        self.return_attention_default = bool(return_attention_default)

        # Your training helpers should treat this model's output as logits.
        self.projection_mix_space = "logit"

        product_ids = list(range(FIRST_PROD_ID, LAST_PROD_ID + 1)) + [UNK_PROD_ID]
        self.product_embed = SpecialPlusFeatureLookup(
            d_model=d_model,
            feature_tensor=feature_tensor,
            product_ids=product_ids,
            vocab_size_src=vocab_size_src,
            product_id_embed=product_id_embed,
        )

        self.decision_embed = nn.Embedding(vocab_size_src, d_model)
        self.offer_pool = AttentionPool(d_model, dropout)
        self.outcome_pool = AttentionPool(d_model, dropout)

        self.inventory_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        self.offer_inventory_attn = (
            OfferInventoryCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            if self.use_offer_inventory_attn else None)

        self.use_user_embedding = bool(use_user_embedding and num_users is not None)
        if self.use_user_embedding:
            self.user_embed = nn.Embedding(num_users, d_model)
        else:
            self.user_embed = None

        # Event fusion: [z_x, z_sat, z_o, z_y, h_H] plus optional user embedding.
        n_pieces = 5 if self.use_offer_inventory_attn else 4   # z_x, [z_sat], z_o, z_y, h_H
        fusion_in = n_pieces * d_model + (d_model if self.use_user_embedding else 0)
        self.event_fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

        if self.encoder_kind == "transformer":
            self.event_model = CausalEventTransformer(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                recency_bias=attn_recency_bias,
            )
        else:
            # Stacked single-layer GRUs with explicit inter-layer dropout: the
            # same model as nn.GRU(num_layers=N, dropout=p) but without the
            # cuDNN dropout-state teardown crash on Windows (see
            # model_recurrent_baseline.py). Everything upstream -- streams,
            # pooling, inventory GRU, cross-attention -- is untouched, so this
            # arm isolates the sequence encoder.
            self.event_model = nn.ModuleList(
                nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1,
                       batch_first=True)
                for _ in range(n_layers))
            self.event_model_dropout = nn.Dropout(dropout)
            self.event_model_norm = nn.LayerNorm(d_model)

        # num_mix_heads > 0 swaps the final projection for Lu & Kannan's
        # per-customer mixture over H projections. The pre-head MLP is kept
        # either way so the two differ only in how the last layer is formed.
        self.num_mix_heads = int(num_mix_heads or 0)
        self.head_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.num_mix_heads > 0:
            if num_users is None:
                raise ValueError("num_mix_heads requires num_users")
            self.mixture_head = UserMixtureOutputHead(
                d_model, vocab_size_tgt, num_users, self.num_mix_heads)
            self.output_head = None
        else:
            self.mixture_head = None
            self.output_head = nn.Linear(d_model, vocab_size_tgt)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def split_legacy_aggregate_input(self, aggregate_input: torch.Tensor):
        """
        Optional backward compatibility path.

        aggregate_input: (B, S*15)
        returns: lto_ids (B,S,4), obtained_ids (B,S,10), prev_dec_ids (B,S)
        """
        B, T = aggregate_input.shape
        S = T // self.ai_rate
        x = aggregate_input[:, : S * self.ai_rate].view(B, S, self.ai_rate)

        lto_ids = x[:, :, : self.lto_len]
        obtained_ids = x[:, :, self.lto_len : self.lto_len + self.obtained_len]
        prev_dec_ids = x[:, :, self.lto_len + self.obtained_len]

        return lto_ids, obtained_ids, prev_dec_ids

    def forward(
        self,
        lto_ids: torch.Tensor,
        obtained_ids: Optional[torch.Tensor] = None,
        prev_dec_ids: Optional[torch.Tensor] = None,
        user_idx: Optional[torch.Tensor] = None,
        projection_gate_mode: Optional[str] = None,
        return_hidden: bool = False,
        return_attention: Optional[bool] = None,
        return_proj_alpha: bool = False,
    ):
        """
        Preferred call:
            logits = model(lto_ids, obtained_ids, prev_dec_ids, user_idx)

        Backward-compatible call:
            logits = model(aggregate_input, user_idx)

        projection_gate_mode and return_proj_alpha are accepted for compatibility
        with your older training/evaluation code.
        """
        # Backward compatibility for old call model(aggregate_input, user_idx, ...).
        if obtained_ids is not None and prev_dec_ids is None and obtained_ids.dim() == 1:
            user_idx = obtained_ids
            obtained_ids = None

        if obtained_ids is None or prev_dec_ids is None:
            lto_ids, obtained_ids, prev_dec_ids = self.split_legacy_aggregate_input(lto_ids)

        if return_attention is None:
            return_attention = self.return_attention_default

        lto_ids = lto_ids.long()
        obtained_ids = obtained_ids.long()
        prev_dec_ids = prev_dec_ids.long()

        lto_mask = lto_ids.ne(PAD_ID)              # (B,S,4)
        out_mask = obtained_ids.ne(PAD_ID)         # (B,S,10)

        # 1. Token embeddings.
        lto_tok = self.product_embed(lto_ids)       # (B,S,4,D)
        out_tok = self.product_embed(obtained_ids)  # (B,S,10,D)

        # 2. Stream-level event summaries.
        z_x = self.offer_pool(lto_tok, lto_mask)    # (B,S,D)
        z_o = self.outcome_pool(out_tok, out_mask)  # (B,S,D)
        z_y = self.decision_embed(prev_dec_ids)     # (B,S,D)

        # 3. Latent inventory state from immediate outcomes.
        h_H, _ = self.inventory_gru(z_o)            # (B,S,D)

        # 4. Current offer attends to cumulative inventory tokens.
        if self.offer_inventory_attn is None:
            z_sat, sat_attn = None, None
        elif return_attention:
            z_sat, sat_attn = self.offer_inventory_attn(
                offer_tok=lto_tok,
                inventory_tok=out_tok,
                inventory_mask=out_mask,
                offer_mask=lto_mask,
                return_attention=True,
            )
        else:
            z_sat = self.offer_inventory_attn(
                offer_tok=lto_tok,
                inventory_tok=out_tok,
                inventory_mask=out_mask,
                offer_mask=lto_mask,
                return_attention=False,
            )
            sat_attn = None

        # 5. Event representation r_t.
        pieces = [z_x, z_o, z_y, h_H] if z_sat is None else [z_x, z_sat, z_o, z_y, h_H]

        if self.use_user_embedding and user_idx is not None:
            z_u = self.user_embed(user_idx.long())                 # (B,D)
            z_u = z_u.unsqueeze(1).expand(-1, z_x.size(1), -1)     # (B,S,D)
            pieces.append(z_u)

        r = self.event_fusion(torch.cat(pieces, dim=-1))           # (B,S,D)

        # 6. Causal sequence model over event representations.
        if self.encoder_kind == "transformer":
            s = self.event_model(r)                                # (B,S,D)
        else:
            s = r.contiguous()
            last = len(self.event_model) - 1
            for i, gru in enumerate(self.event_model):
                s, _ = gru(s)
                if i < last:
                    s = self.event_model_dropout(s).contiguous()
            s = self.event_model_norm(s)

        # 7. Decision logits.
        h = self.head_trunk(s)
        if self.mixture_head is not None:
            logits = self.mixture_head(h, user_idx)                # (B,S,V)
        else:
            logits = self.output_head(h)                           # (B,S,V)

        if return_hidden and return_attention:
            return logits, s, sat_attn
        if return_hidden:
            return logits, s
        if return_attention:
            return logits, sat_attn
        return logits


def build_transformer(
    vocab_size_src: int,
    vocab_size_tgt: int,
    max_seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    nb_features: Optional[int] = None,
    feature_tensor: Optional[torch.Tensor] = None,
    special_token_ids=None,
    kernel_type: str = "exp",
    ai_rate: int = 15,
    num_users: Optional[int] = None,
    projection_mix_space: str = "logit",
    **kwargs,
):
    """
    Drop-in builder with the same broad signature as your old build_transformer().
    """
    if feature_tensor is None:
        raise ValueError("feature_tensor is required for MultiStreamStateSpaceTransformer.")

    return MultiStreamStateSpaceTransformer(
        vocab_size_src=vocab_size_src,
        vocab_size_tgt=vocab_size_tgt,
        max_seq_len=max_seq_len,
        ai_rate=ai_rate,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        feature_tensor=feature_tensor,
        lto_len=kwargs.get("lto_len", 4),
        obtained_len=kwargs.get("obtained_len", 10),
        prev_dec_len=kwargs.get("prev_dec_len", 1),
        num_users=num_users,
        use_user_embedding=kwargs.get("use_user_embedding", True),
        num_mix_heads=kwargs.get("num_mix_heads", 0),
        return_attention_default=kwargs.get("return_attention_default", False),
        encoder=kwargs.get("encoder", "transformer"),
        use_offer_inventory_attn=kwargs.get("use_offer_inventory_attn", True),
        attn_recency_bias=kwargs.get("attn_recency_bias", False),
        product_id_embed=kwargs.get("product_id_embed", True),
    )
