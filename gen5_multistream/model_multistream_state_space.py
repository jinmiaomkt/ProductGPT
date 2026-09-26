from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0
FIRST_PROD_ID = 13
LAST_PROD_ID = 56
N_PRODUCTS = LAST_PROD_ID - FIRST_PROD_ID + 1   # 44 inventory slots
UNK_PROD_ID = 59


def set_product_range(first: int, last: int, unk: int) -> None:
    """R28: switch to a per-product vocabulary (13-130, UNK 133).

    Modules read these when a model is BUILT, so call this before
    build_transformer / build_recurrent_baseline. Built models keep the
    range they were constructed with.
    """
    global FIRST_PROD_ID, LAST_PROD_ID, N_PRODUCTS, UNK_PROD_ID
    FIRST_PROD_ID, LAST_PROD_ID, UNK_PROD_ID = int(first), int(last), int(unk)
    N_PRODUCTS = LAST_PROD_ID - FIRST_PROD_ID + 1


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


def additive_inventory(obtained_ids: torch.Tensor,
                       init_count: Optional[torch.Tensor] = None,
                       init_last: Optional[torch.Tensor] = None):
    """
    Cumulative per-product inventory, lagged to the previous occasion (R23).

    obtained_ids: (B,S,10). Row t already carries o_(t-1) (the R1 shift), so a
        running sum through row t counts acquisitions up to occasion t-1 only.
    init_count:   (B,44) acquisitions BEFORE the window, from the loader, so
        items obtained before truncation are not forgotten. None = zeros.
    init_last:    (B,44) row index, relative to the window start, of each
        product's last acquisition before the window (<= 0). None = never.

    Returns count (B,S,44) float and rows_since_last (B,S,44) float. Counts are
    additive by construction -- they never decrease, and an empty row (37.9% of
    rows, mostly inserted NotBuy days) leaves them unchanged.
    """
    B, S, _ = obtained_ids.shape
    dev = obtained_ids.device
    first, last = FIRST_PROD_ID, LAST_PROD_ID
    n_products = last - first + 1
    valid = (obtained_ids >= first) & (obtained_ids <= last)
    idx = (obtained_ids - first).clamp(0, n_products - 1)
    per_row = torch.zeros(B, S, n_products, device=dev, dtype=torch.float32)
    per_row.scatter_add_(2, idx, valid.to(torch.float32))

    count = per_row.cumsum(dim=1)
    if init_count is not None:
        count = count + init_count.to(dev, torch.float32)[:, None, :]

    never = -1.0e6
    pos = torch.arange(S, device=dev, dtype=torch.float32)[None, :, None].expand(B, S, n_products)
    last = torch.where(per_row > 0, pos, torch.full_like(pos, never))
    last = torch.cummax(last, dim=1).values
    if init_last is not None:
        last = torch.maximum(last, init_last.to(dev, torch.float32)[:, None, :])
    rows_since = (pos - last).clamp(min=0.0)
    rows_since = torch.where(count > 0, rows_since, torch.zeros_like(rows_since))
    return count, rows_since



def _plain_rows(obtained_ids: torch.Tensor, dtype) -> torch.Tensor:
    """Unweighted acquisitions per occasion, (B,S,P)."""
    B, S, _ = obtained_ids.shape
    n_products = LAST_PROD_ID - FIRST_PROD_ID + 1
    valid = (obtained_ids >= FIRST_PROD_ID) & (obtained_ids <= LAST_PROD_ID)
    idx = (obtained_ids - FIRST_PROD_ID).clamp(0, n_products - 1)
    out = torch.zeros(B, S, n_products, device=obtained_ids.device, dtype=dtype)
    out.scatter_add_(2, idx, valid.to(dtype))
    return out


def copy_weights(obtained_ids: torch.Tensor, count_before: torch.Tensor,
                 g: torch.Tensor) -> torch.Tensor:
    """(B,S,P) acquisitions weighted by WHICH copy they are (R33's g).

    Copies acquired in the same occasion must count as 1st, 2nd, 3rd...; using
    the pre-occasion holding for all of them would over-weight duplicates,
    which is the whole point of g. `rank` counts earlier slots of the same row
    holding the same product. (This is the bug the R34 estimator had, caught by
    a brute-force replay -- see EXPERIMENTS.md.)
    """
    B, S, L = obtained_ids.shape
    n_products = LAST_PROD_ID - FIRST_PROD_ID + 1
    valid = (obtained_ids >= FIRST_PROD_ID) & (obtained_ids <= LAST_PROD_ID)
    idx = (obtained_ids - FIRST_PROD_ID).clamp(0, n_products - 1)
    same = (obtained_ids.unsqueeze(3) == obtained_ids.unsqueeze(2)) & valid.unsqueeze(2)
    earlier = torch.tril(torch.ones(L, L, device=obtained_ids.device, dtype=torch.bool), -1)
    rank = (same & earlier[None, None]).sum(dim=3)
    prior = torch.gather(count_before, 2, idx)
    k = (prior + rank).clamp(0, g.numel() - 1).long()
    w = g[k] * valid.to(g.dtype)
    out = torch.zeros(B, S, n_products, device=obtained_ids.device, dtype=g.dtype)
    out.scatter_add_(2, idx, w)
    return out


def decayed_counts(per_row: torch.Tensor, rho: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """
    R_t(p) = sum_{tau < t} w_tau(p) * rho^(t-tau)  -- Guadagni-Little smoothing (R33).

    Chunked so it is neither a 1024-step python loop nor a rho^-t overflow:
    inside a chunk the geometric weights are closed form (exponent < chunk),
    and only the carry between chunks is looped. rho -> 1 gives the plain
    cumulative count.
    """
    B, S, P = per_row.shape
    dev, dt = per_row.device, per_row.dtype
    nb = (S + chunk - 1) // chunk
    pad = nb * chunk - S
    x = torch.cat([per_row, per_row.new_zeros(B, pad, P)], dim=1) if pad else per_row
    x = x.view(B, nb, chunk, P)
    u = torch.arange(chunk, device=dev, dtype=torch.float32)
    r = rho.float()
    up = (r ** (-u)).to(dt)[None, None, :, None]
    dn = (r ** u).to(dt)[None, None, :, None]
    incl = torch.cumsum(x * up, dim=2)
    within = (incl - x * up) * dn
    tail = (x * (r ** (chunk - u)).to(dt)[None, None, :, None]).sum(dim=2)
    step = (r ** chunk).to(dt)
    outs, carry = [], per_row.new_zeros(B, P)
    for j in range(nb):
        outs.append(within[:, j] + carry[:, None, :] * dn[0, 0])
        carry = carry * step + tail[:, j]
    return torch.stack(outs, dim=1).reshape(B, nb * chunk, P)[:, :S]


class _SDPAttention(nn.Module):
    """Multi-head attention through F.scaled_dot_product_attention with a
    boolean keep-mask (True = may attend). Callers guarantee no query row is
    fully masked, which would otherwise produce NaN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.h, self.dh, self.p = n_heads, d_model // n_heads, dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv, keep):
        N, Lq, D = x_q.shape
        Lk = x_kv.size(1)
        q = self.q(x_q).view(N, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(x_kv).view(N, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(x_kv).view(N, Lk, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep[:, None, :, :],
            dropout_p=self.p if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(N, Lq, D))


class SatiationBlock(nn.Module):
    """One layer of offer-inventory computation (R24): the offers attend to each
    other (competition between concurrent banners), then to the inventory
    slots, then a feed-forward layer. Pre-norm residual throughout."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.self_attn = _SDPAttention(d_model, n_heads, dropout)
        self.cross_attn = _SDPAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, self_keep, mem, mem_keep):
        h = self.n1(x)
        x = x + self.drop(self.self_attn(h, h, self_keep))
        x = x + self.drop(self.cross_attn(self.n2(x), mem, mem_keep))
        return x + self.drop(self.ff(self.n3(x)))


class InventorySlots(nn.Module):
    """
    Additive inventory as a set of per-product slots (R23), read by the offers
    through either the original single attention step (sat_layers=0) or a
    stack of SatiationBlocks (sat_layers>=1, R24).

    Replaces two parts of the token path: the inventory GRU (gated, forgets,
    updates on every empty row) and the S x 10 token memory (median customer:
    643 tokens for 13 distinct products). A slot is the product's embedding
    plus its log count and log occasions since last acquired; only owned slots
    are attended. The attribute stock -- counts x the 34 product attributes --
    is McAlister's accumulated-attribute satiation in closed form.

    Returns z_sat, z_inv (pooled owned slots) and z_stock, each (B,S,D).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 feature_tensor: torch.Tensor, sat_layers: int = 0,
                 kernel: str = "none", decay: str = "none", tier: bool = False,
                 n_tiers: int = 3):
        super().__init__()
        self.sat_layers = int(sat_layers)
        # R33, the stock's WRITE side: `kernel` says which products an
        # acquisition touches, `decay` how it fades, `tier` how a duplicate
        # counts. R34 found decay and duplicate weights identified under our
        # offer rotation and a free kernel NOT identified, so "attr" (a few
        # attribute coefficients) is the structural choice and "learned" is
        # kept only as an upper bound for a specification test.
        self.kernel = str(kernel).lower()
        if self.kernel not in ("none", "attr", "learned"):
            raise ValueError(f"kernel must be none|attr|learned, got {kernel!r}")
        self.decay = str(decay).lower()
        if self.decay not in ("none", "exp"):
            raise ValueError(f"decay must be none|exp, got {decay!r}")
        self.use_tier = bool(tier)
        prod = feature_tensor[FIRST_PROD_ID:LAST_PROD_ID + 1].float()       # (44,F)
        std = prod.std(dim=0, keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        self.register_buffer("feat_std", (prod - prod.mean(dim=0, keepdim=True)) / std,
                             persistent=False)
        self.register_buffer("slot_ids", torch.arange(FIRST_PROD_ID, LAST_PROD_ID + 1),
                             persistent=False)
        self.state_mlp = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        n_prod = LAST_PROD_ID - FIRST_PROD_ID + 1
        if self.decay == "exp":
            self.log_half_life = nn.Parameter(torch.tensor(math.log(30.0)))
        if self.use_tier:
            self.tier_raw = nn.Parameter(torch.zeros(max(int(n_tiers) - 1, 1)))
        if self.kernel == "attr":
            # same-attribute indicators from the frozen feature table: the
            # McAlister restriction, a handful of coefficients instead of P^2
            feats = feature_tensor[FIRST_PROD_ID:LAST_PROD_ID + 1].float()
            self.register_buffer("attr_same",
                                 (feats[:, None, :] == feats[None, :, :]).float(),
                                 persistent=False)
            self.attr_w = nn.Parameter(torch.zeros(feats.size(1)))
            self.attr_self = nn.Parameter(torch.tensor(2.0))
        elif self.kernel == "learned":
            self.k_emb = nn.Parameter(torch.randn(n_prod, 32) * 0.1)
            self.k_q = nn.Linear(32, 32, bias=False)
            self.k_k = nn.Linear(32, 32, bias=False)
        self.slot_norm = nn.LayerNorm(d_model)
        self.inv_pool = AttentionPool(d_model, dropout)
        self.stock_proj = nn.Sequential(nn.Linear(prod.size(1) + 1, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))
        if self.sat_layers == 0:
            self.single = OfferInventoryCrossAttention(d_model, n_heads, dropout)
        else:
            self.null_slot = nn.Parameter(torch.zeros(1, 1, d_model))
            self.blocks = nn.ModuleList(SatiationBlock(d_model, n_heads, d_ff, dropout)
                                        for _ in range(self.sat_layers))
            self.out_norm = nn.LayerNorm(d_model)
            self.offer_out = nn.Linear(4 * d_model, d_model)

    def tier_weights(self, dtype, device) -> torch.Tensor:
        """g = [1, s1, s1*s2, ...]: weakly decreasing, first copy fixed at 1."""
        if not self.use_tier:
            return torch.ones(1, dtype=dtype, device=device)
        steps = torch.sigmoid(self.tier_raw).to(dtype)
        return torch.cat([torch.ones(1, dtype=dtype, device=device),
                          torch.cumprod(steps, 0)])

    def kernel_matrix(self, dtype, device):
        """Row-stochastic (P,P): which product an acquisition also satiates."""
        if self.kernel == "none":
            return None
        n = LAST_PROD_ID - FIRST_PROD_ID + 1
        eye = torch.eye(n, dtype=dtype, device=device)
        if self.kernel == "attr":
            logits = (self.attr_same.to(dtype) @ self.attr_w.to(dtype)
                      + self.attr_self.to(dtype) * eye)
        else:
            q, k = self.k_q(self.k_emb), self.k_k(self.k_emb)
            logits = (q @ k.transpose(0, 1) / math.sqrt(q.size(-1))).to(dtype)
        return torch.softmax(logits, dim=1)

    def forward(self, product_embed, offer_tok, offer_mask, obtained_ids,
                init_count=None, init_last=None):
        B, S, Lx, D = offer_tok.shape
        count, since = additive_inventory(obtained_ids, init_count, init_last)
        owned = count > 0                                                     # (B,S,44)

        stock_cnt = count
        if self.use_tier or self.decay == "exp" or self.kernel != "none":
            g = self.tier_weights(count.dtype, count.device)
            per_row = (copy_weights(obtained_ids, count.long(), g) if self.use_tier
                       else _plain_rows(obtained_ids, count.dtype))
            if self.decay == "exp":
                hl = F.softplus(self.log_half_life).clamp_min(1e-2)
                rho = torch.exp(-math.log(2.0) / hl)
                stock_cnt = decayed_counts(per_row, rho)
            else:
                stock_cnt = torch.cumsum(per_row, dim=1) - per_row
                if init_count is not None:
                    stock_cnt = stock_cnt + init_count.to(count.device, count.dtype)[:, None, :]
            kap = self.kernel_matrix(count.dtype, count.device)
            if kap is not None:
                stock_cnt = stock_cnt @ kap.transpose(0, 1)

        state = torch.stack([torch.log1p(stock_cnt.clamp_min(0)), torch.log1p(since),
                             owned.float()], dim=-1)
        base = product_embed(self.slot_ids)                                   # (44,D)
        slots = self.slot_norm(base[None, None] + self.state_mlp(state.to(base.dtype)))

        stock = stock_cnt @ self.feat_std                                         # (B,S,F)
        stock = torch.sign(stock) * torch.log1p(stock.abs())
        total = torch.log1p(count.sum(dim=-1, keepdim=True))
        z_stock = self.stock_proj(torch.cat([stock, total], dim=-1).to(base.dtype))
        z_inv = self.inv_pool(slots, owned)

        if self.sat_layers == 0:
            # Same single attention step as the token path, over slots instead
            # of tokens. Row t's slots already summarise occasions < t, so every
            # row may read its own slots: pass them as a length-1 "memory row"
            # per occasion by attending within the row.
            z_sat = self._single_step(offer_tok, offer_mask, slots, owned)
            return z_sat, z_inv, z_stock

        N = B * S
        x = offer_tok.reshape(N, Lx, D)
        xm = offer_mask.reshape(N, Lx)
        self_keep = xm[:, None, :] | torch.eye(Lx, dtype=torch.bool, device=x.device)[None]
        mem = torch.cat([self.null_slot.expand(N, 1, D), slots.reshape(N, -1, D)], dim=1)
        mk = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=x.device),
                        owned.reshape(N, -1)], dim=1)
        mem_keep = mk[:, None, :].expand(N, Lx, mk.size(1))
        for blk in self.blocks:
            x = blk(x, self_keep, mem, mem_keep)
        x = self.out_norm(x) * xm[..., None]
        z_sat = self.offer_out(x.reshape(B, S, Lx * D))
        return z_sat, z_inv, z_stock

    def _single_step(self, offer_tok, offer_mask, slots, owned):
        attn = self.single
        B, S, Lx, D = offer_tok.shape
        K = slots.size(2)
        q = attn.q_proj(offer_tok).view(B, S, Lx, attn.n_heads, attn.d_head)
        k = attn.k_proj(slots).view(B, S, K, attn.n_heads, attn.d_head)
        v = attn.v_proj(slots).view(B, S, K, attn.n_heads, attn.d_head)
        logits = torch.einsum("bslhd,bskhd->bshlk", q, k) / math.sqrt(attn.d_head)
        keep = owned[:, :, None, None, :]                                     # (B,S,1,1,K)
        logits = logits.masked_fill(~keep, -1e9)
        w = torch.softmax(logits, dim=-1).masked_fill(~keep, 0.0)
        w = attn.dropout(w)
        ctx = torch.einsum("bshlk,bskhd->bslhd", w, v).reshape(B, S, Lx, D)
        return attn.offer_context_pool(attn.out_proj(ctx), offer_mask)


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
        time_bias: str = "none",
    ):
        super().__init__()
        # ALiBi (Press et al. 2022): subtract slope_h * (i - j) from the
        # attention logit of query i on key j, with a fixed geometric slope per
        # head. No parameters. It gives attention a preference for RECENT
        # events that it otherwise lacks entirely -- this stack has no
        # positional encoding, so without it "one event ago" and "fifty events
        # ago" are indistinguishable except through the causal mask. A GRU has
        # recency built in; this is the cheapest way to hand it to attention.
        # time_bias: "none" | "ordinal" | "time".
        #   ordinal  ALiBi on event distance (i - j). Fixed slopes, no params.
        #   time     ALiBi on ELAPSED HOURS, log-compressed: the penalty is
        #            s_h * log1p(t_i - t_j). The event index here mixes two
        #            clocks -- a burst of pulls minutes apart and a three-week
        #            silence are both "one event ago" -- so ordinal distance is
        #            the wrong ruler even though it already helps.
        #
        # The slopes are LEARNABLE in time mode. log1p(hours) spans ~0-7.8 over
        # the observed range of gaps (max 2,401 h), against ~0-1024 for the
        # ordinal index, so the fixed ALiBi schedule is far too gentle on that
        # scale. Rather than guess a scale factor, the per-head decay rate is
        # learned, initialised at 16x the ALiBi values to put it in range.
        # softplus keeps it positive, so the bias can never reward distance.
        self.recency_bias = bool(recency_bias)
        self.time_bias = str(time_bias).lower()
        if self.time_bias not in ("none", "ordinal", "time"):
            raise ValueError(f"time_bias must be none|ordinal|time, got {time_bias!r}")
        if self.recency_bias and self.time_bias == "none":
            self.time_bias = "ordinal"      # --alibi is an alias for ordinal
        self.n_heads = int(n_heads)
        if self.time_bias == "ordinal":
            self.register_buffer("alibi_slopes", self._alibi_slopes(n_heads),
                                 persistent=False)
        elif self.time_bias == "time":
            init = torch.log(torch.expm1(self._alibi_slopes(n_heads) * 16.0))
            self.time_slopes_raw = nn.Parameter(init)
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

    def forward(self, r: torch.Tensor,
                event_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        r:          (B,S,D)
        event_time: (B,S) cumulative hours since the sequence start. Required
                    when time_bias == "time".
        returns:    (B,S,D)
        """
        B, S = r.size(0), r.size(1)
        if self.time_bias == "none":
            causal_mask = torch.triu(
                torch.ones(S, S, device=r.device, dtype=torch.bool),
                diagonal=1,
            )
            return self.norm(self.encoder(r, mask=causal_mask))

        # Additive float mask, one per (batch, head), batch-major as
        # nn.MultiheadAttention expects: index = b * n_heads + h.
        pos = torch.arange(S, device=r.device)
        causal = pos[None, :] > pos[:, None]                            # (S,S)

        if self.time_bias == "ordinal":
            dist = (pos[:, None] - pos[None, :]).clamp_min(0).to(r.dtype)  # (S,S)
            bias = -self.alibi_slopes.to(r.dtype)[:, None, None] * dist    # (H,S,S)
            bias = bias.masked_fill(causal, float("-inf"))
            mask = bias.unsqueeze(0).expand(B, -1, -1, -1)                 # (B,H,S,S)
        else:
            if event_time is None:
                raise ValueError(
                    "time_bias='time' needs event_time; the data file must "
                    "carry an IPT field and the trainer must pass it through.")
            t = event_time.to(r.dtype)                                     # (B,S)
            dt = (t[:, :, None] - t[:, None, :]).clamp_min(0)              # (B,S,S)
            slopes = F.softplus(self.time_slopes_raw).to(r.dtype)          # (H,)
            bias = -slopes[None, :, None, None] * torch.log1p(dt)[:, None] # (B,H,S,S)
            bias = bias.masked_fill(causal[None, None], float("-inf"))
            mask = bias

        # PyTorch's inference fast path (eval mode, no grad, NO autocast) does
        # not apply this per-head additive mask the way the training path does:
        # measured on torch 2.11, eval and train outputs differ by ~2.4 for both
        # the ordinal and time biases, and a changed gap alters only one row.
        # Under autocast the fast path is skipped and the two agree exactly,
        # which is why every bf16 HPCC evaluation so far is correct -- but a CPU
        # or fp32 evaluation would silently score a different model. Force the
        # standard path. Checked by scripts/test_time_bias_causality.py.
        fast = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            out = self.encoder(r, mask=mask.reshape(B * self.n_heads, S, S))
        finally:
            torch.backends.mha.set_fastpath_enabled(fast)
        return self.norm(out)


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
        use_user_embedding: bool = False,
        num_mix_heads: int = 0,
        return_attention_default: bool = False,
        encoder: str = "transformer",
        use_offer_inventory_attn: bool = True,
        attn_recency_bias: bool = False,
        attn_time_bias: str = "none",
        product_id_embed: bool = True,
        time_bias_lag_ipt: bool = True,
        inventory: str = "tokens",
        sat_layers: int = 0,
        block_len: int = 64,
        n_campaigns: int = 0,
        kernel: str = "none",
        decay: str = "none",
        tier: bool = False,
        fuse: str = "gate",
    ):
        super().__init__()
        # inventory: "tokens" = the original path (inventory GRU + attention over
        # every obtained token). "slots" = additive per-product inventory (R23),
        # read by one attention step (sat_layers=0) or stacked SatiationBlocks
        # (sat_layers>=1, R24). See InventorySlots.
        self.inventory_kind = str(inventory).lower()
        if self.inventory_kind not in ("tokens", "slots"):
            raise ValueError(f"inventory must be 'tokens' or 'slots', got {inventory!r}")
        if int(sat_layers) > 0 and self.inventory_kind != "slots":
            raise ValueError("sat_layers > 0 requires inventory='slots'")
        # LABEL LEAKAGE FIX (Sep 2026, EXPERIMENTS.md R21). IPT at row t is the
        # gap ENDING at row t, and rows exist because of outcomes: a draw makes
        # a row, a quiet day makes an inserted NotBuy at the 24-hour mark. So
        # IPT_t says what kind of row t is -- 0.40 nats of label information,
        # against 0.16 for IPT_{t-1} (scripts/ipt_leak_check.py). With the lag,
        # row t's clock reads the time of row t-1, so the bias uses only gaps
        # that had already closed. False reproduces batch 5 (leaky).
        self.time_bias_lag_ipt = bool(time_bias_lag_ipt)
        # Component ablation switches (EXPERIMENTS.md R18). Together with the
        # separate RecurrentBaseline they form a 2x2 over
        #   {sequence encoder: attention, GRU} x {offer-inventory cross-attn: on, off}
        # plus a recency-bias arm and a features-only-product arm, so each
        # piece of the architecture can be credited or blamed on its own.
        encoder = str(encoder).lower()
        if encoder not in ("transformer", "gru", "gru_attn"):
            raise ValueError(
                f"encoder must be 'transformer', 'gru' or 'gru_attn', got {encoder!r}")
        self.encoder_kind = encoder
        # gru_attn is Bahdanau/Luong in modern dress (EXPERIMENTS.md R25):
        # recurrence carries the decay prior, attention retrieves specific past
        # occasions by content. "gate" runs both over the same event
        # representations and blends them with a learned per-position gate;
        # "stack" interleaves them layer by layer. The attention half keeps
        # whatever recency bias the flags ask for, so "does recurrence already
        # supply recency?" is the bias switched off.
        # R32: five ways to combine the two mechanisms, each a different claim
        # about what memory does.
        #   gate    both branches on the same input, blended per position
        #   stack   interleaved layer by layer
        #   seq_ra  recurrence BUILDS the state, attention RETRIEVES among states
        #   seq_ar  attention builds context, recurrence CARRIES the propensity
        #   block   attention within a block of occasions, recurrence across
        #           blocks -- two time scales, and O(S*K) rather than O(S^2)
        self.fuse = str(fuse).lower()
        if self.fuse not in ("gate", "stack", "seq_ra", "seq_ar", "block"):
            raise ValueError(
                f"fuse must be gate|stack|seq_ra|seq_ar|block, got {fuse!r}")
        self.block_len = int(block_len)
        # Diagnostic, not a parameter: the mean gate weight on the recurrent
        # branch, so we can report how much the model leans on recency vs
        # retrieval. Updated under no_grad in forward.
        self.gate_mean = float("nan")
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

        if self.inventory_kind == "tokens":
            self.inventory_gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )
            self.offer_inventory_attn = (
                OfferInventoryCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
                if self.use_offer_inventory_attn else None)
            self.inventory_slots = None
        else:
            if not self.use_offer_inventory_attn:
                raise ValueError("inventory='slots' needs the offer-inventory attention on")
            self.inventory_gru = None
            self.offer_inventory_attn = None
            self.inventory_slots = InventorySlots(d_model, n_heads, d_ff, dropout,
                                                  feature_tensor, sat_layers,
                                                  kernel=kernel, decay=decay, tier=tier)

        # Campaign fixed effects (R33b). A calendar-level demand shock moves in
        # lockstep with "time since the product left the assortment", so without
        # this the decay half-life absorbs the calendar: R34 s6 measured 30 ->
        # 59 -> 147 as the shock grew, and 30 throughout once this is on.
        # Applied to the eight purchase logits only, never to NotBuy, which is
        # the baseline alternative.
        self.camp_bias = nn.Embedding(int(n_campaigns), 1) if n_campaigns else None
        if self.camp_bias is not None:
            nn.init.zeros_(self.camp_bias.weight)

        self.use_user_embedding = bool(use_user_embedding and num_users is not None)
        if self.use_user_embedding:
            self.user_embed = nn.Embedding(num_users, d_model)
        else:
            self.user_embed = None

        # Event fusion: [z_x, z_sat, z_o, z_y, h_H] plus optional user embedding.
        n_pieces = 5 if self.use_offer_inventory_attn else 4   # z_x, [z_sat], z_o, z_y, h_H
        if self.inventory_kind == "slots":
            n_pieces = 6                                        # z_x, z_sat, z_o, z_y, z_inv, z_stock
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
                time_bias=attn_time_bias,
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

            if self.encoder_kind == "gru_attn":
                n_attn = n_layers if self.fuse in ("stack", "block") else 1
                self.attn_branch = nn.ModuleList(
                    CausalEventTransformer(
                        d_model=d_model, n_layers=1, n_heads=n_heads, d_ff=d_ff,
                        dropout=dropout, recency_bias=attn_recency_bias,
                        time_bias=attn_time_bias)
                    for _ in range(n_attn))
                if self.fuse == "gate":
                    # One gate per position and channel, from both branches.
                    self.fuse_gate = nn.Linear(2 * d_model, d_model)
                    self.fuse_norm = nn.LayerNorm(d_model)

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


    def _block_recurrent(self, r: torch.Tensor, event_time) -> torch.Tensor:
        """Attention WITHIN a block of occasions, recurrence ACROSS blocks (R32).

        Two time scales: retrieval inside a campaign-sized window, carry-over
        between windows. Causality holds in both directions -- attention is
        causal inside its block, and block j only ever receives the recurrent
        state summarising blocks < j. Cost is O(S*K), not O(S^2), which is what
        would let max_events rise above 1024.
        """
        B, S, D = r.shape
        K = max(int(self.block_len), 1)
        nb = (S + K - 1) // K
        pad = nb * K - S
        x = torch.cat([r, r.new_zeros(B, pad, D)], dim=1) if pad else r
        et = None
        if event_time is not None:
            et = event_time
            if pad:
                et = torch.cat([et, et[:, -1:].expand(B, pad)], dim=1)
            et = et.reshape(B * nb, K)
        y = x.reshape(B * nb, K, D)
        last = len(self.attn_branch) - 1
        for i, blk in enumerate(self.attn_branch):
            y = y + blk(y, et)
            if i < last:
                y = self.event_model_dropout(y)
        y = y.reshape(B, nb, K, D)
        summary = y.mean(dim=2)                                   # (B, nb, D)
        h = summary.contiguous()
        lastg = len(self.event_model) - 1
        for i, gru in enumerate(self.event_model):
            h, _ = gru(h)
            if i < lastg:
                h = self.event_model_dropout(h).contiguous()
        h = self.event_model_norm(h)
        carry = torch.cat([h.new_zeros(B, 1, D), h[:, :-1]], dim=1)   # blocks < j only
        s = (y + carry[:, :, None, :]).reshape(B, nb * K, D)
        return s[:, :S]

    def forward(
        self,
        lto_ids: torch.Tensor,
        obtained_ids: Optional[torch.Tensor] = None,
        prev_dec_ids: Optional[torch.Tensor] = None,
        user_idx: Optional[torch.Tensor] = None,
        ipt: Optional[torch.Tensor] = None,
        campaign: Optional[torch.Tensor] = None,
        projection_gate_mode: Optional[str] = None,
        inv_init_count: Optional[torch.Tensor] = None,
        inv_init_last: Optional[torch.Tensor] = None,
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

        # 3-4 (slots). Additive inventory, read by the current offer (R23/R24).
        if self.inventory_kind == "slots":
            z_sat, z_inv, z_stock = self.inventory_slots(
                self.product_embed, lto_tok, lto_mask, obtained_ids,
                inv_init_count, inv_init_last)
            sat_attn = None
            h_H = None
        # 3. Latent inventory state from immediate outcomes.
        else:
            h_H, _ = self.inventory_gru(z_o)        # (B,S,D)

        # 4. Current offer attends to cumulative inventory tokens.
        if self.inventory_kind == "slots":
            pass
        elif self.offer_inventory_attn is None:
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
        if self.inventory_kind == "slots":
            pieces = [z_x, z_sat, z_o, z_y, z_inv, z_stock]
        else:
            pieces = [z_x, z_o, z_y, h_H] if z_sat is None else [z_x, z_sat, z_o, z_y, h_H]

        if self.use_user_embedding and user_idx is not None:
            z_u = self.user_embed(user_idx.long())                 # (B,D)
            z_u = z_u.unsqueeze(1).expand(-1, z_x.size(1), -1)     # (B,S,D)
            pieces.append(z_u)

        r = self.event_fusion(torch.cat(pieces, dim=-1))           # (B,S,D)

        # 6. Causal sequence model over event representations.
        if self.encoder_kind == "transformer":
            # IPT is hours since the PREVIOUS ROW, so the cumulative sum is the
            # elapsed time of each row since the sequence start. Inserted
            # NotBuy rows are rows, so within this discrete representation the
            # running total is correct -- the censoring problem noted in R9
            # only bites if inserted rows are dropped.
            event_time = None
            if ipt is not None:
                if self.time_bias_lag_ipt:
                    # Row t carries IPT_{t-1}; see time_bias_lag_ipt in __init__.
                    ipt = torch.cat([torch.zeros_like(ipt[:, :1]), ipt[:, :-1]], dim=1)
                event_time = ipt.cumsum(dim=1)
            s = self.event_model(r, event_time)                    # (B,S,D)
        elif self.encoder_kind == "gru":
            s = r.contiguous()
            last = len(self.event_model) - 1
            for i, gru in enumerate(self.event_model):
                s, _ = gru(s)
                if i < last:
                    s = self.event_model_dropout(s).contiguous()
            s = self.event_model_norm(s)
        else:
            # gru_attn. event_time is only needed by a calendar-time bias.
            event_time = None
            if ipt is not None and self.attn_branch[0].time_bias == "time":
                lag = ipt
                if self.time_bias_lag_ipt:
                    lag = torch.cat([torch.zeros_like(ipt[:, :1]), ipt[:, :-1]], dim=1)
                event_time = lag.cumsum(dim=1)

            if self.fuse == "stack":
                s = r.contiguous()
                last = len(self.event_model) - 1
                for i, gru in enumerate(self.event_model):
                    s, _ = gru(s)
                    s = s + self.attn_branch[i](s, event_time)
                    if i < last:
                        s = self.event_model_dropout(s).contiguous()
                s = self.event_model_norm(s)
            elif self.fuse == "seq_ra":
                # recurrence first, attention over the hidden states it built
                h = r.contiguous()
                last = len(self.event_model) - 1
                for i, gru in enumerate(self.event_model):
                    h, _ = gru(h)
                    if i < last:
                        h = self.event_model_dropout(h).contiguous()
                h = self.event_model_norm(h)
                s = h + self.attn_branch[0](h, event_time)
            elif self.fuse == "seq_ar":
                # attention first, recurrence reads out the contextualised sequence
                a = r + self.attn_branch[0](r, event_time)
                s = a.contiguous()
                last = len(self.event_model) - 1
                for i, gru in enumerate(self.event_model):
                    s, _ = gru(s)
                    if i < last:
                        s = self.event_model_dropout(s).contiguous()
                s = self.event_model_norm(s)
            elif self.fuse == "block":
                s = self._block_recurrent(r, event_time)
            else:
                h = r.contiguous()
                last = len(self.event_model) - 1
                for i, gru in enumerate(self.event_model):
                    h, _ = gru(h)
                    if i < last:
                        h = self.event_model_dropout(h).contiguous()
                h = self.event_model_norm(h)
                a = self.attn_branch[0](r, event_time)
                g = torch.sigmoid(self.fuse_gate(torch.cat([h, a], dim=-1)))
                with torch.no_grad():
                    self.gate_mean = float(g.float().mean())
                s = self.fuse_norm(g * h + (1.0 - g) * a)

        # 7. Decision logits.
        h = self.head_trunk(s)
        if self.mixture_head is not None:
            logits = self.mixture_head(h, user_idx)                # (B,S,V)
        else:
            logits = self.output_head(h)                           # (B,S,V)

        if self.camp_bias is not None and campaign is not None:
            c = campaign.clamp(0, self.camp_bias.num_embeddings - 1).long()
            logits = logits.clone()
            logits[..., 1:9] = logits[..., 1:9] + self.camp_bias(c)

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
        use_user_embedding=kwargs.get("use_user_embedding", False),
        num_mix_heads=kwargs.get("num_mix_heads", 0),
        return_attention_default=kwargs.get("return_attention_default", False),
        encoder=kwargs.get("encoder", "transformer"),
        use_offer_inventory_attn=kwargs.get("use_offer_inventory_attn", True),
        attn_recency_bias=kwargs.get("attn_recency_bias", False),
        attn_time_bias=kwargs.get("attn_time_bias", "none"),
        product_id_embed=kwargs.get("product_id_embed", True),
        time_bias_lag_ipt=kwargs.get("time_bias_lag_ipt", True),
        inventory=kwargs.get("inventory", "tokens"),
        sat_layers=kwargs.get("sat_layers", 0),
        block_len=kwargs.get("block_len", 64),
        n_campaigns=kwargs.get("n_campaigns", 0),
        kernel=kwargs.get("kernel", "none"),
        decay=kwargs.get("decay", "none"),
        tier=kwargs.get("tier", False),
        fuse=kwargs.get("fuse", "gate"),
    )
