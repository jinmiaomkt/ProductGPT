"""
Generation 6: the streaming hybrid (Oct 2026).

WHAT IT IS
----------
The gen-5 stack design (GRU and attention interleaved layer by layer, slots
inventory read by SatiationBlocks -- the b36 incumbent), rebuilt so that the
length of a customer's history never matters:

  computed state   inventory counts, occasions since each product was last
                   acquired, the R33 stock and (optionally) pity depth, all
                   computed over the customer's ENTIRE history -- never a
                   window, so nothing acquired in campaign 1 is forgotten in
                   campaign 28.
  learned state    each GRU layer's hidden state, carried from the first
                   occasion to the last.
  local attention  each attention layer sees only the last `window` occasions
                   (a sliding window, so no block edges), with the same ALiBi
                   slopes as gen 5.

and processed in CHUNKS of `chunk` occasions. GRU states and the attention
key/value caches are carried from one chunk to the next, so GPU memory depends
on the chunk length and the batch size, not on how long a customer is.
Training stops gradients at chunk boundaries (truncated back-propagation
through time); the computed state needs no gradient carried, except the stock,
whose gradient is routed back exactly (see stream_forward).

The input side -- token lookup, the three stream pools, the slots read, the
event fusion and the head -- is gen 5's own modules, inherited unchanged, so
the parameter names are identical and a gen-5 stack checkpoint loads into
this class. That is what the equivalence tests use:

  E1  window and chunk >= sequence length  ->  identical to gen-5 fuse=stack.
  E2  any chunk length gives the same output in eval mode.
  E3  one occasion at a time (generation) gives the same output as one pass.

See scripts/test_gen6_streaming.py.

WHAT IT DOES NOT SUPPORT
------------------------
Only the configuration it exists for: inventory=slots, encoder=gru_attn,
fuse=stack, time bias none or ordinal (ALiBi), model_rev >= 2, the obtained
stream on. The calendar-time bias would need a carried clock and lost to the
ordinal ruler in R21; the duplet has no inventory to carry.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_multistream_state_space as mm

NEVER = -1.0e6
STOCK_FLOOR = 1e-3      # R47 stock_only: holdings below this count as not owned


def _mult_scan(w: torch.Tensor, a: torch.Tensor, d0: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """D_t = a_t * (D_(t-1) + w_t), exactly, for a per-row factor a_t (B,L).

    Closed form inside chunks of `chunk` rows, in float64 so the cumulative
    log-factors cannot overflow; only the carry between chunks is looped.
    w (B,L,P), d0 (B,P). Returns (B,L,P) in w's dtype."""
    B, L, P = w.shape
    w64 = w.double()
    la = torch.log(a.clamp_min(1e-300))
    out, d = [], d0.double()
    for c0 in range(0, L, chunk):
        c1 = min(L, c0 + chunk)
        lc = la[:, c0:c1].cumsum(dim=1)                                 # l_t, inclusive
        lprev = torch.cat([torch.zeros_like(lc[:, :1]), lc[:, :-1]], dim=1)
        inner = torch.cumsum(w64[:, c0:c1] * torch.exp(-lprev)[..., None], dim=1)
        dc = torch.exp(lc)[..., None] * (d[:, None, :] + inner)
        out.append(dc)
        d = dc[:, -1]
    return torch.cat(out, dim=1).to(w.dtype)


class StreamingHybrid(mm.MultiStreamStateSpaceTransformer):
    """Gen-5 stack hybrid with full-history computed state, carried recurrent
    state and sliding-window attention, run in chunks."""

    streaming = True

    def __init__(self, *, window: int = 384, chunk: int = 512, censor_flag: bool = False,
                 slot_state: str = "full", camp_decay: float = 1.0, **kw):
        kw = dict(kw)
        for k, want in (("encoder", "gru_attn"), ("fuse", "stack"), ("inventory", "slots")):
            got = str(kw.get(k, want)).lower()
            if got != want:
                raise ValueError(f"gen 6 is the {want} design only; got {k}={got!r}")
            kw[k] = want
        if str(kw.get("attn_time_bias", "none")).lower() == "time":
            raise ValueError("gen 6 supports time bias none|ordinal; 'time' needs a carried "
                             "clock and lost to the ordinal ruler in R21")
        if not kw.get("use_obtained", True):
            raise ValueError("gen 6 needs the obtained stream (the duplet has no inventory)")
        if int(kw.get("model_rev", mm.MODEL_REV)) < 2:
            raise ValueError("gen 6 is model_rev >= 2 only (the rev-1 stock timing is a bug)")
        super().__init__(**kw)
        # window: occasions each attention query sees, itself included.
        # 0 or None = unlimited (full causal attention; the E1 control).
        self.window = int(window or 0)
        self.chunk = max(int(chunk), 1)
        self.gen = 6
        # R45: tell the model a history is left-censored. A learned vector added
        # to every occasion of a censored customer, and a learned starting state
        # for each GRU layer in place of zeros. Both start at zero, so a fresh
        # model computes the uncensored function, and uncensored customers never
        # see them. Created after the parent's reset_parameters, so they stay zero.
        # R47: forgetting. slot_state="stock_only" removes the never-decaying
        # channels -- the count (hence the owned flag and total holdings) becomes
        # the decayed stock, thresholded at STOCK_FLOOR, and "occasions since last
        # acquired" is zeroed -- so holdings enter the slots only through the
        # decayed stock. camp_decay = delta multiplies the stock at the first row
        # of every new campaign: stock_t = rho * delta_t * (stock_(t-1) + copies_t).
        # delta = 1 is the existing stock exactly.
        self.slot_state = str(slot_state).lower()
        if self.slot_state not in ("full", "stock_only"):
            raise ValueError(f"slot_state must be full|stock_only, got {slot_state!r}")
        self.camp_decay = float(camp_decay)
        if not 0.0 <= self.camp_decay <= 1.0:
            raise ValueError(f"camp_decay must be in [0, 1], got {camp_decay}")
        if self.camp_decay < 1.0 and self.inventory_slots.decay != "exp" and not (
                self.inventory_slots.use_tier or self.inventory_slots.kernel != "none"):
            raise ValueError("camp_decay < 1 needs a stock (decay, tier or kernel on)")
        self.censor_flag = bool(censor_flag)
        if self.censor_flag:
            d = int(kw["d_model"])
            self.censor_vec = nn.Parameter(torch.zeros(d))
            self.censor_h0 = nn.Parameter(torch.zeros(len(self.event_model), d))

    # ------------------------------------------------------------ state
    def init_stream(self, batch_size: int, device) -> Dict[str, Any]:
        """Empty state for `batch_size` customers at their first occasion."""
        P = mm.LAST_PROD_ID - mm.FIRST_PROD_ID + 1
        z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)  # noqa: E731
        return {
            "t": 0,
            "count": z(batch_size, P),
            "last": torch.full((batch_size, P), NEVER, device=device, dtype=torch.float32),
            "stock": z(batch_size, P),          # pre-kernel stock D_t
            "pity": z(batch_size, 8),           # raw pulls since last hit, per banner x tier
            "camp": torch.zeros(batch_size, device=device, dtype=torch.long),  # last campaign seen
            "h": [None] * len(self.event_model),
            "k": [None] * len(self.event_model),
            "v": [None] * len(self.event_model),
        }

    @staticmethod
    def detach_state(st: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(st)
        for key in ("h", "k", "v"):
            out[key] = [x.detach() if x is not None else None for x in st[key]]
        for key in ("count", "last", "stock", "pity", "camp"):
            if key in st:
                out[key] = st[key].detach()
        return out

    # ------------------------------------------------------- computed state
    def computed_scan(self, obtained_ids: torch.Tensor, prev_dec_ids: torch.Tensor,
                      st: Dict[str, Any], detach_carry: bool = True, campaign=None):
        """Computed state for rows st['t'] .. st['t']+L-1, continuing from `st`.

        obtained_ids (B,L,10), prev_dec_ids (B,L). Row t carries o_(t-1) (the
        R1 shift), so every quantity at row t uses acquisitions up to t-1 only,
        exactly as in gen 5. Returns (count, since, stock, pity, carry) where
        stock is the post-kernel stock and carry the state after the last row.

        With an empty `st` and the full sequence this reproduces gen 5's
        additive_inventory, InventorySlots.stock_counts and pity_depth; in
        pieces it reproduces the same numbers (tests T1/T2).
        """
        B, L, _ = obtained_ids.shape
        dev = obtained_ids.device
        slots = self.inventory_slots
        plain = mm._plain_rows(obtained_ids, torch.float32)                 # (B,L,P)
        count = st["count"][:, None, :] + plain.cumsum(dim=1)
        pos = (st["t"] + torch.arange(L, device=dev, dtype=torch.float32))[None, :, None]
        last = torch.where(plain > 0, pos.expand_as(plain), torch.full_like(plain, NEVER))
        last = torch.maximum(torch.cummax(last, dim=1).values, st["last"][:, None, :])
        since = (pos - last).clamp(min=0.0)
        since = torch.where(count > 0, since, torch.zeros_like(since))

        options = slots.use_tier or slots.decay == "exp" or slots.kernel != "none"
        if not options:
            stock_pre = count
        else:
            g = slots.tier_weights(count.dtype, count.device)
            if slots.use_tier:
                before = count - plain
                w = mm.copy_weights(obtained_ids, before.long(), g)
            else:
                w = plain
            rho = None
            if slots.decay == "exp":
                hl = F.softplus(slots.half_life_raw).clamp_min(1e-2)
                rho = torch.exp(-math.log(2.0) / hl)
            if self.camp_decay < 1.0:
                # R47: per-row factor a_t = rho * delta_t, with delta_t = camp_decay
                # at the first row of a new campaign (padding rows, campaign 0,
                # never count as a boundary).
                if campaign is None:
                    raise ValueError("camp_decay < 1 needs the campaign of every row")
                cam = campaign.long()
                prev_c = torch.cat([st["camp"][:, None], cam[:, :-1]], dim=1)
                boundary = (cam != prev_c) & (cam > 0) & (prev_c > 0)
                a = torch.where(boundary, torch.full_like(cam, 1, dtype=torch.float64)
                                * max(self.camp_decay, 1e-12), torch.ones_like(cam, dtype=torch.float64))
                if rho is not None:
                    a = a * rho.double()
                stock_pre = _mult_scan(w, a, st["stock"])
            elif rho is not None:
                within = mm.decayed_counts(w, rho)               # sum_{tau<t in chunk} w rho^(t-tau)
                k = torch.arange(1, L + 1, device=dev, dtype=torch.float32)
                lead = (rho.float() ** k).to(w.dtype)[None, :, None]
                stock_pre = rho * (within + w) + lead * st["stock"][:, None, :]
            else:
                stock_pre = st["stock"][:, None, :] + torch.cumsum(w, dim=1)
        kap = slots.kernel_matrix(count.dtype, count.device)
        stock = stock_pre @ kap.transpose(0, 1) if kap is not None else stock_pre

        pity, pity_carry = None, st["pity"]
        if self.pity_stream:
            pity, pity_carry = self._pity_scan(obtained_ids, prev_dec_ids, st["pity"])

        camp_carry = st["camp"]
        if campaign is not None:
            camp_carry = torch.maximum(st["camp"], campaign.long().max(dim=1).values)
        carry = {"count": count[:, -1].detach(), "last": last[:, -1].detach(),
                 "stock": stock_pre[:, -1] if options else count[:, -1],
                 "pity": pity_carry.detach(), "camp": camp_carry.detach()}
        if detach_carry:
            carry["stock"] = carry["stock"].detach()
        if self.slot_state == "stock_only":
            # The slots see holdings only through the decayed (pre-kernel) stock:
            # it replaces the count, so "owned" and total holdings decay too.
            # The true count stays in `carry` for the duplicate ranks.
            count = torch.where(stock_pre > STOCK_FLOOR, stock_pre, torch.zeros_like(stock_pre))
            since = torch.zeros_like(since)
        return count, since, stock, pity, carry

    def _pity_scan(self, obtained_ids, prev_dec_ids, carry):
        """mm.pity_depth with a carried count of pulls since the last hit."""
        B, L, _ = obtained_ids.shape
        dev = obtained_ids.device
        rarity_of = self.rarity_of
        valid = (obtained_ids >= mm.FIRST_PROD_ID) & (obtained_ids <= mm.LAST_PROD_ID)
        rar = rarity_of[obtained_ids.clamp(0, rarity_of.numel() - 1)]
        n_pull = valid.sum(-1).float()
        is5 = ((rar >= 4.5) & valid).any(-1)
        is4 = ((rar >= 3.5) & valid).any(-1)
        dec = prev_dec_ids.clamp(0, 9).long()
        banner = torch.tensor(mm.BANNER_OF_DEC, device=dev, dtype=torch.long)[dec]
        idx = torch.arange(L, device=dev).view(1, L).expand(B, L)
        out, new = [], []
        j = 0
        for b in range(4):
            on = banner == b
            cum = torch.where(on, n_pull, torch.zeros_like(n_pull)).cumsum(dim=1)
            for hit, cap in ((is5 & on, mm.HARD_PITY5[b]), (is4 & on, mm.HARD_PITY4[b])):
                lasti = torch.cummax(torch.where(hit, idx, torch.full_like(idx, -1)), dim=1).values
                cum_at = torch.gather(cum, 1, lasti.clamp(min=0))
                raw = torch.where(lasti < 0, carry[:, j:j + 1] + cum, cum - cum_at)
                out.append((raw / cap).clamp(0.0, 4.0))
                new.append(raw[:, -1])
                j += 1
        return torch.stack(out, dim=-1), torch.stack(new, dim=-1)

    # --------------------------------------------------------- row pipeline
    def _rows(self, lto_ids, obtained_ids, prev_dec_ids, user_idx, count, since, stock, pity,
              censored=None):
        """Event representations r (B,L,D) from row-local inputs and the
        computed state of those rows. Gen 5's steps 1-5, slots path."""
        lto_ids, obtained_ids, prev_dec_ids = lto_ids.long(), obtained_ids.long(), prev_dec_ids.long()
        lto_mask = lto_ids.ne(mm.PAD_ID)
        out_mask = obtained_ids.ne(mm.PAD_ID)
        lto_tok = self.product_embed(lto_ids)
        out_tok = self.product_embed(obtained_ids)
        z_x = self.offer_pool(lto_tok, lto_mask)
        z_o = self.outcome_pool(out_tok, out_mask)
        z_y = self.decision_embed(prev_dec_ids)
        z_sat, z_inv, z_stock = self._slots_read(lto_tok, lto_mask, count, since, stock)
        pieces = [z_x, z_sat, z_o, z_y, z_inv, z_stock]
        if self.pity_stream:
            pieces.append(self.pity_proj(pity.to(pieces[0].dtype)))
        if self.use_user_embedding and user_idx is not None:
            z_u = self.user_embed(user_idx.long()).unsqueeze(1).expand(-1, z_x.size(1), -1)
            pieces.append(z_u)
        r = self.event_fusion(torch.cat(pieces, dim=-1))
        if self.censor_flag and censored is not None:
            r = r + censored.to(r.device).to(r.dtype)[:, None, None] * self.censor_vec.to(r.dtype)
        return r

    def _slots_read(self, offer_tok, offer_mask, count, since, stock_cnt):
        """InventorySlots.forward from its state onward (the READ side)."""
        S_ = self.inventory_slots
        B, L, Lx, D = offer_tok.shape
        owned = count > 0
        state = torch.stack([torch.log1p(stock_cnt.clamp_min(0)), torch.log1p(since),
                             owned.float()], dim=-1)
        base = self.product_embed(S_.slot_ids)
        slots = S_.slot_norm(base[None, None] + S_.state_mlp(state.to(base.dtype)))
        stock = stock_cnt @ S_.feat_std
        stock = torch.sign(stock) * torch.log1p(stock.abs())
        total = torch.log1p(count.sum(dim=-1, keepdim=True))
        z_stock = S_.stock_proj(torch.cat([stock, total], dim=-1).to(base.dtype))
        z_inv = S_.inv_pool(slots, owned)
        if S_.sat_layers == 0:
            return S_._single_step(offer_tok, offer_mask, slots, owned), z_inv, z_stock
        N = B * L
        x = offer_tok.reshape(N, Lx, D)
        xm = offer_mask.reshape(N, Lx)
        self_keep = xm[:, None, :] | torch.eye(Lx, dtype=torch.bool, device=x.device)[None]
        mem = torch.cat([S_.null_slot.expand(N, 1, D), slots.reshape(N, -1, D)], dim=1)
        mk = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=x.device),
                        owned.reshape(N, -1)], dim=1)
        mem_keep = mk[:, None, :].expand(N, Lx, mk.size(1))
        for blk in S_.blocks:
            x = blk(x, self_keep, mem, mem_keep)
        x = S_.out_norm(x) * xm[..., None]
        return S_.offer_out(x.reshape(B, L, Lx * D)), z_inv, z_stock

    # -------------------------------------------------------------- encoder
    def _local_attn(self, i: int, x: torch.Tensor, st: Dict[str, Any], t0: int):
        """Attention layer i over a sliding window, with a key/value cache.

        Replicates CausalEventTransformer(n_layers=1): a pre-norm
        nn.TransformerEncoderLayer followed by the branch's LayerNorm, using
        that layer's own parameters. Query at absolute occasion p sees keys at
        p - window + 1 .. p (or all keys <= p when window is unlimited).
        """
        blk = self.attn_branch[i]
        layer = blk.encoder.layers[0]
        mha = layer.self_attn
        B, L, D = x.shape
        H = mha.num_heads
        dh = D // H
        h = layer.norm1(x)
        Wq, Wk, Wv = mha.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
        q = F.linear(h, Wq, bq).view(B, L, H, dh).transpose(1, 2)
        k = F.linear(h, Wk, bk).view(B, L, H, dh).transpose(1, 2)
        v = F.linear(h, Wv, bv).view(B, L, H, dh).transpose(1, 2)
        if st["k"][i] is not None:
            k = torch.cat([st["k"][i], k], dim=2)
            v = torch.cat([st["v"][i], v], dim=2)
        M = k.size(2) - L                                     # cached rows
        dev = x.device
        pq = torch.arange(t0, t0 + L, device=dev)
        pk = torch.arange(t0 - M, t0 + L, device=dev)
        dist = pq[:, None] - pk[None, :]                       # (L, M+L)
        ok = dist >= 0
        if self.window > 0:
            ok = ok & (dist < self.window)
        if blk.time_bias == "ordinal":
            bias = -blk.alibi_slopes.to(q.dtype)[:, None, None] * dist.clamp_min(0).to(q.dtype)
        else:
            bias = torch.zeros(H, L, M + L, device=dev, dtype=q.dtype)
        bias = bias.masked_fill(~ok[None], float("-inf"))
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias[None],
            dropout_p=mha.dropout if self.training else 0.0)
        a = mha.out_proj(a.transpose(1, 2).reshape(B, L, D))
        y = x + layer.dropout1(a)
        y = y + layer.dropout2(layer.linear2(layer.dropout(layer.activation(
            layer.linear1(layer.norm2(y))))))
        keep = (self.window - 1) if self.window > 0 else k.size(2)
        if keep > 0:
            nk, nv = k[:, :, -keep:], v[:, :, -keep:]
        else:
            nk = nv = None
        return blk.norm(y), nk, nv

    def _encode(self, r: torch.Tensor, st: Dict[str, Any], t0: int):
        """Gen 5's fuse=stack encoder, with carried GRU states and KV caches."""
        s = r.contiguous()
        hs, ks, vs = [], [], []
        last = len(self.event_model) - 1
        for i, gru in enumerate(self.event_model):
            if st["h"][i] is None:
                s, hn = gru(s)
            else:
                s, hn = gru(s, st["h"][i].to(s.dtype))
            a, nk, nv = self._local_attn(i, s, st, t0)
            s = s + a
            hs.append(hn)
            ks.append(nk)
            vs.append(nv)
            if i < last:
                s = self.event_model_dropout(s).contiguous()
        return self.event_model_norm(s), hs, ks, vs

    def _head(self, s, user_idx, campaign):
        h = self.head_trunk(s)
        logits = self.mixture_head(h, user_idx) if self.mixture_head is not None else self.output_head(h)
        if self.camp_bias is not None and campaign is not None:
            c = campaign.clamp(0, self.camp_bias.num_embeddings - 1).long()
            logits = logits.clone()
            logits[..., 1:9] = logits[..., 1:9] + self.camp_bias(c)
        return logits

    # ------------------------------------------------------------- passes
    def advance(self, lto_ids, obtained_ids, prev_dec_ids, st: Dict[str, Any],
                user_idx=None, campaign=None):
        """Process rows st['t'] .. st['t']+L-1 from a carried state.

        The generation path: call it once on the real prefix, then once per
        occasion. Returns (logits (B,L,V), new state). No gradient is carried
        through the computed state here; use stream_forward to train.
        """
        L = lto_ids.size(1)
        t0 = int(st["t"])
        count, since, stock, pity, carry = self.computed_scan(obtained_ids.long(),
                                                              prev_dec_ids.long(), st,
                                                              campaign=campaign)
        r = self._rows(lto_ids, obtained_ids, prev_dec_ids, user_idx, count, since, stock, pity)
        s, hs, ks, vs = self._encode(r, st, t0)
        logits = self._head(s, user_idx, campaign)
        new = {"t": t0 + L, "h": hs, "k": ks, "v": vs, **carry}
        return logits, new

    def stream_forward(self, lto_ids, obtained_ids, prev_dec_ids, user_idx=None,
                       campaign=None, chunk: Optional[int] = None,
                       on_chunk: Optional[Callable[[int, int, torch.Tensor], None]] = None,
                       prefix_obtained=None, prefix_prev=None, censored=None):
        """Full-history pass over (B,S) rows, `chunk` occasions at a time.

        Without on_chunk, returns the logits (B,S,V). With it, calls
        on_chunk(t0, t1, logits_chunk) for each chunk -- the trainer computes
        and back-propagates that chunk's loss there -- and returns None.

        Gradients stop at chunk boundaries for the GRU states and the
        attention caches (truncated BPTT). The computed state is computed once
        over the whole history; the stock, which depends on learned structural
        parameters (tier weights, attribute kernel, half-life), is handed to
        the chunks as a detached leaf, and after the last chunk its
        accumulated gradient is pushed back through the stock computation, so
        those parameters receive their exact gradient.

        R45 (left-censoring). `prefix_obtained` / `prefix_prev` (B,Lp), LEFT-padded
        so every prefix ends at row -1, are rows removed from the model's context
        but handed to the computed state: inventory, stock and pity then enter
        row 0 exactly as in the full history (gradient kept through the stock).
        `censored` (B,) feeds the censoring flag when the model has one.
        """
        C = max(int(chunk or self.chunk), 1)
        lto_ids, obtained_ids, prev_dec_ids = lto_ids.long(), obtained_ids.long(), prev_dec_ids.long()
        B, S = prev_dec_ids.shape
        st0 = self.init_stream(B, lto_ids.device)
        if prefix_obtained is not None and prefix_obtained.size(1) > 0:
            if self.camp_decay < 1.0:
                raise ValueError("R45 prefixes carry no campaigns; camp_decay < 1 cannot use them")
            Lp = int(prefix_obtained.size(1))
            _, _, _, _, carry = self.computed_scan(prefix_obtained.long(), prefix_prev.long(),
                                                   {**st0, "t": -Lp}, detach_carry=False)
            st0 = {**st0, **carry}
        if self.censor_flag and censored is not None:
            c = censored.to(lto_ids.device).float()
            st0["h"] = [(c[:, None] * self.censor_h0[i][None, :])[None].contiguous()
                        for i in range(len(self.event_model))]
        count, since, stock, pity, _ = self.computed_scan(obtained_ids, prev_dec_ids, st0,
                                                          campaign=campaign)
        track = torch.is_grad_enabled() and stock.requires_grad
        stock_leaf = stock.detach().requires_grad_(True) if track else stock

        st = st0
        outs: List[torch.Tensor] = []
        for t0 in range(0, S, C):
            t1 = min(S, t0 + C)
            sl = slice(t0, t1)
            r = self._rows(lto_ids[:, sl], obtained_ids[:, sl], prev_dec_ids[:, sl], user_idx,
                           count[:, sl], since[:, sl], stock_leaf[:, sl],
                           pity[:, sl] if pity is not None else None, censored)
            s, hs, ks, vs = self._encode(r, st, t0)
            logits = self._head(s, user_idx, campaign[:, sl] if campaign is not None else None)
            st = {"t": t1, "h": hs, "k": ks, "v": vs,
                  "count": st0["count"], "last": st0["last"], "stock": st0["stock"],
                  "pity": st0["pity"]}
            if torch.is_grad_enabled():
                st = self.detach_state(st)
            if on_chunk is None:
                outs.append(logits)
            else:
                on_chunk(t0, t1, logits)
        if track and stock_leaf.grad is not None:
            with torch.autocast(stock.device.type, enabled=False):
                stock.backward(stock_leaf.grad)
        return torch.cat(outs, dim=1) if on_chunk is None else None

    def forward(self, lto_ids, obtained_ids=None, prev_dec_ids=None, user_idx=None,
                ipt=None, campaign=None, inv_init_count=None, inv_init_last=None,
                prefix_obtained=None, prefix_prev=None, censored=None, **kw):
        """Gen-5 call signature, so gen 5's evaluate() and rollout() work.

        Gen 6 always sees a customer's whole history, so a non-empty
        pre-window inventory means the caller truncated the sequence -- refuse
        rather than silently forget it. `ipt` is accepted and unused (no
        calendar-time bias in gen 6).
        """
        if inv_init_count is not None and bool((inv_init_count != 0).any()):
            raise ValueError("gen 6 needs full histories; this batch was truncated "
                             "(non-zero inv_init_count). Build the data with max_events=None.")
        return self.stream_forward(lto_ids, obtained_ids, prev_dec_ids, user_idx, campaign,
                                   prefix_obtained=prefix_obtained, prefix_prev=prefix_prev,
                                   censored=censored)


def build_streaming(*, window: int, chunk: int, vocab_size_src: int, vocab_size_tgt: int,
                    max_seq_len: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                    dropout: float, feature_tensor: torch.Tensor, ai_rate: int = 15,
                    num_users: Optional[int] = None, **kw) -> StreamingHybrid:
    """Mirror of gen 5's build_transformer for the streaming hybrid."""
    return StreamingHybrid(
        window=window, chunk=chunk, censor_flag=kw.get("censor_flag", False),
        slot_state=kw.get("slot_state", "full"), camp_decay=kw.get("camp_decay", 1.0),
        vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt,
        max_seq_len=max_seq_len or 0, ai_rate=ai_rate, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        feature_tensor=feature_tensor,
        lto_len=kw.get("lto_len", 4), obtained_len=kw.get("obtained_len", 10),
        prev_dec_len=kw.get("prev_dec_len", 1), num_users=num_users,
        use_user_embedding=kw.get("use_user_embedding", False),
        num_mix_heads=kw.get("num_mix_heads", 0),
        encoder="gru_attn", fuse="stack", inventory="slots",
        use_offer_inventory_attn=True,
        attn_recency_bias=kw.get("attn_recency_bias", False),
        attn_time_bias=kw.get("attn_time_bias", "none"),
        product_id_embed=kw.get("product_id_embed", True),
        sat_layers=kw.get("sat_layers", 0),
        n_campaigns=kw.get("n_campaigns", 0),
        kernel=kw.get("kernel", "none"), decay=kw.get("decay", "none"),
        decay_init=kw.get("decay_init", 30.0), decay_freeze=kw.get("decay_freeze", False),
        use_product_features=kw.get("use_product_features", True),
        use_obtained=True, pity_stream=kw.get("pity_stream", False),
        tier=kw.get("tier", False), model_rev=kw.get("model_rev", mm.MODEL_REV),
    )


def build_from_cfg(cfg: Dict[str, Any], feature_tensor: torch.Tensor,
                   num_users: Optional[int]) -> StreamingHybrid:
    """Rebuild from a run's cfg (trainer and rollout share this)."""
    return build_streaming(
        window=cfg["window"], chunk=cfg["chunk"], censor_flag=cfg.get("censor_flag", False),
        slot_state=cfg.get("slot_state", "full"), camp_decay=cfg.get("camp_decay", 1.0),
        vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
        max_seq_len=0, d_model=cfg["d_model"], n_layers=cfg["N"],
        n_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        feature_tensor=feature_tensor, ai_rate=cfg["ai_rate"], num_users=num_users,
        lto_len=cfg["lto_len"], obtained_len=cfg["obtained_len"],
        prev_dec_len=cfg["prev_dec_len"],
        use_user_embedding=cfg.get("use_user_embedding", False),
        num_mix_heads=cfg.get("num_mix_heads", 0),
        attn_recency_bias=cfg.get("attn_recency_bias", False),
        attn_time_bias=cfg.get("attn_time_bias", "none"),
        product_id_embed=cfg.get("product_id_embed", True),
        sat_layers=cfg.get("sat_layers", 0),
        n_campaigns=(cfg.get("n_campaigns", 32) if cfg.get("camp_fe") else 0),
        kernel=cfg.get("kernel", "none"), decay=cfg.get("decay", "none"),
        decay_init=cfg.get("decay_init", 30.0), decay_freeze=cfg.get("decay_freeze", False),
        use_product_features=cfg.get("use_product_features", True),
        pity_stream=cfg.get("pity_stream", False), tier=cfg.get("tier", False),
        model_rev=cfg.get("model_rev", mm.MODEL_REV),
    )
