"""R46 part B: a rung as a rollout policy, its state updated one row at a time.

`state.build` computes every state block for whole sequences at once. A
rollout needs the same blocks row by row, from streams that become partly
generated. `Tracker` is that online version, for B sequences in parallel, with
build()'s arithmetic (float32 where build() works in float32). Gate B2 holds it
to build(): replayed on real streams it must reproduce every block on every
row, and the predictive probabilities T1 scored. scripts/test_r46_policy.py
checks this on synthetic panels; scripts/r46b_generate.py on the real
customers before any rollout.

Row conventions are build()'s. Row u carries
    lto[u]       the offer shown before decision u
    obtained[u]  what decision u-1 yielded (the shifted stream)
    prev[u]      y_(u-1); 10 at a history's first row
    ipt[u]       the gap ENDING at row u, read only from row u+1 on (R20)

`policy_rollout` is simulation/rollout.py's loop with the rung in place of the
network: one uniform per sequence and row, the same inverse-CDF draw, the same
environment calls in the same order, and the same rule that a sequence keeps
its real streams until its own first holdout row.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "simulation", REPO / "gen5_multistream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parametric import model as M   # noqa: E402
from parametric import state as ST  # noqa: E402

F64 = torch.float64
N_CLASSES = 9
FIRST, LAST, NP = ST.FIRST, ST.LAST, ST.NP
# gen 5's pity_depth constants (model_multistream_state_space.py)
BANNER5 = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, -1], dtype=np.int64)
HARD5 = np.array([90.0, 90.0, 90.0, 80.0], np.float32)
HARD4 = np.array([10.0, 10.0, 10.0, 10.0], np.float32)
BLOCKS = ("off_a", "off_b", "unoff_b", "off_w", "phase", "prev_idx", "last_banner", "bawa",
          "runs", "ind", "copies_a", "copies_b", "copies_w", "elem_a", "elem_b", "owned",
          "pity", "guar", "wloss")
STREAMS = ("prev", "obtained", "lto", "campaign", "ipt", "label", "inv_init", "length")


def _kd(k: np.ndarray, n: int) -> np.ndarray:
    """build()'s copy dummies: k == 1, ..., k == n-1, k >= n."""
    return np.stack([(k == j) if j < n else (k >= n) for j in range(1, n + 1)], -1).astype(np.float32)


def level7_to_6(feature_file_7) -> np.ndarray:
    """(256,) level-7 token id -> level-6 id; identity below the product range.

    The map scripts/eval_per_occasion.py uses: ProductVocab7's NewProductIndex6
    column, and the three special tokens after the last product to 57, 58, 59.
    Gate B1 checks it reproduces the level-6 streams on every real row.
    """
    import pandas as pd
    tab = pd.read_excel(feature_file_7)
    m = np.arange(256, dtype=np.int64)
    i7 = tab["NewProductIndex7"].astype(int).to_numpy()
    m[i7] = tab["NewProductIndex6"].astype(int).to_numpy()
    last = int(i7.max())
    m[last + 1], m[last + 2], m[last + 3] = 57, 58, 59
    return m


class Context:
    """What every tracker shares: the product table and the offer calendar.

    The calendar comes from the FULL occasion table, as it did when the T1
    features were built (state.main -> build(tab, product_table())).
    """

    def __init__(self, tab: Dict[str, np.ndarray], prod: Dict[str, np.ndarray]):
        cal = ST.offer_calendar(tab)
        pr = ST.prior_runs(cal)
        top = max(int(tab["campaign"].max()), max(pr, default=0)) + 1
        self.runs_tab = np.zeros((top, 60), np.int64)
        for c, v in pr.items():
            self.runs_tab[c] = v
        self.prod = prod
        self.rarity_f = np.asarray(prod["rarity"], np.float32)
        self.lim5_fig_e = np.zeros((NP, 7), np.float32)
        for pid in range(FIRST, LAST + 1):
            if prod["lim5_fig"][pid] and prod["element"][pid] >= 0:
                self.lim5_fig_e[pid - FIRST, prod["element"][pid]] = 1.0
        self.lim5 = prod["lim5"][FIRST:LAST + 1].astype(np.float32)


def sequences(tab: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    """The level-6 streams of table rows `rows`, cut to their longest length."""
    L = int(tab["length"][rows].max())
    out = {}
    for k in STREAMS:
        v = tab[k][rows]
        out[k] = v[:, :L].copy() if v.ndim >= 2 and k != "inv_init" else v.copy()
    return out


class Tracker:
    """The rung's state for B sequences, advanced one row at a time.

    `propose(u)` computes row u's blocks (u is a per-sequence row index) from
    the carries after row u-1 and the streams at rows <= u, and returns the
    predictive probabilities. `commit(y, active)` adopts the row's carries and
    feeds decision y to the class filter, for the sequences in `active` only.
    `write(...)` overwrites a later row's streams with generated values.
    """

    def __init__(self, spec: M.Spec, theta: torch.Tensor, ctx: Context,
                 seq: Dict[str, np.ndarray]):
        self.spec, self.theta, self.ctx = spec, theta, ctx
        self.prev = seq["prev"].astype(np.int64).copy()
        self.obt = seq["obtained"].astype(np.int64).copy()
        self.lto = seq["lto"].astype(np.int64)
        self.camp = seq["campaign"].astype(np.int64)
        self.ipt = seq["ipt"].astype(np.float32)
        self.label = seq["label"].astype(np.int64)
        self.length = seq["length"].astype(np.int64)
        B, self.L = self.prev.shape
        self.B, self.ar = B, np.arange(B)
        P = spec.unpack(theta)
        self.a_d = float(torch.sigmoid(P["a_dec"][0])) if "a_dec" in P else 0.0
        self.a_o = float(torch.sigmoid(P["a_out"][0])) if "a_out" in P else 0.0
        _, logpi = M.log_probs(spec, theta, self._dummy())
        self.logpi = logpi                                          # (K,)
        self.K = int(logpi.shape[0])
        i0 = lambda: np.zeros(B, np.int64)
        f0 = lambda *s: np.zeros((B,) + s, np.float32)
        self.c = dict(LB=np.full(B, -1, np.int64), r=i0(), a=i0(), b=i0(), tr=i0(), g=i0(), wl=i0(),
                      rows_in=f0(), hours_in=f0(), first=np.zeros(B, bool),
                      LD=np.zeros((B, 6)), LO=np.zeros((B, 3)),
                      cnt=seq["inv_init"].astype(np.float32).copy(),
                      pc=f0(4), l5=f0(4), l4=f0(4))
        self.cumF = torch.zeros(self.K, B, dtype=F64)
        self._pending = None

    def _dummy(self) -> Dict[str, torch.Tensor]:
        """A one-row input of the right shapes, to read the class weights."""
        z = lambda *s: torch.zeros((1, 1) + s, dtype=F64)
        return dict(off_a=z(12), off_b=z(12), unoff_b=z(1), off_w=z(5), phase=z(3),
                    prev_idx=torch.zeros(1, 1, dtype=torch.long),
                    last_banner=torch.full((1, 1), -1, dtype=torch.long), bawa=z(2), runs=z(3),
                    ind=z(9), copies_a=z(3), copies_b=z(3), copies_w=z(2), elem_a=z(1), elem_b=z(1),
                    owned=z(1), pity=z(4, 3), guar=z(1), wloss=z(2), LD=z(6), LO=z(3))

    # ------------------------------------------------------------- one row
    def _blocks(self, u: np.ndarray):
        ar, c, ctx = self.ar, self.c, self.ctx
        prod = ctx.prod
        um = np.maximum(u - 1, 0)
        p = self.prev[ar, u]
        o = self.obt[ar, u]
        x = self.lto[ar, u]
        xp = np.where((u > 0)[:, None], self.lto[ar, um], 0)
        cmp = self.camp[ar, u]

        # R0: the offer, and the campaign phase on the lagged clock
        cc = np.clip(cmp, 0, ctx.runs_tab.shape[0] - 1)
        run_of = lambda ids: ctx.runs_tab[cc, np.clip(ids, 0, 59)]
        off_a = ST._fig_offer(x[:, 0], prod, run_of(x[:, 0]))
        off_b = ST._fig_offer(x[:, 1], prod, run_of(x[:, 1]))
        unoff_b = (x[:, 1] < FIRST).astype(np.float32)[:, None]
        w1, w2 = x[:, 2], x[:, 3]
        wt = 0.5 * (ST._onehot(np.where(w1 >= FIRST, prod["wtype"][np.clip(w1, 0, 59)], -1), 5)
                    + ST._onehot(np.where(w2 >= FIRST, prod["wtype"][np.clip(w2, 0, 59)], -1), 5))
        r1, r2 = run_of(w1), run_of(w2)
        off_w = np.concatenate([wt, (0.5 * ((r1 > 0).astype(np.float32) + (r2 > 0))
                                     ).astype(np.float32)[:, None]], -1)
        first = (u == 0) | (cmp != self.camp[ar, um])
        rows_in = np.where(first, np.float32(0), c["rows_in"] + np.float32(1)).astype(np.float32)
        add = np.where(c["first"], np.float32(0), self.ipt[ar, um]).astype(np.float32)
        hours_in = np.where(first, np.float32(0), c["hours_in"] + add).astype(np.float32)
        phase = np.stack([first.astype(np.float32), np.log1p(rows_in),
                          np.log1p(np.maximum(hours_in, np.float32(0)))], -1)

        # R1-R2: the last decision, runs, guarantee, weapon losses
        prev_idx = np.where((p >= 1) & (p <= 9), p, 0)
        valid_o = (o >= FIRST) & (o <= LAST)
        is5 = valid_o & (prod["rarity"][np.clip(o, 0, 59)] == 5)
        w5 = is5.any(-1)
        pb = ST.BANNER_OF_DEC[np.clip(p, 0, 10)]
        f0 = np.select([pb == 1, pb == 2, pb == 3], [xp[:, 0], xp[:, 1], xp[:, 2]], -999)
        f1 = np.where(pb == 3, xp[:, 3], -999)
        hit = (is5 & ((o == f0[:, None]) | (o == f1[:, None]))).any(-1)
        lost = w5 & (pb >= 1) & ~hit
        buy = (p >= 1) & (p <= 8)
        ten = ST.IS_TEN[np.clip(p, 0, 10)]
        LB = np.where(buy, pb, c["LB"])
        r = np.where(buy, np.where(pb == c["LB"], c["r"] + 1, 1), c["r"])
        a = np.where(buy, 0, np.where(p == 9, c["a"] + 1, 0))
        b = np.where(buy, c["b"] + 1, 0)
        tr = np.where(buy & ten, c["tr"] + 1, 0)
        g = np.where(buy & w5 & ((pb == 1) | (pb == 2)), np.where(hit, 0, 1), c["g"])
        wl = np.where(buy & w5 & (pb == 3), np.where(hit, 0, np.minimum(c["wl"] + 1, 2)), c["wl"])
        run_len = np.minimum(r, 100).astype(np.float32)
        bawa = np.stack([run_len / 10.0, (run_len / 10.0) ** 2], -1).astype(np.float32)
        runs = np.stack([np.log1p(a.astype(np.float32)), np.log1p(b.astype(np.float32)),
                         np.log1p(tr.astype(np.float32))], -1)

        # R3: inputs to the decayed stocks, and the stocks themselves
        ind = np.zeros((self.B, 9), np.float32)
        for bn in range(4):
            ind[:, bn] = pb == bn
        ind[:, 4], ind[:, 5], ind[:, 6], ind[:, 7], ind[:, 8] = pb >= 0, ten, w5, hit, lost
        LD = self.a_d * c["LD"] + (1.0 - self.a_d) * ind[:, :6].astype(np.float64)
        LO = self.a_o * c["LO"] + (1.0 - self.a_o) * ind[:, 6:9].astype(np.float64)

        # R4: holdings, inclusive of row u's obtained block (= o_(u-1))
        cnt = c["cnt"].copy()
        rr, cc2 = np.nonzero(valid_o)
        np.add.at(cnt, (rr, o[rr, cc2] - FIRST), np.float32(1))
        own = (cnt > 0).astype(np.float32)
        e_own = own @ ctx.lim5_fig_e
        owned_n = own @ ctx.lim5
        copies = np.zeros((self.B, 4), np.float32)
        for s in range(4):
            ids = x[:, s]
            ok = (ids >= FIRST) & (ids <= LAST)
            copies[:, s] = np.where(ok, cnt[ar, np.clip(ids - FIRST, 0, NP - 1)], 0.0)
        elem = np.zeros((self.B, 2), np.float32)
        for s in range(2):
            ids = x[:, s]
            ok = (ids >= FIRST) & (ids <= LAST)
            el = np.where(ok, prod["element"][np.clip(ids, 0, 59)], -1)
            same = np.where(el >= 0, e_own[ar, np.clip(el, 0, 6)], 0.0)
            mine = np.where(ok, own[ar, np.clip(ids - FIRST, 0, NP - 1)], 0.0)
            elem[:, s] = same - mine * (prod["lim5_fig"][np.clip(ids, 0, 59)] > 0)

        # R4: gen 5's pity_depth, per banner: pulls since the last 5-star / 4-star
        rarf = ctx.rarity_f[np.clip(o, 0, 59)]
        n_pull = valid_o.sum(-1).astype(np.float32)
        h5 = ((rarf >= 4.5) & valid_o).any(-1)
        h4 = ((rarf >= 3.5) & valid_o).any(-1)
        b5 = BANNER5[np.clip(p, 0, 9)]
        pc, l5, l4 = c["pc"].copy(), c["l5"].copy(), c["l4"].copy()
        pity = np.zeros((self.B, 4, 3), np.float32)
        for bn in range(4):
            on = b5 == bn
            pc[:, bn] = pc[:, bn] + np.where(on, n_pull, np.float32(0))
            l5[:, bn] = np.where(h5 & on, pc[:, bn], l5[:, bn])
            l4[:, bn] = np.where(h4 & on, pc[:, bn], l4[:, bn])
            d5 = np.clip((pc[:, bn] - l5[:, bn]) / HARD5[bn], 0.0, 4.0).astype(np.float32)
            pity[:, bn, 0] = d5
            pity[:, bn, 1] = np.maximum(d5 - ST.SOFT_FRAC[bn], 0.0)
            pity[:, bn, 2] = np.clip((pc[:, bn] - l4[:, bn]) / HARD4[bn], 0.0, 4.0)

        F = dict(off_a=off_a, off_b=off_b, unoff_b=unoff_b, off_w=off_w, phase=phase,
                 prev_idx=prev_idx, last_banner=LB, bawa=bawa, runs=runs, ind=ind,
                 copies_a=_kd(copies[:, 0], 3), copies_b=_kd(copies[:, 1], 3),
                 copies_w=_kd(np.maximum(copies[:, 2], copies[:, 3]), 2),
                 elem_a=np.log1p(np.maximum(elem[:, :1], 0.0)).astype(np.float32),
                 elem_b=np.log1p(np.maximum(elem[:, 1:], 0.0)).astype(np.float32),
                 owned=np.log1p(owned_n).astype(np.float32)[:, None], pity=pity,
                 guar=g.astype(np.float32)[:, None], wloss=_kd(wl, 2))
        new = dict(LB=LB, r=r, a=a, b=b, tr=tr, g=g, wl=wl, rows_in=rows_in, hours_in=hours_in,
                   first=first, LD=LD, LO=LO, cnt=cnt, pc=pc, l5=l5, l4=l4)
        return F, new

    @torch.no_grad()
    def propose(self, u: np.ndarray, *, keep_blocks: bool = False):
        """(B,9) predictive probabilities at per-sequence rows u."""
        u = np.clip(np.asarray(u, dtype=np.int64), 0, self.L - 1)
        F, new = self._blocks(u)
        X = {k: torch.as_tensor(v[:, None], dtype=torch.long if k in ("prev_idx", "last_banner") else F64)
             for k, v in F.items()}
        X["LD"] = torch.as_tensor(new["LD"][:, None], dtype=F64)
        X["LO"] = torch.as_tensor(new["LO"][:, None], dtype=F64)
        lp, _ = M.log_probs(self.spec, self.theta, X)
        lp = lp[:, :, 0, :]                                        # (K,B,9)
        if self.K == 1:
            logp = lp[0]
        else:
            lw = self.logpi[:, None] + self.cumF
            lw = lw - torch.logsumexp(lw, 0, keepdim=True)
            logp = torch.logsumexp(lw[..., None] + lp, 0)
        self._pending = (new, lp)
        prob = logp.exp().numpy()
        return (prob, F) if keep_blocks else prob

    def commit(self, y: np.ndarray, active: np.ndarray) -> None:
        """Adopt the proposed row for `active` sequences; y feeds the class filter."""
        new, lp = self._pending
        act = np.asarray(active, dtype=bool)
        for k, v in new.items():
            m = act.reshape((-1,) + (1,) * (v.ndim - 1))
            self.c[k] = np.where(m, v, self.c[k])
        if self.K > 1:
            y = np.asarray(y, dtype=np.int64)
            ok = act & (y >= 1) & (y <= N_CLASSES)
            yi = torch.as_tensor(np.clip(y, 1, N_CLASSES) - 1)
            ly = lp.gather(-1, yi[None, :, None].expand(self.K, -1, 1))[..., 0]   # (K,B)
            self.cumF = self.cumF + ly * torch.as_tensor(ok, dtype=F64)[None]
        self._pending = None

    def warm(self, pre: np.ndarray) -> None:
        """Window rows [0, pre) of every sequence, on its real streams and
        decisions, right-aligned so that every sequence finishes together."""
        pre = np.asarray(pre, dtype=np.int64)
        top = int(pre.max()) if pre.size else 0
        for j in range(top):
            u = j - (top - pre)
            self.propose(u)
            self.commit(self.label[self.ar, np.clip(u, 0, self.L - 1)],
                        (u >= 0) & (u < self.length))

    def write(self, mask: np.ndarray, u: np.ndarray, prev: np.ndarray, obtained6: np.ndarray) -> None:
        """Row u's previous decision and obtained block, for the sequences in `mask`."""
        m = np.asarray(mask, dtype=bool) & (u < self.L)
        idx = np.flatnonzero(m)
        self.prev[idx, u[idx]] = prev[idx]
        self.obt[idx, u[idx]] = obtained6[idx]


# ───────────────────────────── replay (gate B2) ─────────────────────────────
@torch.no_grad()
def replay(spec: M.Spec, theta: torch.Tensor, ctx: Context, seq: Dict[str, np.ndarray],
           *, blocks: bool = True):
    """Every row of every sequence on the real streams and decisions.

    Returns the (B,L,9) predictive probabilities and, if `blocks`, every state
    block as (B,L,...) -- what state.build and model.predictive give on the
    same rows.
    """
    tr = Tracker(spec, theta, ctx, seq)
    B, L = tr.B, tr.L
    prob = np.zeros((B, L, N_CLASSES))
    out: Dict[str, np.ndarray] = {}
    for u in range(L):
        active = u < tr.length
        P, F = tr.propose(np.full(B, u), keep_blocks=True)
        prob[:, u] = P
        if blocks:
            for k, v in F.items():
                if k not in out:
                    out[k] = np.zeros((B, L) + v.shape[1:], v.dtype)
                out[k][:, u] = v
        tr.commit(tr.label[:, u], active)
    return prob, out


# ───────────────────────────── the rollout ─────────────────────────────
@torch.no_grad()
def policy_rollout(spec: M.Spec, theta: torch.Tensor, ctx: Context, seq: Dict[str, np.ndarray],
                   offset: np.ndarray, batch: Dict[str, torch.Tensor], env, vocab, map76: np.ndarray,
                   *, start: int, n_rep: int = 1, seed: int = 0,
                   teacher_forced: bool = False,
                   record: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """simulation.rollout.rollout() with a rung as the policy.

    `batch` is the level-7 R36i chunk; `seq` the same customers' level-6
    windows, with grid row t of customer i at window row offset[i] + t.
    Returns what rollout() returns: level-7 decisions, alive, obtained, lto.
    If `record` is a dict, record["prob"] gets the (B,S,9) probabilities the
    policy drew from, for gate B2 (at a sequence's first holdout row, and on
    every holdout row when teacher-forced, they must be T1's).
    """
    from gacha_env import PityState
    from rollout import _replicate, _warm_pity

    buf = _replicate(batch, n_rep)
    B, S = buf["label"].shape
    rng = np.random.default_rng(seed)
    alive = (buf["label"] != 0).cpu().numpy()
    gen = np.zeros((B, S), dtype=np.int64)
    obtained7 = buf["obtained"].cpu().numpy().copy()
    offers7 = buf["lto"].cpu().numpy()
    states = [PityState() for _ in range(B)]
    if env is not None and not teacher_forced:
        _warm_pity(states, buf, vocab, env, start)

    rep = lambda v: np.repeat(v, n_rep, axis=0)
    tr = Tracker(spec, theta, ctx, {k: rep(v) for k, v in seq.items()})
    off = rep(np.asarray(offset, dtype=np.int64))
    len7 = tr.length - off

    # The real prefix: window rows [0, off + start), so that every sequence
    # reaches grid row `start` together.
    tr.warm(off + start)

    if record is not None:
        record["prob"] = np.zeros((B, S, N_CLASSES), np.float32)
    for t in range(start, S):
        u = off + t
        live = t < len7
        P = tr.propose(u)
        if record is not None:
            record["prob"][:, t] = P
        y = np.zeros(B, dtype=np.int64)
        if alive[:, t].any():
            draw = (P.cumsum(axis=1) < rng.random((B, 1))).sum(axis=1)
            y = np.where(alive[:, t], np.clip(draw + 1, 1, N_CLASSES), 0)
            gen[:, t] = y
        real_y = tr.label[tr.ar, np.clip(u, 0, tr.L - 1)]
        tr.commit(np.where(alive[:, t] & (not teacher_forced), y, real_y), live)
        if t + 1 >= S:
            break
        if teacher_forced:
            continue
        # Same rule as rollout(): rows not yet generating keep their real streams.
        act = alive[:, t] & (y > 0)
        if act.any():
            for b in np.flatnonzero(act):
                obtained7[b, t + 1] = env.step(int(y[b]), offers7[b, t], states[b], rng)
            tr.write(act, u + 1, y, map76[obtained7[:, t + 1]])

    return {"decisions": gen, "alive": alive, "start": start,
            "obtained": obtained7, "lto": offers7}
