"""The observed pre-decision state, built from what row t can see.

Row t of the occasion table carries
    lto[t]       the offer shown before the decision            x_t
    obtained[t]  what the PREVIOUS decision yielded (shifted)   o_(t-1)
    prev[t]      the previous decision token (10 = none)        y_(t-1)
    ipt[t]       the gap ENDING at row t -- never used at row t (R20)
so every feature below is a function of rows <= t of those three streams,
rows < t of ipt, and the pre-window inventory counts. Nothing reads label[t].

The blocks map onto the R46 ladder:
    R0  off_fig_a / off_fig_b / off_wep / unoff_b / phase
    R1  prev_idx
    R2  last_banner + bawa, runs
    R3  ind (inputs to the decayed stocks; the decays are parameters)
    R4  copies_a / copies_b / copies_w, elem_a / elem_b, owned, pity, guar, wloss
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FIRST, LAST = 13, 56                    # level-6 product ids
NP = LAST - FIRST + 1
NONE_TOKEN = 10                         # SOS: no previous decision
# decision 1..9 -> banner 0 regular, 1 figure A, 2 figure B, 3 weapon; -1 NotBuy
BANNER_OF_DEC = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, -1, -1], dtype=np.int64)
IS_TEN = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=bool)
ELEMENTS = ("Ice", "Rock", "Water", "Fire", "Thunder", "Wind", "Grass")
WTYPES = ("OneHandSword", "TwoHandSword", "Arrow", "Magic", "Polearm")
# soft pity as a fraction of the hard cap, matching pity_depth's scaling
SOFT_FRAC = np.array([74 / 90, 74 / 90, 74 / 90, 63 / 80])


# ───────────────────────────── product table ─────────────────────────────
def product_table(path=None) -> Dict[str, np.ndarray]:
    """Per token id (0..59): rarity, figure, limited, element, weapon type, female."""
    import pandas as pd
    if path is None:
        import paths
        path = paths.product_feature_xlsx()
    t = pd.read_excel(path)
    size = 60
    out = {k: np.zeros(size, np.int64) for k in ("rarity", "fig", "lto", "female")}
    out["element"] = np.full(size, -1, np.int64)
    out["wtype"] = np.full(size, -1, np.int64)
    for _, r in t.iterrows():
        i = int(r["NewProductIndex6"])
        out["rarity"][i] = int(r["Rarity"])
        out["fig"][i] = int(r["type_figure"])
        out["lto"][i] = int(r["LTO"])
        out["female"][i] = int(r.get("GenderFemale", 0) or 0)
        for e, name in enumerate(ELEMENTS):
            if int(r.get(f"Ethnicity{name}", 0) or 0) == 1:
                out["element"][i] = e
        for w, name in enumerate(WTYPES):
            if int(r.get(f"WeaponType{name}", 0) or 0) == 1:
                out["wtype"][i] = w
    out["lim5"] = ((out["rarity"] == 5) & (out["lto"] == 1)).astype(np.int64)
    out["lim5_fig"] = out["lim5"] * out["fig"]
    return out


def offer_calendar(tab: Dict[str, np.ndarray]) -> Dict[int, tuple]:
    """Campaign -> its offer tuple (the most common one; one per campaign)."""
    camp, lto, lab = tab["campaign"], tab["lto"], tab["label"]
    valid = (lab >= 1) & (lab <= 9)
    cnt: Dict[int, Counter] = {}
    for c, row in zip(camp[valid].tolist(), lto[valid].tolist()):
        cnt.setdefault(int(c), Counter())[tuple(int(v) for v in row)] += 1
    return {c: k.most_common(1)[0][0] for c, k in cnt.items()}


def prior_runs(calendar: Dict[int, tuple]) -> Dict[int, np.ndarray]:
    """Campaign -> (60,) number of EARLIER campaigns that featured each id."""
    out, seen = {}, np.zeros(60, np.int64)
    for c in sorted(calendar):
        out[c] = seen.copy()
        for v in set(calendar[c]):
            if FIRST <= v <= LAST:
                seen[v] += 1
    return out


# ───────────────────────────── the builder ─────────────────────────────
def _onehot(idx: np.ndarray, n: int, drop_first: bool = True) -> np.ndarray:
    oh = np.zeros(idx.shape + (n,), np.float32)
    ok = idx >= 0
    oh[ok, idx[ok]] = 1.0
    return oh[..., 1:] if drop_first else oh


def _fig_offer(ids: np.ndarray, prod, runs: np.ndarray) -> np.ndarray:
    """(…,13): element dummies (6), weapon-type dummies (4), female, rerun, log prior runs."""
    have = (ids >= FIRST) & (ids <= LAST)
    safe = np.where(have, ids, 0)
    el = np.where(have, prod["element"][safe], -1)
    wt = np.where(have, prod["wtype"][safe], -1)
    fem = np.where(have, prod["female"][safe], 0).astype(np.float32)
    rr = np.where(have, runs, 0).astype(np.float32)
    return np.concatenate([_onehot(el, 7), _onehot(wt, 5), fem[..., None],
                           (rr > 0).astype(np.float32)[..., None],
                           np.log1p(rr)[..., None]], axis=-1)


def build(tab: Dict[str, np.ndarray], prod: Dict[str, np.ndarray],
          calendar: Dict[int, tuple] | None = None) -> Dict[str, np.ndarray]:
    lab, prev, lto, obt = tab["label"], tab["prev"].astype(np.int64), tab["lto"].astype(np.int64), tab["obtained"].astype(np.int64)
    camp, ipt, length = tab["campaign"].astype(np.int64), tab["ipt"], tab["length"]
    N, S = lab.shape
    if calendar is None:
        calendar = offer_calendar(tab)
    pr = prior_runs(calendar)
    runs_tab = np.zeros((max(int(camp.max()), max(pr, default=0)) + 1, 60), np.int64)
    for c, v in pr.items():
        runs_tab[c] = v
    alive = np.arange(S)[None, :] < length[:, None]

    # ---- R0: the offer ---------------------------------------------------
    cc = np.clip(camp, 0, runs_tab.shape[0] - 1)
    run_of = lambda ids: runs_tab[cc, np.clip(ids, 0, 59)]
    off_a = _fig_offer(lto[..., 0], prod, run_of(lto[..., 0]))
    off_b = _fig_offer(lto[..., 1], prod, run_of(lto[..., 1]))
    unoff_b = ((lto[..., 1] < FIRST) & alive).astype(np.float32)[..., None]
    w1, w2 = lto[..., 2], lto[..., 3]
    wt = 0.5 * (_onehot(np.where(w1 >= FIRST, prod["wtype"][np.clip(w1, 0, 59)], -1), 5)
                + _onehot(np.where(w2 >= FIRST, prod["wtype"][np.clip(w2, 0, 59)], -1), 5))
    r1, r2 = run_of(w1), run_of(w2)
    off_w = np.concatenate([wt, (0.5 * ((r1 > 0).astype(np.float32) + (r2 > 0))).astype(np.float32)[..., None],
                            np.log1p(0.5 * (r1 + r2)).astype(np.float32)[..., None]], axis=-1)

    # campaign phase, on the buy decision; the clock is lagged one row (R21)
    first = np.zeros((N, S), np.float32)
    first[:, 0] = 1.0
    first[:, 1:] = (camp[:, 1:] != camp[:, :-1]).astype(np.float32)
    rows_in = np.zeros((N, S), np.float32)
    hours_in = np.zeros((N, S), np.float32)
    for t in range(1, S):
        new = first[:, t] > 0
        rows_in[:, t] = np.where(new, 0.0, rows_in[:, t - 1] + 1.0)
        # hours from the entry row to row t-1: add ipt[t-1] unless t-1 IS the entry
        add = np.where(first[:, t - 1] > 0, 0.0, ipt[:, t - 1])
        hours_in[:, t] = np.where(new, 0.0, hours_in[:, t - 1] + add)
    phase = np.stack([first, np.log1p(rows_in), np.log1p(np.maximum(hours_in, 0.0))], -1)

    # ---- R1: last decision -------------------------------------------------
    prev_idx = np.where((prev >= 1) & (prev <= 9), prev, 0).astype(np.int64)
    prev_idx[~alive] = 0

    # ---- R2/R3/R4: sequential states ----------------------------------------
    rar = prod["rarity"]
    valid_o = (obt >= FIRST) & (obt <= LAST)
    is5 = valid_o & (rar[np.clip(obt, 0, 59)] == 5)
    w5 = is5.any(-1)
    pb = BANNER_OF_DEC[np.clip(prev, 0, 10)]                       # banner of y_(t-1)
    lto_prev = np.concatenate([np.zeros((N, 1, 4), np.int64), lto[:, :-1]], axis=1)
    feat_prev = np.full((N, S, 2), -999, np.int64)                 # featured ids at t-1
    feat_prev[..., 0] = np.select([pb == 1, pb == 2, pb == 3],
                                  [lto_prev[..., 0], lto_prev[..., 1], lto_prev[..., 2]], -999)
    feat_prev[..., 1] = np.where(pb == 3, lto_prev[..., 3], -999)
    hit_feat = (is5 & ((obt == feat_prev[..., :1]) | (obt == feat_prev[..., 1:2]))).any(-1)
    lost = w5 & (pb >= 1) & ~hit_feat

    last_banner = np.full((N, S), -1, np.int64)
    run_len = np.zeros((N, S), np.float32)
    nb = np.zeros((N, S), np.float32)
    br = np.zeros((N, S), np.float32)
    tr = np.zeros((N, S), np.float32)
    guar = np.zeros((N, S), np.float32)
    wloss = np.zeros((N, S), np.int64)
    banner_l = BANNER_OF_DEC.tolist()
    ten_l = IS_TEN.tolist()
    for i in range(N):
        L = int(length[i])
        P, W5, HF = prev[i, :L].tolist(), w5[i, :L].tolist(), hit_feat[i, :L].tolist()
        o_lb, o_r, o_a, o_b, o_c, o_g, o_w = ([0] * L for _ in range(7))
        LB, r, a, b, c, g, wl = -1, 0, 0, 0, 0, 0, 0
        for t in range(L):
            p = P[t]
            if 1 <= p <= 8:
                bb = banner_l[p]
                if bb == LB:
                    r += 1
                else:
                    LB, r = bb, 1
                a, b = 0, b + 1
                c = c + 1 if ten_l[p] else 0
                if W5[t]:
                    if bb == 1 or bb == 2:
                        g = 0 if HF[t] else 1
                    elif bb == 3:
                        wl = 0 if HF[t] else min(wl + 1, 2)
            elif p == 9:
                a, b, c = a + 1, 0, 0
            else:
                a = b = c = 0
            o_lb[t], o_r[t], o_a[t], o_b[t], o_c[t], o_g[t], o_w[t] = LB, min(r, 100), a, b, c, g, wl
        last_banner[i, :L], run_len[i, :L] = o_lb, o_r
        nb[i, :L], br[i, :L], tr[i, :L] = o_a, o_b, o_c
        guar[i, :L], wloss[i, :L] = o_g, o_w
    bawa = np.stack([run_len / 10.0, (run_len / 10.0) ** 2], -1)
    runs = np.stack([np.log1p(nb), np.log1p(br), np.log1p(tr)], -1)

    # inputs to the decayed stocks (R3): what y_(t-1) and o_(t-1) were
    ind = np.zeros((N, S, 9), np.float32)
    for bnr in range(4):
        ind[..., bnr] = (pb == bnr)
    ind[..., 4] = (pb >= 0)
    ind[..., 5] = IS_TEN[np.clip(prev, 0, 10)]
    ind[..., 6] = w5
    ind[..., 7] = hit_feat
    ind[..., 8] = lost
    ind[~alive] = 0.0

    # ---- R4: holdings, inclusive of row t's obtained block (= o_(t-1)) -------
    lim5_fig_e = np.zeros((NP, 7), np.float32)
    for pid in range(FIRST, LAST + 1):
        if prod["lim5_fig"][pid] and prod["element"][pid] >= 0:
            lim5_fig_e[pid - FIRST, prod["element"][pid]] = 1.0
    lim5 = prod["lim5"][FIRST:LAST + 1].astype(np.float32)
    copies = np.zeros((N, S, 4), np.float32)                       # A, B, W1, W2
    elem_same = np.zeros((N, S, 2), np.float32)
    owned_n = np.zeros((N, S), np.float32)
    for i in range(N):
        L = int(length[i])
        per = np.zeros((L, NP), np.float32)
        o = obt[i, :L]
        rr, cc2 = np.nonzero((o >= FIRST) & (o <= LAST))
        np.add.at(per, (rr, o[rr, cc2] - FIRST), 1.0)
        cnt = np.cumsum(per, axis=0) + tab["inv_init"][i][None, :]
        own = (cnt > 0).astype(np.float32)
        e_own = own @ lim5_fig_e                                    # (L,7)
        owned_n[i, :L] = own @ lim5
        for s in range(4):
            ids = lto[i, :L, s]
            ok = (ids >= FIRST) & (ids <= LAST)
            copies[i, :L, s] = np.where(ok, cnt[np.arange(L), np.clip(ids - FIRST, 0, NP - 1)], 0.0)
        for s in range(2):
            ids = lto[i, :L, s]
            ok = (ids >= FIRST) & (ids <= LAST)
            el = np.where(ok, prod["element"][np.clip(ids, 0, 59)], -1)
            same = np.where(el >= 0, e_own[np.arange(L), np.clip(el, 0, 6)], 0.0)
            mine = np.where(ok, own[np.arange(L), np.clip(ids - FIRST, 0, NP - 1)], 0.0)
            elem_same[i, :L, s] = same - mine * (prod["lim5_fig"][np.clip(ids, 0, 59)] > 0)
    kd = lambda k, n: np.stack([(k == j) if j < n else (k >= n) for j in range(1, n + 1)], -1).astype(np.float32)
    copies_a = kd(copies[..., 0], 3)
    copies_b = kd(copies[..., 1], 3)
    copies_w = kd(np.maximum(copies[..., 2], copies[..., 3]), 2)
    elem_a = np.log1p(np.maximum(elem_same[..., :1], 0.0))
    elem_b = np.log1p(np.maximum(elem_same[..., 1:], 0.0))
    owned = np.log1p(owned_n)[..., None]

    pity = pity_features(obt, prev, prod)                           # (N,S,4,3)

    return dict(off_a=off_a, off_b=off_b, unoff_b=unoff_b, off_w=off_w, phase=phase,
                prev_idx=prev_idx, last_banner=last_banner, bawa=bawa.astype(np.float32),
                runs=runs.astype(np.float32), ind=ind, copies_a=copies_a, copies_b=copies_b,
                copies_w=copies_w, elem_a=elem_a.astype(np.float32),
                elem_b=elem_b.astype(np.float32), owned=owned.astype(np.float32), pity=pity,
                guar=guar[..., None], wloss=kd(wloss, 2), alive=alive)


def main() -> None:
    import argparse
    import time
    from parametric import data as D
    ap = argparse.ArgumentParser(description="R46: build the state features once")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tab = D.load(Path(a.table))
    F = build(tab, product_table())
    np.savez(a.out, **F)
    shapes = {k: tuple(v.shape) for k, v in F.items()}
    print(f"state features for {tab['label'].shape[0]} customers in {time.time() - t0:.0f}s: {shapes}")


def pity_features(obt: np.ndarray, prev: np.ndarray, prod) -> np.ndarray:
    """(N,S,4,3) per banner: 5-star depth, soft-pity hinge, 4-star depth.

    The depths are gen 5's own `pity_depth` (pulls since the last hit, per
    banner, scaled by the hard cap), so the parametric models read exactly the
    counters the R38 pity stream gives the neural model.
    """
    import torch
    import model_multistream_state_space as m
    rarity_of = torch.zeros(60, dtype=torch.float32)
    rarity_of[:] = torch.as_tensor(prod["rarity"], dtype=torch.float32)
    out = np.zeros(obt.shape[:2] + (4, 3), np.float32)
    for a in range(0, obt.shape[0], 256):
        o = torch.as_tensor(obt[a:a + 256], dtype=torch.long)
        p = torch.as_tensor(prev[a:a + 256], dtype=torch.long)
        d = m.pity_depth(o, p, rarity_of).numpy()                  # (b,S,8)
        for bnr in range(4):
            d5, d4 = d[..., 2 * bnr], d[..., 2 * bnr + 1]
            out[a:a + 256, :, bnr, 0] = d5
            out[a:a + 256, :, bnr, 1] = np.maximum(d5 - SOFT_FRAC[bnr], 0.0)
            out[a:a + 256, :, bnr, 2] = d4
    return out


if __name__ == "__main__":
    main()
