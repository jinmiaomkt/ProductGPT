"""Forward simulation from a rung: synthetic panels for G3, and the core of part B.

simulate_panel draws decisions one row at a time. At row t it rebuilds the
state from rows <= t with state.build -- the same builder the real data uses --
evaluates the rung, draws y_t, and turns y_t into the next row's obtained block
with a toy acquisition process. Because the state is rebuilt from the prefix,
a panel simulated here is also a causality test of the builder: the features
of row t must not change when later rows are appended.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from parametric import model as M
from parametric import state as ST

BANNER_OF_DEC = ST.BANNER_OF_DEC


def pools(prod) -> Dict[str, np.ndarray]:
    ids = np.arange(60)
    rng_ok = (ids >= ST.FIRST) & (ids <= ST.LAST)
    sel = lambda m: ids[rng_ok & m]
    return {
        "lim_fig": sel((prod["lim5"] == 1) & (prod["fig"] == 1)),
        "lim_wep": sel((prod["lim5"] == 1) & (prod["fig"] == 0)),
        "std5_fig": sel((prod["rarity"] == 5) & (prod["lto"] == 0) & (prod["fig"] == 1)),
        "std5_wep": sel((prod["rarity"] == 5) & (prod["lto"] == 0) & (prod["fig"] == 0)),
        "four_fig": sel((prod["rarity"] == 4) & (prod["fig"] == 1)),
        "four_wep": sel((prod["rarity"] == 4) & (prod["fig"] == 0)),
        "three": sel(prod["rarity"] == 3),
    }


def synthetic_calendar(n_rows: int, camp_len: int, pl, rng) -> Tuple[np.ndarray, np.ndarray]:
    """(S,) campaign ids from 1 and (S,4) offers: figure A, figure B or none, two weapons."""
    n_c = n_rows // camp_len + 1
    offers = np.zeros((n_c, 4), np.int64)
    for c in range(n_c):
        figs = rng.choice(pl["lim_fig"], 2, replace=False)
        offers[c, 0] = figs[0]
        offers[c, 1] = figs[1] if rng.random() < 0.5 else 0
        offers[c, 2:] = rng.choice(pl["lim_wep"], 2, replace=False)
    camp = np.arange(n_rows) // camp_len
    return camp + 1, offers[camp]


def draw_obtained(dec: np.ndarray, offer: np.ndarray, pl, rng, p5: float = 0.08,
                  p4: float = 0.2) -> np.ndarray:
    """(N,10) toy acquisitions for decisions dec (N,) under offers (N,4)."""
    N = len(dec)
    out = np.zeros((N, 10), np.int64)
    for i in range(N):
        d = int(dec[i])
        if not 1 <= d <= 8:
            continue
        b = int(BANNER_OF_DEC[d])
        for s in range(10 if d % 2 == 0 else 1):
            u = rng.random()
            if u < p5:
                if b == 1 and rng.random() < 0.5:
                    out[i, s] = offer[i, 0]
                elif b == 2 and offer[i, 1] > 0 and rng.random() < 0.5:
                    out[i, s] = offer[i, 1]
                elif b == 3 and rng.random() < 0.75:
                    out[i, s] = offer[i, 2 + int(rng.random() < 0.5)]
                else:
                    pool = pl["std5_wep"] if b == 3 or (b == 0 and rng.random() < 0.5) else pl["std5_fig"]
                    out[i, s] = rng.choice(pool)
            elif u < p5 + p4:
                out[i, s] = rng.choice(pl["four_wep"] if rng.random() < 0.5 else pl["four_fig"])
            else:
                out[i, s] = rng.choice(pl["three"])
    return out


def empty_table(N: int, S: int) -> Dict[str, np.ndarray]:
    return dict(label=np.zeros((N, S), np.int8), prev=np.zeros((N, S), np.int8),
                lto=np.zeros((N, S, 4), np.int16), obtained=np.zeros((N, S, 10), np.int16),
                ipt=np.zeros((N, S), np.float32), campaign=np.zeros((N, S), np.int8),
                holdout=np.zeros((N, S), np.int8), length=np.full(N, S, np.int32),
                is_out=np.zeros(N, np.int8), inv_init=np.zeros((N, ST.NP), np.float32),
                uid_hash=np.arange(N, dtype=np.int64))


def to_X(F: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.long if k in ("prev_idx", "last_banner") else torch.float64)
            for k, v in F.items() if k != "alive"}


@torch.no_grad()
def simulate_panel(spec: M.Spec, theta: torch.Tensor, prod, *, N: int, S: int,
                   camp_len: int = 12, seed: int = 0, record_features: bool = False):
    """Simulate N customers for S rows from the rung (spec, theta).

    A latent-class rung draws each customer's class once from the class weights.
    Returns the occasion table, and (optionally) the row-t features as seen at t.
    """
    rng = np.random.default_rng(seed)
    pl = pools(prod)
    # One calendar, but each customer enters it at a different row, so the
    # panel spans many more offer tuples than one customer sees -- as the real
    # customers span 26 training campaigns.
    max_shift = 4 * S
    camp, offers = synthetic_calendar(S + max_shift, camp_len, pl, rng)
    shift = rng.integers(0, max_shift, N)
    rows = shift[:, None] + np.arange(S)[None, :]
    tab = empty_table(N, S)
    tab["campaign"] = camp[rows].astype(np.int16)
    tab["lto"][:] = offers[rows]
    tab["prev"][:, 0] = ST.NONE_TOKEN
    cal = {int(c): tuple(int(v) for v in offers[np.argmax(camp == c)]) for c in np.unique(camp)}
    cls = None
    seen = {}
    for t in range(S):
        sub = {k: (v[:, :t + 1] if v.ndim >= 2 and k not in ("inv_init",) else v)
               for k, v in tab.items()}
        sub["length"] = np.full(N, t + 1, np.int32)
        F = ST.build(sub, prod, calendar=cal)
        lp, logpi = M.log_probs(spec, theta, to_X(F))
        if cls is None:
            cls = rng.choice(len(logpi), size=N, p=logpi.exp().numpy())
        p = lp[cls, np.arange(N), t].exp().numpy()
        p = p / p.sum(1, keepdims=True)
        y = np.array([rng.choice(9, p=p[i]) + 1 for i in range(N)])
        tab["label"][:, t] = y
        if record_features:
            seen[t] = {k: v[:, t].copy() for k, v in F.items()}
        if t + 1 < S:
            tab["prev"][:, t + 1] = y
            tab["obtained"][:, t + 1] = draw_obtained(y, tab["lto"][:, t].astype(np.int64), pl, rng)
            tab["ipt"][:, t + 1] = np.where(y == 9, 24.0, np.round(rng.exponential(2.0, N), 2))
    tab["class"] = cls
    return (tab, seen, cal) if record_features else (tab, cal)


def true_theta(spec: M.Spec, seed: int = 0, scale: float = 0.3) -> torch.Tensor:
    """A plausible parameter vector: about half NotBuy, short decision memory."""
    g = torch.Generator().manual_seed(seed)
    shares = np.array([0.18, 0.02, 0.15, 0.04, 0.06, 0.02, 0.06, 0.02, 0.45])
    th = spec.init(marginal=shares)
    th = th + scale * torch.randn(spec.size, generator=g, dtype=torch.float64)
    off = spec.offsets()
    logit = lambda p: float(np.log(p / (1 - p)))
    fixed = {"a_dec": logit(0.6), "a_out": logit(0.8), "rho": logit(0.15)}
    for n, v in fixed.items():
        if n in off:
            th[off[n][0]] = v
    if "nest" in off:
        th[off["nest"][0]] = logit(0.7)
    if "cls" in off:
        a, b = off["cls"]
        th[a:b] = 0.0
    if "d_asc" in off:
        a, b = off["d_asc"]
        th[a:b] = torch.tensor([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5], dtype=torch.float64).repeat((b - a) // 8)
    return th
