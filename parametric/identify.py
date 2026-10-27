"""R46 part C: simulate a rung on the real design, and the quantities part C reads.

The real design is what part A's fit saw, minus the decisions: the in-sample
customers' window rows up to their last training row, with the real offers,
campaigns, clock, lengths and pre-window inventory. `simulate` replaces the
decisions with draws from a rung, and each purchase's acquisitions with draws
from R36a's AcquisitionEnv on level-6 ids. The state is policy.Tracker (part
B's gate B2 holds it to state.build), so a refit on `state.build` of the
simulated table sees exactly the state the draws were made from (gate C-G2).

    design(tab, roles)          the customers, prefix lengths and streams
    level6_env(tab, prod, u)    AcquisitionEnv on level-6 ids, training rows only
    simulate(...)               one replicate's decisions and acquisitions
    synthetic_table(...)        the simulated occasion table (fit.py's input)
    check_design / check_simulation   gates C-G1 and C-G2
    embed_h6, continuous_h6     the Keane nulls
    quantities(spec, theta)     the SD and HG quantities, as torch scalars

Nothing here prints a record: counts, shares and checksums only.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "simulation", REPO / "gen5_multistream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parametric import model as M    # noqa: E402
from parametric import policy as PL  # noqa: E402
from parametric import state as ST   # noqa: E402

F64 = torch.float64
SHARED_EXCLUDE = ("d_asc", "d_inc", "d_loy", "d_nb", "cls")
SD_KEYS = ("loy", "inc", "tenl", "repeat_med")


# ───────────────────────────── the real design ─────────────────────────────
def design(tab: Dict[str, np.ndarray], roles: Dict[str, np.ndarray], *,
           max_users: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """In-sample customers with training rows, their prefix lengths T (through
    the last training row) and their streams."""
    tr = roles["train"]
    users = np.flatnonzero((tab["is_out"] == 0) & tr.any(1))
    if max_users:
        users = users[:max_users]
    T = np.array([int(np.flatnonzero(tr[i]).max()) + 1 for i in users], dtype=np.int64)
    seq = PL.sequences(tab, users)
    seq["length"] = T.copy()
    return users, T, seq


def level6_env(tab: Dict[str, np.ndarray], prod: Dict[str, np.ndarray], users: np.ndarray,
               *, holdout_from: int = 27):
    """R36a's AcquisitionEnv, fitted on `users`' rows before campaign `holdout_from`."""
    from gacha_env import AcquisitionEnv, ProductTable
    ids = np.arange(ST.FIRST, ST.LAST + 1)
    vocab = ProductTable(ids, prod["rarity"][ids], prod["fig"][ids], prod["lto"][ids],
                         ST.FIRST, ST.LAST)
    items = []
    for i in users:
        L = int(tab["length"][i])
        items.append(dict(lto=tab["lto"][i, :L], obtained=tab["obtained"][i, :L],
                          prev_decision=tab["prev"][i, :L], label=tab["label"][i, :L],
                          campaign=tab["campaign"][i, :L]))
    return AcquisitionEnv(vocab).fit(items, holdout_from=holdout_from)


# ───────────────────────────── one replicate ─────────────────────────────
@torch.no_grad()
def simulate(spec: M.Spec, theta: torch.Tensor, ctx: PL.Context, seq: Dict[str, np.ndarray],
             T: np.ndarray, env, *, seed: int, record: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Decisions from (spec, theta) on rows [0, T_i), acquisitions from `env`.

    A latent-class rung draws each customer's class once from its weights.
    Rows before T keep their real offers, campaigns and clock; row 0 keeps its
    real previous decision and obtained block. One generator per replicate:
    the class draws, then per row one uniform per customer (part B's
    inverse-CDF draw) and the environment's draws.

    `record={"users": idx}` keeps, for those sequence indices, every block
    and the drawn class's probabilities on every row (gate C-G2).
    """
    from gacha_env import PityState
    rng = np.random.default_rng(seed)
    tr = PL.Tracker(spec, theta, ctx, seq)
    B, L = tr.B, tr.L
    if tr.K > 1:
        p = tr.logpi.exp().numpy()
        cls = rng.choice(tr.K, size=B, p=p / p.sum())
    else:
        cls = np.zeros(B, dtype=np.int64)
    states = [PityState() for _ in range(B)]
    label = np.zeros((B, L), dtype=np.int64)
    rec = None
    if record is not None:
        rec = np.asarray(record["users"], dtype=np.int64)
        record["blocks"], record["prob"] = {}, np.zeros((len(rec), L, PL.N_CLASSES))
    Tmax = int(T.max())
    for t in range(Tmax):
        live = t < T
        u = np.full(B, t, dtype=np.int64)
        if rec is not None:
            P, F = tr.propose(u, keep_blocks=True, cls=cls)
            for k, v in F.items():
                if k not in record["blocks"]:
                    record["blocks"][k] = np.zeros((len(rec), L) + v.shape[1:], v.dtype)
                record["blocks"][k][:, t] = v[rec]
            record["prob"][:, t] = P[rec]
        else:
            P = tr.propose(u, cls=cls)
        draw = (P.cumsum(axis=1) < rng.random((B, 1))).sum(axis=1)
        y = np.where(live, np.clip(draw + 1, 1, PL.N_CLASSES), 0)
        label[:, t] = y
        tr.commit(y, live)
        if t + 1 >= Tmax:
            break
        nxt = live & (t + 1 < T)
        obt = np.zeros((B, 10), dtype=np.int64)
        for b in np.flatnonzero(nxt & (y >= 1) & (y <= 8)):
            obt[b] = env.step(int(y[b]), tr.lto[b, t], states[b], rng)
        tr.write(nxt, np.full(B, t + 1, dtype=np.int64), y, obt)
    return dict(label=label, prev=tr.prev, obtained=tr.obt, cls=cls)


def synthetic_table(tab: Dict[str, np.ndarray], users: np.ndarray, T: np.ndarray,
                    seq: Dict[str, np.ndarray], sim: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The occasion table of one replicate: real exogenous streams, simulated
    decisions and acquisitions, cut at T."""
    S = seq["prev"].shape[1]
    alive = np.arange(S)[None, :] < T[:, None]
    a3 = alive[..., None]
    return dict(
        label=np.where(alive, sim["label"], 0).astype(np.int8),
        prev=np.where(alive, sim["prev"], 0).astype(np.int8),
        lto=np.where(a3, seq["lto"], 0).astype(np.int16),
        obtained=np.where(a3, sim["obtained"], 0).astype(np.int16),
        ipt=np.where(alive, seq["ipt"], 0).astype(np.float32),
        campaign=np.where(alive, seq["campaign"], 0).astype(np.int8),
        holdout=np.where(alive, tab["holdout"][users, :S], 0).astype(np.int8),
        length=T.astype(np.int32),
        is_out=np.zeros(len(users), np.int8),
        inv_init=tab["inv_init"][users].astype(np.float32),
        uid_hash=tab["uid_hash"][users].copy())


def checksum(*arrays: np.ndarray) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ───────────────────────────── gates ─────────────────────────────
def check_design(tab, roles, users, T, tab_s, roles_s) -> Dict:
    """C-G1: the simulated table has the real design on rows [0, T)."""
    S = tab_s["label"].shape[1]
    alive = np.arange(S)[None, :] < T[:, None]
    real = {k: tab[k][users, :S] for k in ("lto", "campaign", "ipt", "holdout")}
    same = {}
    for k, v in real.items():
        a = np.where(alive[..., None] if v.ndim == 3 else alive, v, 0)
        b = np.where(alive[..., None] if v.ndim == 3 else alive, tab_s[k], 0)
        same[k] = bool(np.array_equal(a.astype(np.float64), b.astype(np.float64)))
    same["inv_init"] = bool(np.array_equal(tab["inv_init"][users], tab_s["inv_init"]))
    tr_real = roles["train"][users, :S]
    tr_sim = roles_s["train"]
    same["train_mask"] = bool(np.array_equal(tr_real, tr_sim))
    return dict(customers=int(len(users)), rows=int(T.sum()), train_rows=int(tr_sim.sum()),
                equal=same, ok=all(same.values()))


@torch.no_grad()
def check_simulation(spec, theta, F_s: Dict[str, np.ndarray], tab_s, record: dict,
                     cls: np.ndarray, *, piece: int = 20) -> Dict:
    """C-G2: state.build on the simulated streams gives the blocks the draws
    used, and the drawn class's probabilities, on every live row."""
    rec = np.asarray(record["users"], dtype=np.int64)
    T = tab_s["length"][rec]
    L = record["prob"].shape[1]
    alive = np.arange(L)[None, :] < T[:, None]
    worst = {}
    for k in PL.BLOCKS:
        a = record["blocks"][k][alive].astype(np.float64)
        b = F_s[k][rec, :L][alive].astype(np.float64)
        worst[k] = float(np.abs(a - b).max()) if a.size else 0.0
    pe = 0.0
    for s in range(0, len(rec), piece):
        r = rec[s:s + piece]
        X = {k: torch.as_tensor(F_s[k][r, :L], dtype=torch.long if k in ("prev_idx", "last_banner") else F64)
             for k in F_s if k != "alive"}
        lp, _ = M.log_probs(spec, theta, X)
        p = lp[torch.as_tensor(cls[r]), torch.arange(len(r))].exp().numpy()
        m = alive[s:s + piece]
        pe = max(pe, float(np.abs(p[m] - record["prob"][s:s + piece][m]).max()))
    return dict(users=int(len(rec)), block_max=max(worst.values()), prob_max=pe,
                bad_blocks=sorted(k for k, v in worst.items() if v > 1e-6),
                ok=max(worst.values()) <= 1e-6 and pe <= 1e-6)


# ───────────────────────────── a simulated world, summarised ─────────────────────────────
def world_summary(tab_s: Dict[str, np.ndarray], mask: np.ndarray, prod: Dict[str, np.ndarray]) -> Dict:
    """Decision shares on `mask` rows, and what each banner's pulls yielded."""
    lab = tab_s["label"][mask].astype(np.int64)
    shares = np.bincount(lab - 1, minlength=9) / max(len(lab), 1)
    prev, obt, lto = tab_s["prev"].astype(np.int64), tab_s["obtained"].astype(np.int64), tab_s["lto"].astype(np.int64)
    m1 = mask.copy()
    m1[:, 0] = False                                  # row 0's block is pre-window
    pb = ST.BANNER_OF_DEC[np.clip(prev, 0, 10)]
    valid = (obt >= ST.FIRST) & (obt <= ST.LAST)
    r5 = valid & (prod["rarity"][np.clip(obt, 0, 59)] == 5)
    lto_prev = np.concatenate([np.zeros_like(lto[:, :1]), lto[:, :-1]], 1)
    out = {"decision_shares": [round(float(v), 5) for v in shares], "banners": {}}
    for b, name in enumerate(("regular", "figure_A", "figure_B", "weapon")):
        sel = m1 & (pb == b)
        n_items = int(valid[sel].sum())
        n5 = int(r5[sel].sum())
        d = {"pull_rows": int(sel.sum()), "items": n_items,
             "five_star_per_item": round(n5 / max(n_items, 1), 5)}
        if b >= 1:
            slots = (0,) if b == 1 else (1,) if b == 2 else (2, 3)
            feat = np.zeros(obt.shape, bool)
            for s in slots:
                feat |= obt == lto_prev[..., s:s + 1]
            d["featured_share_of_5star"] = round(int((r5 & feat)[sel].sum()) / max(n5, 1), 4)
        out["banners"][name] = d
    return out


# ───────────────────────────── the Keane nulls ─────────────────────────────
def pack(spec: M.Spec, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
    th = torch.zeros(spec.size, dtype=F64)
    for n, (lo, hi) in spec.offsets().items():
        if n in parts:
            th[lo:hi] = torch.as_tensor(parts[n], dtype=F64).reshape(-1)
    return th


def embed_h6(theta_h6: torch.Tensor, K: int, *, a_dec: float) -> torch.Tensor:
    """H6's estimate as an R6 parameter vector: equal lag rows (H6's constants),
    zero loyalty, run, Bawa and state-dependent class terms, everything else
    copied. `a_dec` (a logit) is inert while the decision stocks are zero."""
    ph = M.Spec("H6", K=K).unpack(theta_h6)
    s6 = M.Spec("R6", K=K)
    parts = {n: v for n, v in ph.items() if n in s6.names and n != "asc"}
    parts["lag"] = ph["asc"][None, :].repeat(10, 1)
    parts["a_dec"] = torch.tensor([a_dec], dtype=F64)
    return pack(s6, parts)


def continuous_h6(theta_h6: torch.Tensor, K: int, *, K_big: int = 200, seed: int = 46499
                  ) -> Tuple[M.Spec, torch.Tensor, Dict]:
    """H6 with its K class-constant vectors replaced by K_big equally weighted
    classes drawn from a normal with the same class-weighted mean and covariance."""
    sh = M.Spec("H6", K=K)
    ph = sh.unpack(theta_h6)
    w = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), ph["cls"]]), 0).numpy()
    A = ph["asc"].numpy()[None, :] + np.vstack([np.zeros((1, 8)), ph["d_asc"].numpy()])   # (K,8)
    m = w @ A
    C = ((A - m).T * w) @ (A - m)
    ev, U = np.linalg.eigh(C)
    Lf = U * np.sqrt(np.clip(ev, 0.0, None))
    rng = np.random.default_rng(seed)
    draws = m[None, :] + rng.standard_normal((K_big, 8)) @ Lf.T
    sb = M.Spec("H6", K=K_big)
    parts = {n: v for n, v in ph.items() if n not in ("asc", "d_asc", "cls")}
    parts["asc"] = torch.as_tensor(draws[0])
    parts["d_asc"] = torch.as_tensor(draws[1:] - draws[0][None, :])
    parts["cls"] = torch.zeros(K_big - 1, dtype=F64)
    info = dict(target_mean=m.round(4).tolist(), draw_mean=draws.mean(0).round(4).tolist(),
                target_sd=np.sqrt(np.diag(C)).round(4).tolist(),
                draw_sd=draws.std(0).round(4).tolist(), rank=int((ev > 1e-10 * ev.max()).sum()))
    return sb, pack(sb, parts), info


def bootstrap_weights(n: int, seed: int) -> np.ndarray:
    """Customer bootstrap: how many times each of n customers is drawn."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n)).astype(np.int64)


# ───────────────────────────── the quantities ─────────────────────────────
def quantities(spec: M.Spec, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Part C's SD and HG quantities (class-weighted for a latent-class rung),
    differentiable in theta for the delta method."""
    P = spec.unpack(theta)
    q: Dict[str, torch.Tensor] = {}
    if spec.flags["classes"] and spec.K > 1:
        w = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), P["cls"]]), 0)
        zk = lambda v: torch.cat([torch.zeros(1, dtype=F64), v])
        cw = lambda base, dev: base + ((w * zk(P[dev])).sum() if dev in P else 0.0)
    else:
        cw = lambda base, dev: base
    if "loy" in P:
        q["loy"] = cw(P["loy"][0], "d_loy")
        q["inc"] = cw(P["inc"][0], "d_inc")
        q["tenl"] = P["tenl"][0]
    if "lag" in P:
        lag = P["lag"]
        rep = []
        for j in range(8):
            others = torch.cat([lag[1:j + 1, j], lag[j + 2:, j]])     # previous decisions 1..9 except j+1
            rep.append(lag[j + 1, j] - others.mean())
        rep = torch.stack(rep)
        s = torch.sort(rep).values
        q["repeat_med"] = 0.5 * (s[3] + s[4])
        for j in range(8):
            q[f"repeat[{j}]"] = rep[j]
    for n, label in (("a_dec", "hl_dec"), ("a_out", "hl_out")):
        if n in P:
            q[label] = np.log(0.5) / torch.log(torch.sigmoid(P[n][0]))
    if "bawa" in P:
        for j in range(2):
            q[f"bawa[{j}]"] = P["bawa"][j]
        for j in range(3):
            q[f"runs[{j}]"] = P["runs"][j]
    if "cop_f" in P:
        for j in range(3):
            q[f"cop_f[{j}]"] = P["cop_f"][j]
        for j in range(2):
            q[f"cop_w[{j}]"] = P["cop_w"][j]
        q["owned"], q["elem"], q["guar"] = P["owned"][0], P["elem"][0], P["guar"][0]
        for j in range(2):
            q[f"wloss[{j}]"] = P["wloss"][j]
        for b in range(4):
            q[f"pity{b}[0]"] = P[f"pity{b}"][0]
            q[f"pity{b}[1]"] = P[f"pity{b}"][1]
    return q


SD_QUANTITIES = ("loy", "inc", "tenl", "repeat_med", "hl_dec", "hl_out")
HG_QUANTITIES = ("cop_f[0]", "cop_f[1]", "cop_f[2]", "cop_w[0]", "cop_w[1]", "owned", "elem",
                 "guar", "wloss[0]", "wloss[1]") + tuple(f"pity{b}[{j}]" for b in range(4) for j in (0, 1))


def quantities_with_se(spec: M.Spec, theta: torch.Tensor, V: Optional[torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Each quantity's value and delta-method SE (NaN-safe: unidentified rows of V count as 0)."""
    t = theta.clone().requires_grad_(True)
    q = quantities(spec, t)
    Vz = torch.nan_to_num(V, nan=0.0) if V is not None else None
    out = {}
    for k, v in q.items():
        g, = torch.autograd.grad(v, t, retain_graph=True)
        se = float(torch.sqrt(torch.clamp(g @ Vz @ g, min=0.0))) if Vz is not None else float("nan")
        out[k] = {"value": float(v.detach()), "se": se}
    return out


def shared_labels(labels: List[str]) -> np.ndarray:
    return np.array([not lab.split("[")[0] in SHARED_EXCLUDE for lab in labels])
