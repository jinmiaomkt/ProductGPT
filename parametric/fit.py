"""Estimate one R46 rung, score the four cells, write the results.

    python -m parametric.fit --table T.npz --feats F.npz --rung R3 --out-dir D
    python -m parametric.fit ... --rung R6 --K 4 --start 2 --init-from D/R5/result.json

Fit:   in-sample customers x campaigns < 27 (the neural train role).
Select: in-sample customers x campaign 27 (reported as `val`).
Score: the three test cells of build_loaders, plus `val`.

CPU only. Threads follow $NCPUS (PBS) or --threads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parametric import data as D          # noqa: E402
from parametric import model as M         # noqa: E402
from parametric import state as ST        # noqa: E402

F64 = torch.float64
REV = np.array([1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=np.float64)
EVAL_ROLES = ("val",) + D.CELLS
INT_KEYS = ("prev_idx", "last_banner")


@dataclass
class Chunk:
    users: np.ndarray
    X: Dict[str, torch.Tensor]
    y: torch.Tensor
    masks: Dict[str, torch.Tensor]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────────── data ─────────────────────────────
def load_features(table: Path, feats: Path, *, build_if_missing: bool = False):
    tab = D.load(table)
    if feats.exists():
        with np.load(feats) as z:
            F = {k: z[k] for k in z.files}
    elif build_if_missing:
        log("building state features")
        F = ST.build(tab, ST.product_table())
        np.savez(feats, **F)
        log(f"wrote {feats}")
    else:
        raise SystemExit(f"{feats} missing: run the export job first")
    return tab, F


def make_chunks(tab, F, roles: Dict[str, np.ndarray], users: np.ndarray,
                row_budget: int = 196_608) -> List[Chunk]:
    """Users sorted by length, cut so each chunk holds about `row_budget` padded rows."""
    length = tab["length"]
    users = users[np.argsort(length[users], kind="stable")]
    out, i = [], 0
    while i < len(users):
        j = i
        while j < len(users):
            S = int(math.ceil(length[users[j]] / 64) * 64)
            if (j - i + 1) * S > row_budget and j > i:
                break
            j += 1
        u = users[i:j]
        S = int(math.ceil(length[u].max() / 64) * 64)
        X = {}
        for k, v in F.items():
            if k == "alive":
                continue
            a = v[u, :S]
            X[k] = torch.as_tensor(a, dtype=torch.long if k in INT_KEYS else F64)
        y = torch.as_tensor(tab["label"][u, :S].astype(np.int64))
        masks = {k: torch.as_tensor(m[u, :S]) for k, m in roles.items()}
        out.append(Chunk(u, X, y, masks))
        i = j
    return out


# ───────────────────────────── estimation ─────────────────────────────
def objective(spec, theta, chunks, n_rows, l2, *, backward: bool) -> float:
    tot = 0.0
    for ch in chunks:
        ll = M.customer_ll(spec, theta, ch.X, ch.y, ch.masks["train"]).sum()
        loss = -ll / n_rows
        if backward:
            loss.backward()
        tot += float(loss.detach())
    reg = 0.5 * l2 * (theta ** 2).sum()
    if backward:
        reg.backward()
    return tot + float(reg.detach())


def grad_hess(spec, theta, chunks, *, scores: bool = False):
    """Gradient and exact Hessian of the TOTAL masked log-likelihood (and per-customer scores).

    g = d/dtheta sum_i w_i ll_i ; dg_p/dtheta = Hessian row p ; dg_p/dw_i = s_ip.
    One double-backward per parameter per chunk.
    """
    P = theta.numel()
    G = torch.zeros(P, dtype=F64)
    H = torch.zeros(P, P, dtype=F64)
    out_s = []
    th = theta.clone().requires_grad_(True)
    for ch in chunks:
        w = torch.ones(len(ch.users), dtype=F64, requires_grad=True)
        ll = M.customer_ll(spec, th, ch.X, ch.y, ch.masks["train"])
        g, = torch.autograd.grad((w * ll).sum(), th, create_graph=True)
        G += g.detach()
        Sc = torch.zeros(len(ch.users), P, dtype=F64) if scores else None
        for p in range(P):
            h, s = torch.autograd.grad(g[p], (th, w), retain_graph=True, allow_unused=True)
            if h is not None:
                H[p] += h.detach()
            if scores and s is not None:
                Sc[:, p] = s.detach()
        if scores:
            out_s.append(Sc[ch.masks["train"].any(-1)])
    return G, 0.5 * (H + H.T), (torch.cat(out_s, 0) if scores else None)


def newton(spec, theta, chunks, n_rows, *, lam, iters=8, tol=1e-6):
    """Damped Newton on -LL + lam/2 |theta|^2 (total scale), with backtracking.

    L-BFGS crawls when a few never-observed transitions push their utilities
    toward -inf; the exact Hessian finishes the identified directions.
    """
    f = lambda t: n_rows * objective(spec, t, chunks, n_rows, lam / n_rows, backward=False)
    fx = f(theta)
    hist = []
    for it in range(iters):
        g, H, _ = grad_hess(spec, theta, chunks)
        grad = -g + lam * theta
        A = -H + lam * torch.eye(theta.numel(), dtype=F64)
        gmax = float(grad.abs().max())
        if gmax < tol:
            hist.append({"newton": it, "f": fx, "grad_max": gmax})
            break
        # Newton only where the surface is locally concave. On an indefinite
        # Hessian (a ridge, e.g. a nest dissimilarity traded against free
        # pack constants) a damped step walks along the ridge -- G3's R5 did
        # exactly that -- so stop and keep the L-BFGS point instead.
        try:
            L = torch.linalg.cholesky(A)
        except RuntimeError:
            hist.append({"newton": it, "f": fx, "grad_max": gmax, "stopped": "hessian not negative definite"})
            log(f"  newton {it}: Hessian not negative definite -- keeping the L-BFGS point")
            break
        mu = 0.0
        d = -torch.cholesky_solve(grad[:, None], L)[:, 0]
        step, ok = 1.0, False
        for _ in range(30):
            f_new = f(theta + step * d)
            if f_new <= fx - 1e-4 * step * float(-(grad @ d)):
                ok = True
                break
            step *= 0.5
        hist.append({"newton": it, "f": fx, "grad_max": gmax, "step": step})
        log(f"  newton {it}  total -LL+pen {fx:.6f}  max|grad| {gmax:.2e}  step {step:g}")
        if not ok or step < 0.25:
            # a short step on a concave patch means the quadratic model is poor here
            if ok:
                theta, fx = theta + step * d, f_new
            break
        theta = theta + step * d
        fx = f_new
    return theta, hist


def fit(spec, theta0, chunks, n_rows, *, l2, max_rounds=15, tol=1e-10, newton_iters=8):
    """L-BFGS on mean NLL + l2/2 |theta|^2, then damped Newton polish.

    `l2` is per row; callers pass ridge / n_rows so the prior precision on the
    TOTAL likelihood is `ridge` whatever the sample size.
    """
    theta = theta0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=100, history_size=50,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-13)

    def closure():
        opt.zero_grad()
        return torch.tensor(objective(spec, theta, chunks, n_rows, l2, backward=True), dtype=F64)

    prev, hist = float("inf"), []
    for r in range(max_rounds):
        opt.step(closure)
        loss = float(closure())        # the loss AFTER the step, and a fresh gradient
        gnorm = float(theta.grad.abs().max())
        hist.append({"round": r, "loss": loss, "grad_max": gnorm})
        log(f"  round {r:>2}  train NLL {loss:.8f}  max|grad| {gnorm:.2e}")
        if abs(prev - loss) < tol or gnorm < 1e-7:
            break
        prev = loss
    th = theta.detach()
    if newton_iters:
        th, nh = newton(spec, th, chunks, n_rows, lam=l2 * n_rows, iters=newton_iters)
        hist += nh
    return th, hist


def sandwich(spec, theta, chunks):
    """Exact Hessian and per-customer scores in one pass of P double-backwards per chunk.

    g = d/dtheta sum_i w_i ll_i ; dg_p/dtheta = Hessian row p ; dg_p/dw_i = s_ip.
    Cluster-robust covariance V = A^-1 (S'S) A^-1 with A = -H, clustered by customer.
    """
    P = theta.numel()
    _, H, Smat = grad_hess(spec, theta, chunks, scores=True)
    A = -H
    # A parameter whose feature never varies on the fitting rows has a zero row
    # in A: the data say nothing about it. It keeps its (penalised) start value
    # and is reported as not identified (SE = NaN), never as a precise zero.
    d = torch.diagonal(A)
    ident = d > 1e-10 * float(d.abs().max())
    ii = torch.nonzero(ident).flatten()
    Ai = A[ii][:, ii]
    evals, evecs = torch.linalg.eigh(Ai)
    # Near-flat directions: a coefficient that loads on them sits on a ridge
    # (traded against others at almost no cost in likelihood), and its Wald SE
    # is not to be trusted -- report it as weakly identified.
    flat = evals < 1e-4 * float(evals.abs().median())
    load = (evecs[:, flat] ** 2).sum(1) if bool(flat.any()) else torch.zeros(len(ii), dtype=F64)
    ridge = [int(ii[k]) for k in torch.nonzero(load > 0.1).flatten()]
    Ainv = torch.linalg.pinv(Ai, hermitian=True)
    Si = Smat[:, ii]
    V = torch.full((P, P), float("nan"), dtype=F64)
    V[ii[:, None], ii[None, :]] = Ainv @ (Si.T @ Si) @ Ainv
    return V, {"min_eig": float(evals.min()), "max_eig": float(evals.max()),
               "n_clusters": int(Smat.shape[0]), "negative_definite": bool(evals.min() > 0),
               "not_identified": [int(i) for i in torch.nonzero(~ident).flatten()],
               "ridge": ridge}


# ───────────────────────────── evaluation ─────────────────────────────
@torch.no_grad()
def score(spec, theta, chunks, roles_np, tab, *, prior_only=False):
    """Per-occasion predictive probabilities for every evaluation role."""
    out = {r: {"uid_hash": [], "pos": [], "label": [], "prob": []} for r in EVAL_ROLES}
    for ch in chunks:
        lp = M.predictive(spec, theta, ch.X, ch.y, ch.masks["valid"], prior_only=prior_only)
        pr = lp.exp().numpy()
        for r in EVAL_ROLES:
            m = ch.masks[r].numpy()
            if not m.any():
                continue
            bi, ti = np.nonzero(m)
            out[r]["uid_hash"].append(tab["uid_hash"][ch.users[bi]])
            out[r]["pos"].append(ti.astype(np.int16))
            out[r]["label"].append(ch.y.numpy()[bi, ti].astype(np.int8))
            out[r]["prob"].append(pr[bi, ti].astype(np.float64))
    res = {}
    for r, d in out.items():
        if not d["label"]:
            continue
        res[r] = {k: np.concatenate(v) for k, v in d.items()}
    return res


def metrics(y: np.ndarray, P: np.ndarray) -> Dict:
    from sklearn.metrics import average_precision_score
    n = len(y)
    P = P / P.sum(1, keepdims=True)
    idx = y.astype(np.int64) - 1
    nll = -np.log(np.clip(P[np.arange(n), idx], 1e-12, None))
    pred = P.argmax(1) + 1
    tp = np.array([((pred == k) & (y == k)).sum() for k in range(1, 10)])
    pc = np.array([(pred == k).sum() for k in range(1, 10)])
    tc = np.array([(y == k).sum() for k in range(1, 10)])
    f1 = []
    for k in range(9):
        pr = tp[k] / pc[k] if pc[k] else 0.0
        rc = tp[k] / tc[k] if tc[k] else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    ap = [float(average_precision_score((y == k + 1).astype(np.int8), P[:, k]))
          if tc[k] else float("nan") for k in range(9)]
    conf = P.max(1)
    correct = (pred == y)
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).sum()
              for b in range(15) if (bins == b).any()) / n
    onehot = np.zeros_like(P)
    onehot[np.arange(n), idx] = 1.0
    true_dist = tc / n
    return {
        "n": int(n), "nll": float(nll.mean()),
        "nll_purchase": float(nll[y <= 8].mean()), "nll_notbuy": float(nll[y == 9].mean()),
        "nll_per_class": {str(k): float(nll[y == k].mean()) for k in range(1, 10) if tc[k - 1]},
        "hit": float(correct.mean()), "f1_macro": float(np.mean(f1)),
        "auprc_macro": float(np.nanmean(ap)), "auprc_per_class": ap,
        "rev_mae": float(np.abs(P @ REV - REV[idx]).mean()),
        "ece": float(ece), "brier": float(((P - onehot) ** 2).sum(1).mean()),
        "true_dist": [round(float(v), 6) for v in true_dist],
        "pred_dist": [round(float(v), 6) for v in P.mean(0)],
        "marginal_entropy": float(-(true_dist[true_dist > 0] * np.log(true_dist[true_dist > 0])).sum()),
    }


def markov_closed_form(tab, F, roles) -> float:
    """G3a: the Markov-1 MLE NLL on the train rows, from counts."""
    m = roles["train"]
    prev = F["prev_idx"][m]
    y = tab["label"][m].astype(np.int64)
    c = np.zeros((10, 9))
    np.add.at(c, (prev, y - 1), 1.0)
    p = c / c.sum(1, keepdims=True).clip(min=1)
    nz = c > 0
    return float(-(c[nz] * np.log(p[nz])).sum() / m.sum())


def transformed(spec, theta, V) -> Dict[str, Dict[str, float]]:
    """Half-lives, nest dissimilarities and rho, with delta-method SEs."""
    off = spec.offsets()
    out: Dict = {}
    sig = torch.sigmoid
    Vz = torch.nan_to_num(V, nan=0.0) if V is not None else None

    def add(name, fn):
        t = theta.clone().requires_grad_(True)
        v = fn(t)
        g, = torch.autograd.grad(v, t)
        se = float(torch.sqrt(g @ Vz @ g)) if Vz is not None else float("nan")
        out[name] = {"value": float(v), "se": se}

    for n, label in (("a_dec", "decisions"), ("a_out", "outcomes")):
        if n in off:
            i = off[n][0]
            add(f"alpha_{label}", lambda t, i=i: sig(t[i]))
            add(f"half_life_{label}", lambda t, i=i: math.log(0.5) / torch.log(sig(t[i])))
    if "nest" in off:
        i = off["nest"][0]
        add("lambda_top", lambda t: sig(t[i]))
    if "rho" in off:
        i = off["rho"][0]
        add("rho", lambda t: sig(t[i]))
    if "cls" in off:
        a, b = off["cls"]
        w = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), theta[a:b]]), 0)
        out["class_weights"] = [float(v) for v in w]
    return out


RARE_ROWS, RARE_EVENTS = 1_000, 50
NEST_STARTS = [0.9, 0.7, 0.5, 0.3]
_BLOCKS = {"fig": [("off_a", slice(0, 11), (3, 4)), ("off_b", slice(0, 11), (5, 6))],
           "rr_a": [("off_a", slice(11, 12), (3, 4))], "rr_b": [("off_b", slice(11, 12), (5, 6))],
           "unoff_b": [("unoff_b", slice(0, 1), (5, 6))], "wep": [("off_w", slice(0, 5), (7, 8))],
           "phase": [("phase", slice(0, 3), tuple(range(1, 9)))],
           "cop_f": [("copies_a", slice(0, 3), (3, 4)), ("copies_b", slice(0, 3), (5, 6))],
           "cop_w": [("copies_w", slice(0, 2), (7, 8))], "wloss": [("wloss", slice(0, 2), (7, 8))],
           "guar": [("guar", slice(0, 1), (3, 4, 5, 6))],
           "elem": [("elem_a", slice(0, 1), (3, 4)), ("elem_b", slice(0, 1), (5, 6))],
           "owned": [("owned", slice(0, 1), tuple(range(1, 9)))]}


def support(spec, F, y: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, int]]:
    """Per coefficient: fitting rows where its feature is non-zero, and the
    EVENTS among them (choices of an alternative the coefficient acts on).

    A coefficient on a rare state (rows < 1,000 or events < 50) has an
    unreliable sandwich SE -- near-separation makes the score variance tiny --
    and is flagged; part C's customer bootstrap is the interval to use.
    """
    m = mask.astype(bool)
    yy = y[m]
    out: Dict[str, Dict[str, int]] = {}

    def put(label, nz, alts):
        d = out.setdefault(label, {"rows": 0, "events": 0})
        d["rows"] += int(nz.sum())
        d["events"] += int((nz & np.isin(yy, alts)).sum())

    blocks = dict(_BLOCKS)
    if "runs" in spec.names:
        blocks["runs"] = [("runs", slice(0, 2), tuple(range(1, 9))), ("runs", slice(2, 3), (2, 4, 6, 8))]
    for n, parts in blocks.items():
        if n not in spec.names:
            continue
        j0 = 0
        for blk, sl, alts in parts:
            v = F[blk][m][:, sl]
            same_coef = n in ("fig", "cop_f", "elem")      # A and B share these coefficients
            for j in range(v.shape[1]):
                put(f"{n}[{j if same_coef else j0 + j}]", v[:, j] != 0, alts)
            if not same_coef:
                j0 += v.shape[1]
    if "lag" in spec.names:
        p = F["prev_idx"][m]
        for a in range(10):
            for b in range(8):
                put(f"lag[{a},{b}]", p == a, (b + 1,))
    for b in range(4):
        if f"pity{b}" in spec.names:
            v = F["pity"][m][:, b]
            for j in range(3):
                put(f"pity{b}[{j}]", v[:, j] != 0, (2 * b + 1, 2 * b + 2))
    if "bawa" in spec.names:
        lb = F["last_banner"][m]
        yb = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, -1])[yy]
        for j in range(2):
            nz = F["bawa"][m][:, j] != 0
            d = out.setdefault(f"bawa[{j}]", {"rows": 0, "events": 0})
            d["rows"] += int(nz.sum())
            d["events"] += int((nz & (yb == lb)).sum())
    for d in out.values():
        d["rare"] = d["rows"] < RARE_ROWS or d["events"] < RARE_EVENTS
    return out


def param_labels(spec) -> List[str]:
    labs = []
    for n, s in zip(spec.names, spec.shapes):
        for ix in np.ndindex(*s):
            labs.append(f"{n}[{','.join(map(str, ix))}]")
    return labs


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--rung", required=True, choices=M.RUNGS)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--init-from", default=None, help="result.json to warm-start shared parameters")
    ap.add_argument("--theta-from", default=None,
                    help="result.json of THIS rung: start at its estimate (confirm convergence, add SEs)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--se", action="store_true", help="cluster-robust sandwich SEs")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NCPUS", "8")))
    ap.add_argument("--ridge", type=float, default=0.01,
                    help="prior precision on the TOTAL likelihood, N(0, 1/ridge) per coefficient; "
                         "for numerical stability only (bounds never-observed transitions)")
    ap.add_argument("--build-features", action="store_true")
    ap.add_argument("--max-users", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    tag = {"R5": f"R5_s{a.start}", "R6": f"R6_K{a.K}_s{a.start}"}.get(a.rung, a.rung)
    out = Path(a.out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    log(f"R46 fit {tag}  threads={a.threads}  torch {torch.__version__}")

    tab, F = load_features(Path(a.table), Path(a.feats), build_if_missing=a.build_features)
    roles = D.roles(tab)
    N = tab["label"].shape[0]
    allu = np.arange(N if not a.max_users else min(N, a.max_users))
    inu = allu[tab["is_out"][allu] == 0]
    n_train = int(roles["train"][inu].sum())
    log(f"users {len(allu)} (in-sample {len(inu)}), train rows {n_train:,}")

    spec = M.Spec(a.rung, K=a.K)
    marg = np.bincount(tab["label"][roles["train"]].astype(np.int64) - 1, minlength=9) / n_train
    theta0 = spec.init(marginal=marg)
    if a.rung == "R5":
        # The nest likelihood is not globally concave, so R5 is fitted from
        # four starting dissimilarities and the best TRAINING likelihood is
        # kept (compare.py and the R6 warm start apply the same rule).
        lt = NEST_STARTS[a.start % len(NEST_STARTS)]
        theta0[spec.offsets()["nest"][0]] = math.log(lt / (1 - lt))
        log(f"nest start {a.start}: lambda_top {lt}")
    if a.init_from:
        if "*" in a.init_from:
            import glob
            cands = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(a.init_from))]
            if not cands:
                raise SystemExit(f"--init-from {a.init_from}: no result.json matched")
            src = min(cands, key=lambda r: r["train_nll"])
            log(f"warm start: best of {len(cands)} by training NLL is {src['tag']} ({src['train_nll']:.6f})")
        else:
            src = json.loads(Path(a.init_from).read_text())
        src_spec = M.Spec(src["rung"], K=src.get("K", 1))
        src_p = src_spec.unpack(torch.tensor(src["theta"], dtype=F64))
        off = spec.offsets()
        for n, (lo, hi) in off.items():
            if n in src_p and src_p[n].numel() == hi - lo:
                theta0[lo:hi] = src_p[n].reshape(-1)
        g = torch.Generator().manual_seed(1000 + a.start)
        for n in ("d_asc", "d_inc", "d_loy", "d_nb"):
            if n in off:
                lo, hi = off[n]
                theta0[lo:hi] = 0.5 * torch.randn(hi - lo, generator=g, dtype=F64)
        log(f"warm start from {a.init_from}; class deviations ~ N(0, 0.5^2), seed {1000 + a.start}")
    if a.theta_from:
        src = json.loads(Path(a.theta_from).read_text())
        if src["rung"] != a.rung or src.get("K", 1) != spec.K:
            raise SystemExit(f"--theta-from is {src['rung']} K={src.get('K')}, not {a.rung} K={spec.K}")
        theta0 = torch.tensor(src["theta"], dtype=F64)
        log(f"starting at the estimate in {a.theta_from}")
    log(f"rung {a.rung}: {spec.size} parameters  {dict(zip(spec.names, spec.shapes))}")

    train_chunks = make_chunks(tab, F, roles, inu)
    log(f"{len(train_chunks)} training chunks")
    theta, hist = fit(spec, theta0, train_chunks, n_train, l2=a.ridge / n_train,
                      max_rounds=2 if a.theta_from else 15)
    train_nll = objective(spec, theta, train_chunks, n_train, 0.0, backward=False)
    log(f"fitted: train NLL {train_nll:.6f}  ({time.time() - t0:.0f}s)")

    res = {"rung": a.rung, "K": spec.K, "start": a.start, "tag": tag,
           "n_params": spec.size, "names": spec.names, "shapes": spec.shapes,
           "labels": param_labels(spec), "theta": theta.tolist(),
           "train_nll": train_nll, "n_train": n_train, "fit_history": hist,
           "ridge": a.ridge, "max_users": a.max_users}
    m_tr = np.zeros_like(roles["train"])
    m_tr[inu] = roles["train"][inu]
    sup = support(spec, F, tab["label"].astype(np.int64), m_tr)
    res["support"] = sup
    res["rare"] = sorted(k for k, v in sup.items() if v["rare"])
    log(f"rare-state coefficients (rows < {RARE_ROWS} or events < {RARE_EVENTS}): {len(res['rare'])}")
    if a.rung == "M1":
        cf = markov_closed_form(tab, F, {"train": m_tr})
        res["G3a_markov_closed_form"] = cf
        res["G3a_gap"] = train_nll - cf
        log(f"G3a: fitted {train_nll:.8f} vs closed form {cf:.8f}  gap {train_nll - cf:+.2e}")

    V = None
    if a.se:
        log("sandwich standard errors")
        V, diag = sandwich(spec, theta, train_chunks)
        res["se"] = torch.sqrt(torch.clamp(torch.diagonal(V), min=0)).tolist()
        res["se_diag"] = diag
        res["ridge"] = [res["labels"][i] for i in diag["ridge"]]
        log(f"Hessian negative definite: {diag['negative_definite']}; ridge coefficients: {res['ridge']}")
        torch.save(V, out / "cov.pt")
    res["derived"] = transformed(spec, theta, V)

    del train_chunks
    eval_chunks = make_chunks(tab, F, roles, allu)
    scored = score(spec, theta, eval_chunks, roles, tab)
    res["cells"] = {}
    for r, d in scored.items():
        res["cells"][r] = metrics(d["label"].astype(np.int64), d["prob"])
        np.savez_compressed(out / f"occ_{r}.npz", uid_hash=d["uid_hash"], pos=d["pos"],
                            label=d["label"], prob=d["prob"].astype(np.float32))
        log(f"  {r:<32} n={res['cells'][r]['n']:>9,}  NLL {res['cells'][r]['nll']:.4f}  "
            f"buy {res['cells'][r]['nll_purchase']:.4f}  NotBuy {res['cells'][r]['nll_notbuy']:.4f}")
    if spec.K > 1:
        pri = score(spec, theta, eval_chunks, roles, tab, prior_only=True)
        res["cells_prior_weights"] = {r: metrics(d["label"].astype(np.int64), d["prob"])
                                      for r, d in pri.items()}
    res["seconds"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(res, indent=1))
    log(f"done {tag} in {res['seconds']:.0f}s -> {out / 'result.json'}")


if __name__ == "__main__":
    main()
