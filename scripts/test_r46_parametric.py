"""R46 unit tests: the state builder, the model pieces, and gate G3.

Laptop CPU only, synthetic data only, minutes not hours:

    python scripts/test_r46_parametric.py            # everything
    python scripts/test_r46_parametric.py --quick    # skip the recovery fits

  T1  decay() equals the explicit recursion
  T2  the nested logit with both dissimilarities at 1 is the MNL
  T3  a hand-built history gives the run, spell, guarantee, copy and outcome
      states worked out by hand
  T4  causality: every row's features, rebuilt from that row's prefix, equal
      the features built from the whole sequence
  T5  G3a: a lag-only MNL reproduces the closed-form Markov-1 MLE
  T6  G3: each rung recovers its own parameters on a panel simulated from itself
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parametric import data as D, fit as FT, model as M, simulate as SIM, state as ST  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def t1_decay():
    g = torch.Generator().manual_seed(0)
    X = torch.rand(3, 150, 4, generator=g, dtype=torch.float64)
    a = torch.tensor(0.7, dtype=torch.float64)
    ref = torch.zeros_like(X)
    L = torch.zeros(3, 4, dtype=torch.float64)
    for t in range(150):
        L = a * L + (1 - a) * X[:, t]
        ref[:, t] = L
    err = float((M.decay(X, a) - ref).abs().max())
    check("T1 decay closed form", err < 1e-12, f"max err {err:.1e}")


def t2_nested():
    g = torch.Generator().manual_seed(1)
    U = torch.randn(2, 5, 7, 9, generator=g, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    err = float((M.nested_logp(U, one, one) - torch.log_softmax(U, -1)).abs().max())
    s = float(M.nested_logp(U, torch.tensor(0.6, dtype=torch.float64),
                            torch.tensor(0.3, dtype=torch.float64)).exp().sum(-1).sub(1).abs().max())
    check("T2 nested logit = MNL at lambda 1, sums to 1 otherwise", err < 1e-12 and s < 1e-12,
          f"err {err:.1e}, sum err {s:.1e}")


def t3_hand(prod):
    pl = SIM.pools(prod)
    A, W1, W2 = int(pl["lim_fig"][0]), int(pl["lim_wep"][0]), int(pl["lim_wep"][1])
    std5 = int(pl["std5_fig"][0])
    three = int(pl["three"][0])
    tab = SIM.empty_table(1, 6)
    tab["lto"][0, :] = [A, 0, W1, W2]
    tab["campaign"][0, :] = 1
    tab["prev"][0] = [10, 3, 9, 9, 4, 1]
    tab["obtained"][0, 1, 0] = A                         # Buy1 A won the featured
    tab["obtained"][0, 4, :] = [three] * 9 + [std5]      # Buy10 A lost the 50/50
    tab["obtained"][0, 5, 0] = three                     # Buy1 regular, a 3-star
    F = ST.build(tab, prod)
    got = dict(lb=F["last_banner"][0].tolist(), r=(F["bawa"][0, :, 0] * 10).round().tolist(),
               nb=np.expm1(F["runs"][0, :, 0]).round().tolist(),
               br=np.expm1(F["runs"][0, :, 1]).round().tolist(),
               tr=np.expm1(F["runs"][0, :, 2]).round().tolist(),
               guar=F["guar"][0, :, 0].tolist(), cop=F["copies_a"][0, :, 0].tolist(),
               w5=F["ind"][0, :, 6].tolist(), wf=F["ind"][0, :, 7].tolist(),
               lost=F["ind"][0, :, 8].tolist(), first=F["phase"][0, :, 0].tolist())
    want = dict(lb=[-1, 1, 1, 1, 1, 0], r=[0, 1, 1, 1, 2, 1], nb=[0, 0, 1, 2, 0, 0],
                br=[0, 1, 0, 0, 1, 2], tr=[0, 0, 0, 0, 1, 0], guar=[0, 0, 0, 0, 1, 1],
                cop=[0, 1, 1, 1, 1, 1], w5=[0, 1, 0, 0, 1, 0], wf=[0, 1, 0, 0, 0, 0],
                lost=[0, 0, 0, 0, 1, 0], first=[1, 0, 0, 0, 0, 0])
    bad = {k: (got[k], want[k]) for k in want if [float(v) for v in got[k]] != [float(v) for v in want[k]]}
    check("T3 hand-built history", not bad, str(bad) if bad else "")


def t4_causal(prod):
    spec = M.Spec("R4")
    th = SIM.true_theta(spec, seed=3)
    tab, seen, cal = SIM.simulate_panel(spec, th, prod, N=40, S=60, seed=3, record_features=True)
    F = ST.build(tab, prod, calendar=cal)
    worst = 0.0
    for t, row in seen.items():
        for k, v in row.items():
            if k == "alive":
                continue
            worst = max(worst, float(np.abs(np.asarray(F[k][:, t], np.float64) - np.asarray(v, np.float64)).max()))
    check("T4 features of row t do not depend on later rows", worst == 0.0, f"max diff {worst}")


def _fit_on(tab, cal, prod, spec, theta0, *, se=True):
    F = ST.build(tab, prod, calendar=cal)
    roles = D.roles(tab, val_from=10_000)          # synthetic: every row is a fitting row
    users = np.arange(tab["label"].shape[0])
    chunks = FT.make_chunks(tab, F, roles, users, row_budget=40_000)
    n = int(roles["train"].sum())
    th, _ = FT.fit(spec, theta0, chunks, n, l2=0.01 / n)
    V = FT.sandwich(spec, th, chunks)[0] if se else None
    return th, V, F, roles, chunks, n


def t5_markov(prod):
    spec_sim = M.Spec("R2")
    tab, cal = SIM.simulate_panel(spec_sim, SIM.true_theta(spec_sim, seed=5), prod, N=200, S=80, seed=5)
    spec = M.Spec("M1")
    th, _, F, roles, chunks, n = _fit_on(tab, cal, prod, spec, spec.init(), se=False)
    nll = FT.objective(spec, th, chunks, n, 0.0, backward=False)
    cf = FT.markov_closed_form(tab, F, roles)
    # the N(0, 10^2) stability prior keeps never-seen transitions finite, which
    # costs the fit a little probability on them: allow 1e-4 nats
    check("T5 G3a lag-only MNL = closed-form Markov-1", 0 <= nll - cf < 1e-4,
          f"fitted {nll:.8f} closed {cf:.8f}")


def t6_recover(prod, rung, *, N, S, K=1, seed=11, report=False):
    t0 = time.time()
    spec = M.Spec(rung, K=K)
    th_true = SIM.true_theta(spec, seed=seed)
    if report:
        tab, seen, cal = SIM.simulate_panel(spec, th_true, prod, N=N, S=S, seed=seed, record_features=True)
        Ffull = ST.build(tab, prod, calendar=cal)
        diffs = {k: max(float(np.abs(np.asarray(Ffull[k][:, t], np.float64) - np.asarray(row[k], np.float64)).max())
                        for t, row in seen.items()) for k in seen[0] if k != "alive"}
        bad = {k: v for k, v in diffs.items() if v > 0}
        print(f"      causality on this panel: {'all features identical' if not bad else bad}")
        del seen
    else:
        tab, cal = SIM.simulate_panel(spec, th_true, prod, N=N, S=S, seed=seed)
    start = th_true + 0.05 * torch.randn(spec.size, dtype=torch.float64,
                                         generator=torch.Generator().manual_seed(seed)) \
        if K > 1 else spec.init(marginal=np.bincount(tab["label"].ravel() - 1, minlength=9) / tab["label"].size)
    th, V, F, roles, chunks, n = _fit_on(tab, cal, prod, spec, start)
    se = torch.sqrt(torch.diagonal(V)).numpy()
    ident = np.isfinite(se) & (se > 0)              # a feature with no variation is not identified
    z = np.where(ident, (th - th_true).numpy() / np.where(ident, se, 1.0), 0.0)
    off = spec.offsets()
    struct = [i for n in ("a_dec", "a_out", "rho", "nest", "cls", "d_asc") if n in off
              for i in range(*off[n]) if ident[i]]
    zs = np.abs(z[struct]).max() if struct else 0.0
    mz2 = float(np.mean(z[ident] ** 2))
    worst = spec_label(spec, int(np.argmax(np.abs(z))))
    ok = 0.25 < mz2 < 4.0 and np.abs(z).max() < 5.0 and zs < 3.5
    check(f"T6 G3 recovery {rung}{'' if K == 1 else f' K={K}'}", ok,
          f"P={spec.size} ({int((~ident).sum())} without variation) rows={N * S} mean z^2 {mz2:.2f} "
          f"max|z| {np.abs(z).max():.2f} at {worst}, structural max|z| {zs:.2f} ({time.time() - t0:.0f}s)")
    if report:
        ll_hat = -FT.objective(spec, th, chunks, 1.0, 0.0, backward=False)
        ll_true = -FT.objective(spec, th_true, chunks, 1.0, 0.0, backward=False)
        ll_start = -FT.objective(spec, start, chunks, 1.0, 0.0, backward=False)
        print(f"      total log-lik: estimate {ll_hat:.2f}, truth {ll_true:.2f}, start {ll_start:.2f}; "
              f"2(LL_hat - LL_true) = {2 * (ll_hat - ll_true):.1f} against chi2({int(ident.sum())}) "
              f"mean {int(ident.sum())}, sd {np.sqrt(2 * ident.sum()):.0f}"
              f"{'  <- OPTIMISER STOPPED SHORT' if ll_hat < ll_true - 1e-6 else ''}")
        g, H, _ = FT.grad_hess(spec, th, chunks)
        ii = np.nonzero(ident)[0]
        Vm = np.linalg.pinv(-H.numpy()[np.ix_(ii, ii)])
        ratio = np.sqrt(np.clip(np.diag(Vm), 1e-300, None)) / se[ii]
        zm = (th - th_true).numpy()[ii] / np.sqrt(np.clip(np.diag(Vm), 1e-300, None))
        print(f"      model-based SE / sandwich SE: median {np.median(ratio):.2f}, "
              f"range {ratio.min():.2f}-{ratio.max():.2f}; median z^2 with model-based SE {np.median(zm ** 2):.3f}")
        zi = z[ident]
        print(f"      median z^2 {np.median(zi ** 2):.3f} (chi2(1): 0.455); share |z|>1.96 "
              f"{np.mean(np.abs(zi) > 1.96):.3f} (0.05); share |z|>2.58 {np.mean(np.abs(zi) > 2.58):.3f} (0.01)")
        labs = FT.param_labels(spec)
        th_n, tt = th.numpy(), th_true.numpy()
        print(f"      {'parameter':<14}{'true':>8}{'est':>8}{'se':>8}{'z':>7}  rows where its feature is non-zero")
        nz = feature_support(spec, F, roles["train"])
        for i in np.argsort(-np.abs(z))[:10]:
            print(f"      {labs[i]:<14}{tt[i]:>8.3f}{th_n[i]:>8.3f}{se[i]:>8.3f}{z[i]:>7.2f}  {nz.get(labs[i], '-')}")


def feature_support(spec, F, mask) -> dict:
    """How many fitting rows carry a non-zero value of the feature behind each coefficient."""
    out = {}
    m = mask.astype(bool)
    blocks = {"fig": ("off_a", slice(0, 11)), "rr_a": ("off_a", slice(11, 13)),
              "rr_b": ("off_b", slice(11, 13)), "wep": ("off_w", slice(0, 6)),
              "phase": ("phase", slice(0, 3)), "bawa": ("bawa", slice(0, 2)),
              "runs": ("runs", slice(0, 3)), "cop_f": ("copies_a", slice(0, 3)),
              "cop_w": ("copies_w", slice(0, 2)), "wloss": ("wloss", slice(0, 2)),
              "guar": ("guar", slice(0, 1)), "elem": ("elem_a", slice(0, 1)),
              "owned": ("owned", slice(0, 1)), "unoff_b": ("unoff_b", slice(0, 1))}
    for n, (blk, sl) in blocks.items():
        if n in spec.names:
            v = F[blk][m][:, sl]
            for j in range(v.shape[1]):
                out[f"{n}[{j}]"] = int((v[:, j] != 0).sum())
    if "lag" in spec.names:
        p = F["prev_idx"][m]
        for a in range(10):
            for b in range(8):
                out[f"lag[{a},{b}]"] = int((p == a).sum())
    for b in range(4):
        if f"pity{b}" in spec.names:
            v = F["pity"][m][:, b]
            for j in range(3):
                out[f"pity{b}[{j}]"] = int((v[:, j] != 0).sum())
    return out


def spec_label(spec, i: int) -> str:
    return FT.param_labels(spec)[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="T1-T5 only")
    ap.add_argument("--only", default=None, help="run just one recovery fit, e.g. R0 or R6")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--N", type=int, default=None, help="override the recovery panel's customers")
    ap.add_argument("--S", type=int, default=None, help="override the recovery panel's rows")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    prod = ST.product_table()
    recov = {"R0": dict(N=300, S=80), "M1": dict(N=400, S=100), "R1": dict(N=400, S=100),
             "R1b": dict(N=400, S=100), "R3": dict(N=500, S=120),
             "R5": dict(N=700, S=120), "R6": dict(N=600, S=120, K=2)}
    if a.only:
        kw = dict(recov[a.only])
        kw.update({k: v for k, v in (("N", a.N), ("S", a.S)) if v})
        t6_recover(prod, a.only, seed=a.seed, report=True, **kw)
    else:
        t1_decay()
        t2_nested()
        t3_hand(prod)
        t4_causal(prod)
        t5_markov(prod)
        if not a.quick:
            for r, kw in recov.items():
                t6_recover(prod, r, **kw)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
