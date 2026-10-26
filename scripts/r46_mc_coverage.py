"""R46 G3 follow-up: do the sandwich SEs cover? A Monte Carlo over panels.

One replication cannot calibrate standard errors: correlated estimates make
its marginal z's cluster. This draws many panels from ONE fixed parameter
vector, fits each, and compares the spread of the estimates with the average
sandwich SE, coefficient by coefficient.

    # one replication (a PBS array element)
    python scripts/r46_mc_coverage.py --rung R1b --N 800 --S 120 --rep 7 --out-dir D
    # read them all
    python scripts/r46_mc_coverage.py --aggregate D/R1b

Synthetic data only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parametric import fit as FT, model as M, simulate as SIM, state as ST   # noqa: E402
import test_r46_parametric as T                                              # noqa: E402

THETA_SEED = 11


def one(rung: str, N: int, S: int, rep: int, K: int, out: Path) -> None:
    t0 = time.time()
    spec = M.Spec(rung, K=K)
    prod = ST.product_table()
    th_true = SIM.true_theta(spec, seed=THETA_SEED)
    tab, cal = SIM.simulate_panel(spec, th_true, prod, N=N, S=S, seed=1000 + rep)
    start = th_true + 0.05 * torch.randn(spec.size, dtype=torch.float64,
                                         generator=torch.Generator().manual_seed(rep)) \
        if K > 1 else spec.init(marginal=np.bincount(tab["label"].ravel() - 1, minlength=9) / tab["label"].size)
    th, V, F, roles, chunks, n = T._fit_on(tab, cal, prod, spec, start)
    se = torch.sqrt(torch.diagonal(V)).numpy()
    sup = FT.support(spec, F, tab["label"].astype(np.int64), roles["train"])
    labs = FT.param_labels(spec)
    rare = np.array([sup.get(l, {}).get("rare", False) for l in labs])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"rep{rep:03d}.npz", theta_hat=th.numpy(), theta_true=th_true.numpy(),
             se=se, V=V.numpy(), rare=rare, labels=np.array(labs))
    print(f"rep {rep}: {rung} N={N} S={S} done in {time.time() - t0:.0f}s")


def aggregate(d: Path) -> None:
    from scipy.stats import chi2
    files = sorted(d.glob("rep*.npz"))
    Z = [np.load(f, allow_pickle=False) for f in files]
    th = np.stack([z["theta_hat"] for z in Z])
    se = np.stack([z["se"] for z in Z])
    tt = Z[0]["theta_true"]
    labs = Z[0]["labels"]
    rare = np.stack([z["rare"] for z in Z]).mean(0) > 0.5
    ok = np.all(np.isfinite(se) & (se > 0), axis=0)
    R = len(Z)
    emp_sd = th.std(0, ddof=1)
    mean_se = np.nanmean(np.where(se > 0, se, np.nan), 0)
    ratio = emp_sd / mean_se
    cover = (np.abs(th - tt) <= 1.96 * se).mean(0)
    bias_z = (th.mean(0) - tt) / (emp_sd / np.sqrt(R))
    sel = ok & ~rare
    W = []
    for z in Z:
        c = np.nonzero(sel)[0]
        dlt = (z["theta_hat"] - tt)[c]
        W.append(float(dlt @ np.linalg.pinv(z["V"][np.ix_(c, c)]) @ dlt))
    k = int(sel.sum())
    print(f"{d.name}: {R} replications, {k} non-rare identified coefficients")
    print(f"  empirical sd / mean sandwich SE: median {np.median(ratio[sel]):.3f}, "
          f"IQR {np.quantile(ratio[sel], 0.25):.3f}-{np.quantile(ratio[sel], 0.75):.3f}, "
          f"max {ratio[sel].max():.2f} at {labs[sel][np.argmax(ratio[sel])]}")
    print(f"  95% CI coverage: median {np.median(cover[sel]):.3f}, mean {cover[sel].mean():.3f} (nominal 0.95)")
    print(f"  bias: share of |mean bias / MC se| > 2.58: {np.mean(np.abs(bias_z[sel]) > 2.58):.3f} (0.01)")
    print(f"  joint Wald: mean {np.mean(W):.1f} against chi2({k}) mean {k}; "
          f"share p < 0.05: {np.mean([chi2.sf(w, k) < 0.05 for w in W]):.2f}")
    worst = np.argsort(-ratio * sel)[:8]
    print(f"  {'coefficient':<14}{'true':>8}{'mean est':>10}{'emp sd':>9}{'mean se':>9}{'ratio':>7}{'cover':>7}")
    for i in worst:
        print(f"  {labs[i]:<14}{tt[i]:>8.3f}{th[:, i].mean():>10.3f}{emp_sd[i]:>9.3f}{mean_se[i]:>9.3f}"
              f"{ratio[i]:>7.2f}{cover[i]:>7.2f}")
    (d / "coverage.json").write_text(json.dumps({
        "replications": R, "k": k, "ratio_median": float(np.median(ratio[sel])),
        "coverage_mean": float(cover[sel].mean()), "wald_mean": float(np.mean(W))}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--S", type=int, default=120)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out-dir")
    ap.add_argument("--aggregate")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.aggregate:
        aggregate(Path(a.aggregate))
        return
    tag = a.rung if a.K == 1 else f"{a.rung}_K{a.K}"
    one(a.rung, a.N, a.S, a.rep, a.K, Path(a.out_dir) / tag)


if __name__ == "__main__":
    main()
