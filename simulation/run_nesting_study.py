"""
Does the attention inventory recover the classical models when they are true?

    python simulation/run_nesting_study.py            # E1, E2, E3
    python simulation/run_nesting_study.py --quick

E1  truth = Guadagni-Little (kappa = I, geometric decay). Does the LEARNED
    model recover the smoothing constant, and does its kernel stay diagonal?
E2  truth = McAlister (kappa = attribute similarity). Does it recover the
    attribute structure?
E3  the same truth as E2 with a ROTATING assortment instead of full
    availability -- the general form of the limited-time-product problem.

Ordinary brand choice throughout: households pick one brand and receive it.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brand_choice_sim import BrandConfig, fit, simulate  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "nesting"


def spearman(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def offdiag(m):
    return m[~np.eye(m.shape[0], dtype=bool)]


def attr_lift(k, attr_of):
    same = attr_of[:, None] == attr_of[None, :]
    off = ~np.eye(len(attr_of), dtype=bool)
    return float(k[same & off].mean() / k[~same & off].mean())


def report(name, rows, keys):
    print(f"\n{name}")
    print("  " + "  ".join(k.rjust(max(11, len(k))) for k in keys))
    for r in rows:
        print("  " + "  ".join(
            (f"{r[k]:>{max(11, len(k))}.3f}" if isinstance(r.get(k), float)
             else str(r.get(k, "")).rjust(max(11, len(k)))) for k in keys))


def run(cfg, epochs, seeds=(0, 1, 2)):
    out = []
    for sd in seeds:
        d = simulate(replace(cfg, seed=100 + sd))
        r = fit(d, kernel="learned", epochs=epochs, seed=sd)
        r["spearman"] = spearman(offdiag(r["kappa_hat"]), offdiag(d.kappa_true))
        r["attr_lift"] = attr_lift(r["kappa_hat"], d.attr_of)
        r["true_half_life"] = cfg.half_life()
        out.append(r)
    return out


def agg(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    epochs = 120 if a.quick else 400
    # taste_sd = 0 in the core experiments: first ask whether the estimator can
    # recover a correctly specified truth. E2b then switches heterogeneity on,
    # which is the contamination R34 s2 found with the gacha data.
    base = BrandConfig(n_households=300 if a.quick else 800,
                       n_occasions=100 if a.quick else 200, taste_sd=0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    print("=" * 92)
    print("E1  truth = GUADAGNI-LITTLE: kappa = I, loyalty decays geometrically")
    print("=" * 92)
    rows = []
    for alpha in ([0.6, 0.9] if a.quick else [0.5, 0.7, 0.9]):
        cfg = replace(base, kernel="identity", alpha=alpha, gamma=1.5)
        rs = run(cfg, epochs)
        rows.append({"true alpha": alpha, "alpha hat": agg(rs, "alpha_hat"),
                     "true half-life": cfg.half_life(),
                     "half-life hat": agg(rs, "half_life_hat"),
                     "gamma hat": agg(rs, "gamma_hat"),
                     "off-diag mass": agg(rs, "offdiag_mass")})
        print(f"  alpha {alpha}: recovered {rows[-1]['alpha hat']:.3f}   "
              f"off-diagonal mass {rows[-1]['off-diag mass']:.3f}", flush=True)
    report("E1 (true gamma 1.5; off-diagonal mass near 0 means the kernel stayed diagonal)",
           rows, ["true alpha", "alpha hat", "true half-life", "half-life hat",
                  "gamma hat", "off-diag mass"])
    res["E1"] = rows

    print("\n" + "=" * 92)
    print("E2  truth = McALISTER: kappa = attribute similarity, satiation across a group")
    print("=" * 92)
    rows = []
    for w in ([2.0] if a.quick else [1.0, 2.0, 3.0]):
        cfg = replace(base, kernel="attr", attr_weight=w, gamma=-1.5, alpha=0.9)
        rs = run(cfg, epochs)
        rows.append({"attr weight": w, "spearman": agg(rs, "spearman"),
                     "attr lift": agg(rs, "attr_lift"),
                     "alpha hat": agg(rs, "alpha_hat"),
                     "gamma hat": agg(rs, "gamma_hat")})
        print(f"  attribute weight {w}: kernel recovery {rows[-1]['spearman']:+.3f}, "
              f"lift {rows[-1]['attr lift']:.2f}", flush=True)
    report("E2 (true gamma -1.5 = satiation; lift > 1 means the attribute structure was found)",
           rows, ["attr weight", "spearman", "attr lift", "alpha hat", "gamma hat"])
    res["E2"] = rows

    print("=" * 92)
    print("E2b  the same truth, with UNOBSERVED taste heterogeneity the estimator lacks")
    print("=" * 92)
    rows = []
    for sd in ([0.0, 0.5] if a.quick else [0.0, 0.25, 0.5, 1.0]):
        cfg = replace(base, kernel="attr", attr_weight=2.0, gamma=-1.5, alpha=0.9,
                      taste_sd=sd)
        rs = run(cfg, epochs)
        rows.append({"taste sd": sd, "spearman": agg(rs, "spearman"),
                     "attr lift": agg(rs, "attr_lift"),
                     "off-diag mass": agg(rs, "offdiag_mass"),
                     "gamma hat": agg(rs, "gamma_hat")})
        print(f"  taste sd {sd}: kernel recovery {rows[-1]['spearman']:+.3f}, "
              f"lift {rows[-1]['attr lift']:.2f}", flush=True)
    report("E2b (heterogeneity the estimator does not model)", rows,
           ["taste sd", "spearman", "attr lift", "off-diag mass", "gamma hat"])
    res["E2b"] = rows

    print("\n" + "=" * 92)
    print("E3  the same truth, but the assortment ROTATES (limited-time products)")
    print("=" * 92)
    rows = []
    for avail, size, every, label in (("all", 12, 0, "all brands always available"),
                                      ("rotating", 4, 10, "4 of 12, rotating every 10"),
                                      ("rotating", 4, 30, "4 of 12, rotating every 30")):
        cfg = replace(base, kernel="attr", attr_weight=2.0, gamma=-1.5, alpha=0.9,
                      availability=avail, avail_size=size, rotate_every=max(every, 1))
        rs = run(cfg, epochs)
        rows.append({"assortment": label, "spearman": agg(rs, "spearman"),
                     "attr lift": agg(rs, "attr_lift"),
                     "alpha hat": agg(rs, "alpha_hat")})
        print(f"  {label:<32} kernel recovery {rows[-1]['spearman']:+.3f}", flush=True)
    report("E3 (identification of the kernel against assortment availability)",
           rows, ["assortment", "spearman", "attr lift", "alpha hat"])
    res["E3"] = rows

    res["config"] = asdict(base)
    (OUT / ("nesting_quick.json" if a.quick else "nesting.json")).write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
