"""
R31 analysis: is the vocabulary x architecture crossover real, and where does it live?

Reads the per-occasion files written by eval_per_occasion.py, averages each
model group over its seeds, and computes the pre-registered difference-in-
differences

    DiD = (A7 - B7) - (A6 - B6)          negative = crossover in A's favour

per occasion, where A is the attention model and B the recurrent comparison,
6/7 the vocabulary level. Inference is a CLUSTER BOOTSTRAP OVER CUSTOMERS:
customers are resampled with replacement and the occasion-weighted mean DiD is
recomputed, so the unit of evidence is ~2,500 customers rather than 3 seeds.

Before any statistic, two integrity checks:
  1. every model scores exactly the same occasions with the same labels;
  2. each file's mean NLL matches the test-cell NLL its training run recorded
     in final.json (proves the evaluator rebuilt the model and split exactly).

Only aggregates are printed.

USAGE
    python3 scripts/r31_analysis.py --res-dir ~/ProductGPT/work/results/r31 \
        --runs-root ~/ProductGPT/runs
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CELL = "outsample_users_holdout_period"

# (name, attention group A, recurrent group B) -- tag stems without _s<seed>.
COMPARISONS = [
    ("PRIMARY  token inventory: transformer vs GRU encoder",
     {"6": "b11_tf_alibi", "7": "b12_tf_alibi"}, {"6": "b11_gru_cross", "7": "b12_gru_cross"}),
    ("vs plain GRU (token inventory transformer)",
     {"6": "b11_tf_alibi", "7": "b12_tf_alibi"}, {"6": "b11_gru", "7": "b12_gru"}),
    ("CONTRAST per-product counts: transformer vs GRU encoder",
     {"6": "b13_v6_tf_slots2", "7": "b13_v7_tf_slots2"}, {"6": "b13_v6_gru_enc_slots2", "7": "b13_v7_gru_enc_slots2"}),
    ("CAPACITY transformer (tokens) vs width-matched GRU",
     {"6": "b11_tf_alibi", "7": "b12_tf_alibi"}, {"6": "b13_v6_gruw", "7": "b13_v7_gruw"}),
]


def load_group(res: Path, runs: Path, stem: str):
    files = sorted(res.glob(f"{stem}_s[0-9].npz"))
    if not files:
        return None, []
    arrays, checks = [], []
    for f in files:
        d = dict(np.load(f))
        arrays.append(d)
        fin = glob.glob(str(runs / f"*_b4_{f.stem}" / "**" / "final.json"), recursive=True)
        if fin:
            rec = json.loads(Path(fin[0]).read_text())["test_cells"][CELL]["nll"]
            checks.append((f.stem, float(d["nll"].mean()), float(rec)))
    return arrays, checks


def order_of(d):
    """Sort occasions by (customer, position) without combining into one integer."""
    return np.lexsort((d["pos"], d["uid_hash"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    res, runs = Path(a.res_dir).expanduser(), Path(a.runs_root).expanduser()
    rng = np.random.default_rng(a.seed)

    for title, A, Bm in COMPARISONS:
        print("=" * 96)
        print(title)
        print("=" * 96)
        groups, bad = {}, False
        for role, spec in (("A", A), ("B", Bm)):
            for lvl, stem in spec.items():
                arrs, checks = load_group(res, runs, stem)
                if not arrs:
                    print(f"  missing: {stem} -- comparison skipped")
                    bad = True
                    break
                for name, mine, rec in checks:
                    flag = "ok" if abs(mine - rec) < 5e-4 else "MISMATCH"
                    print(f"  check {name:<30} evaluator {mine:.4f} vs recorded {rec:.4f}  {flag}")
                    bad |= flag != "ok"
                groups[(role, lvl)] = arrs
            if bad:
                break
        if bad:
            print("  -> not analysed (missing files or evaluator mismatch)\n")
            continue

        ref = groups[("A", "6")][0]
        order = order_of(ref)
        u_ref, p_ref, l_ref = ref["uid_hash"][order], ref["pos"][order], ref["label"][order]
        aligned = {}
        for gk, arrs in groups.items():
            mats = []
            for d in arrs:
                o = order_of(d)
                if not (np.array_equal(d["uid_hash"][o], u_ref) and np.array_equal(d["pos"][o], p_ref)
                        and np.array_equal(d["label"][o], l_ref)):
                    raise SystemExit(f"occasion sets differ for {gk}: cannot pair")
                mats.append(d["nll"][o])
            aligned[gk] = np.mean(mats, axis=0)          # seed-averaged per-occasion NLL
        n_seeds = {gk: len(v) for gk, v in groups.items()}
        cust = ref["uid_hash"][order]
        label = ref["label"][order]
        owned = ref["owned_offer"][order]
        n_ltd = ref["n_ltd"][order]
        did = (aligned[("A", "7")] - aligned[("B", "7")]) - (aligned[("A", "6")] - aligned[("B", "6")])

        ucust, cidx = np.unique(cust, return_inverse=True)
        print(f"  {len(did):,} occasions, {len(ucust):,} customers; seeds per cell {n_seeds}")
        subgroups = [
            ("all occasions", np.ones_like(did, bool)),
            ("purchase decisions (1-8)", label <= 8),
            ("NotBuy (9)", label == 9),
            ("offer includes an OWNED limited 5-star", owned == 1),
            ("limited offer, none owned  [placebo]", owned == 0),
            ("no limited product on offer", owned == -1),
            ("owns <=3 distinct limited 5-stars", n_ltd <= 3),
            ("owns 4-8", (n_ltd >= 4) & (n_ltd <= 8)),
            ("owns >=9", n_ltd >= 9),
        ]
        # per-customer sums for a fast cluster bootstrap
        print(f"  {'subgroup':<42} {'occasions':>10} {'DiD':>9}  {'95% CI':>20}  {'P(DiD<0)':>9}")
        boots = {}
        # ONE set of customer resamples shared by every subgroup, so contrasts
        # between subgroups (owned minus placebo) are computed draw by draw.
        draws = rng.integers(0, len(ucust), size=(a.boot, len(ucust)))
        for name, m in subgroups:
            s = np.bincount(cidx, weights=np.where(m, did, 0.0), minlength=len(ucust))
            c = np.bincount(cidx, weights=m.astype(float), minlength=len(ucust))
            if c.sum() == 0:
                continue
            est = s.sum() / c.sum()
            b = s[draws].sum(1) / np.maximum(c[draws].sum(1), 1)
            boots[name] = b
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"  {name:<42} {int(c.sum()):>10,} {est:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]  {np.mean(b < 0):>9.3f}")
        if "offer includes an OWNED limited 5-star" in boots and "limited offer, none owned  [placebo]" in boots:
            d = boots["offer includes an OWNED limited 5-star"] - boots["limited offer, none owned  [placebo]"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {'owned minus placebo (C, prediction: < 0)':<42} {'':>10} {np.mean(d):>+9.4f}  [{lo:+.4f}, {hi:+.4f}]  {np.mean(d < 0):>9.3f}")
        print()


if __name__ == "__main__":
    main()
