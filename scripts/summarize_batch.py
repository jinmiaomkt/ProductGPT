"""
Aggregate a seeded batch of runs into mean +/- sd per configuration.

Written for batch 2 (scripts/submit_batch2.sh), whose run tags look like
    b2_<config>_s<seed>        e.g. b2_tf_noemb_s1, b2_gru_s3
Runs sharing <config> are pooled across seeds. All batch-2 runs share one
customer partition (split_seed=33), so the spread across seeds is training
noise, and configurations can be compared seed-by-seed as well as on average.

Every number here is at the checkpoint VALIDATION selected -- the honest,
reportable figure. Holdout-tracking fields in history.json are ignored; use
scripts/drift_diagnostic.py for those.

WHAT TO LOOK AT
  * the out-of-sample x holdout cell is the headline: unseen customers in an
    unseen period, and identical in composition to earlier designs;
  * the gap between two configurations should be judged against the seed sd,
    not in isolation -- a 0.01-nat difference with sd 0.02 is nothing;
  * "paired" wins count how many seeds config A beat B on the same seed,
    which is a fairer small-n comparison than two overlapping means.

USAGE
    python3 scripts/summarize_batch.py                       # prefix b2_
    python3 scripts/summarize_batch.py --prefix b2_ --cell insample_users_holdout_period
    python3 scripts/summarize_batch.py --runs-root /path/to/runs

Pure standard library so it runs under the cluster's system python3.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from typing import Dict, List

CELLS = ["outsample_users_holdout_period", "insample_users_holdout_period",
         "outsample_users_calib_period"]
METRICS = [("nll", True), ("f1_macro", False), ("auprc_macro", False),
           ("rev_mae", True), ("hit", False)]


def ms(xs: List[float]) -> str:
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{st.mean(xs):.4f}+/-{st.stdev(xs):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--prefix", default="b2_")
    ap.add_argument("--cell", default=CELLS[0], help="cell used for rankings and pairs")
    a = ap.parse_args()

    pat = re.compile(r".*_" + re.escape(a.prefix) + r"(?P<cfg>.+)_s(?P<seed>\d+)$")
    runs: Dict[str, Dict[int, dict]] = defaultdict(dict)
    missing: List[str] = []
    for d in sorted(glob.glob(os.path.join(a.runs_root, f"*{a.prefix}*"))):
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        fin = glob.glob(os.path.join(d, "**", "final.json"), recursive=True)
        if not fin:
            missing.append(os.path.basename(d))
            continue
        j = json.load(open(fin[0]))
        runs[m.group("cfg")][int(m.group("seed"))] = j

    if not runs:
        print(f"No runs matching *{a.prefix}<config>_s<seed> under {a.runs_root}")
        return 1
    if missing:
        print(f"NOT FINISHED / NO OUTPUT ({len(missing)}): {', '.join(missing)}\n")

    cfgs = sorted(runs)
    # ---- per-configuration table, one block per cell ----------------------
    for cell in CELLS:
        print("=" * 100)
        print(f"{cell}   (mean +/- sd over seeds, at the VALIDATION-selected epoch)")
        print("=" * 100)
        print(f"{'config':<18}{'n':>3}{'sel ep':>9}{'params':>11}  "
              + "".join(f"{m:>18}" for m, _ in METRICS))
        for c in cfgs:
            seeds = runs[c]
            eps = [j.get("best_epoch") for j in seeds.values() if j.get("best_epoch") is not None]
            par = next(iter(seeds.values())).get("params", 0)
            row = []
            for m, _ in METRICS:
                xs = [j["test_cells"][cell][m] for j in seeds.values()
                      if cell in (j.get("test_cells") or {})]
                row.append(ms(xs))
            ep_s = f"{st.mean(eps):.1f}" if eps else "--"
            print(f"{c:<18}{len(seeds):>3}{ep_s:>9}{par:>11,}  "
                  + "".join(f"{x:>18}" for x in row))
        print()

    # ---- ranking + paired comparison on the chosen cell -------------------
    def nll(c, s):
        return runs[c][s]["test_cells"][a.cell]["nll"]

    ranked = sorted(cfgs, key=lambda c: st.mean(nll(c, s) for s in runs[c]))
    print("=" * 100)
    print(f"RANKING by mean NLL on {a.cell}")
    print("=" * 100)
    best = ranked[0]
    for i, c in enumerate(ranked, 1):
        xs = [nll(c, s) for s in runs[c]]
        delta = st.mean(xs) - st.mean(nll(best, s) for s in runs[best])
        shared = sorted(set(runs[c]) & set(runs[best]))
        wins = sum(nll(best, s) < nll(c, s) for s in shared)
        pair = (f"   best wins {wins}/{len(shared)} paired seeds"
                if c != best and shared else "")
        print(f"  {i}. {c:<18} {ms(xs):>18}   d vs best {delta:+.4f}{pair}")
    sds = [st.stdev([nll(c, s) for s in runs[c]]) for c in cfgs if len(runs[c]) > 1]
    if sds:
        print(f"\n  typical seed sd: {st.median(sds):.4f} nats. A gap much smaller than "
              "this is not a difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
