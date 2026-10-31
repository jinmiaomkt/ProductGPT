"""R47 part B3 -- the campaign-boundary factor delta and the pre-registered rule.

Opens the HOLDOUT. Run it once, after all 12 B3 runs have finished
(EXPERIMENTS.md R47, B3 approved 31 Oct 2026):

    python3 scripts/r47_b3_read.py

A run is ~/ProductGPT/runs/*_r47b3_hl100_d<DELTA>_s<SEED>/**/final.json.

Rule (pre-registered): the best delta < 1 is chosen on campaign-27 validation
(mean over seeds). A campaign-boundary effect is detected when
d = NLL(delta 1) - NLL(best delta < 1), paired by seed, has mean > 2 x sd(d)
on validation AND on holdout, with the same sign on all 3 seeds. Otherwise
"not identified from purchase likelihood", with the full profile reported.
Also reported, with no decision: delta 1 against B2's half-life-100 runs,
which have the same configuration and seeds (GPU re-run noise).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Dict

from r47_b12_read import CELLS, SEEDS, hold, rule, show, val

HL = "100"
DELTAS = ["1", "0.5", "0.1", "0"]


def find(root: str, tag: str, s: int) -> dict:
    fin = glob.glob(os.path.join(root, f"*_{tag}_s{s}", "**", "final.json"), recursive=True)
    if len(fin) != 1:
        raise SystemExit(f"{tag} s{s}: expected one final.json, found {len(fin)}")
    return json.load(open(fin[0]))


def check(j: dict, label: str, want: dict) -> None:
    c = j["cfg"]
    bad = {k: c.get(k) for k, v in want.items() if c.get(k) != v}
    if bad:
        raise SystemExit(f"{label}: cfg {bad}, expected {want}")
    if j.get("stopped_for_time_budget"):
        raise SystemExit(f"{label}: stopped for the time budget")


def load(root: str) -> Dict[str, Dict[int, dict]]:
    runs: Dict[str, Dict[int, dict]] = {}
    for d in DELTAS:
        runs[d] = {}
        for s in SEEDS:
            j = find(root, f"r47b3_hl{HL}_d{d}", s)
            check(j, f"b3 d{d} s{s}", {"seed": s, "slot_state": "stock_only", "decay": "exp",
                                       "decay_init": float(HL), "decay_freeze": True,
                                       "camp_decay": float(d), "window": 384})
            runs[d][s] = j
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    a = ap.parse_args()
    runs = load(a.runs_root)

    print(f"===== B3 (stock_only, half-life {HL} frozen) -- HOLDOUT OPENED")
    print(f"{'delta':>5} {'val mean':>9} {'sd':>7} {'hold mean':>10} {'sd':>7}   "
          f"{'val per seed':<26} {'hold per seed':<26} best ep")
    vm = {}
    for d in DELTAS:
        v = [val(runs[d][s]) for s in SEEDS]
        h = [hold(runs[d][s]) for s in SEEDS]
        vm[d] = st.mean(v)
        eps = [runs[d][s]["best_epoch"] for s in SEEDS]
        print(f"{d:>5} {st.mean(v):>9.4f} {st.stdev(v):>7.4f} {st.mean(h):>10.4f} "
              f"{st.stdev(h):>7.4f}   {', '.join(f'{x:.4f}' for x in v):<26} "
              f"{', '.join(f'{x:.4f}' for x in h):<26} {eps}")
    best = min(DELTAS[1:], key=lambda k: vm[k])
    print(f"best delta < 1 on validation: {best}; "
          f"best over the grid, 1 included: {min(DELTAS, key=lambda k: vm[k])}")
    rv = rule([val(runs["1"][s]) - val(runs[best][s]) for s in SEEDS])
    rh = rule([hold(runs["1"][s]) - hold(runs[best][s]) for s in SEEDS])
    print(f"d = NLL(delta 1) - NLL(delta {best}), paired by seed "
          f"(positive = a campaign reset helps):")
    show("val", rv)
    show("holdout", rh)
    detected = all(r["same_sign"] and r["beyond"] and r["mean"] > 0 for r in (rv, rh))
    reverse = all(r["same_sign"] and r["beyond"] and r["mean"] < 0 for r in (rv, rh))
    print("  VERDICT: " + ("CAMPAIGN-BOUNDARY EFFECT DETECTED" if detected else
                           "not identified from purchase likelihood"
                           + ("; no reset is better by the same rule" if reverse else "")))
    print("  other cells (holdout, no decision):")
    for cell in CELLS[1:]:
        dd = [hold(runs["1"][s], cell) - hold(runs[best][s], cell) for s in SEEDS]
        print(f"    {cell:<31} " + "  ".join(f"{x:+.4f}" for x in dd) + f"   mean {st.mean(dd):+.4f}")

    print("\nReported with no decision: delta 1 (B3) against B2's half-life-100 runs, "
          "same configuration and seeds")
    for s in SEEDS:
        b2 = find(a.runs_root, f"r47b2_hl{HL}", s)
        check(b2, f"b2 hl{HL} s{s}", {"seed": s, "slot_state": "stock_only",
                                     "decay_init": float(HL), "camp_decay": 1.0})
        r3 = runs["1"][s]
        print(f"  s{s}: val {val(r3):.4f} vs {val(b2):.4f} ({val(r3) - val(b2):+.4f}); "
              f"holdout {hold(r3):.4f} vs {hold(b2):.4f} ({hold(r3) - hold(b2):+.4f}); "
              f"best epoch {r3['best_epoch']} vs {b2['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
