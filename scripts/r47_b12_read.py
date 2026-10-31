"""R47 parts B1 and B2 -- the half-life profiles and the pre-registered rule.

Opens the HOLDOUT. Run it once, after all 42 B1/B2 runs have finished
(EXPERIMENTS.md R47):

    python3 scripts/r47_b12_read.py

A run is ~/ProductGPT/runs/*_r47b<part>_hl<HL>_s<SEED>/**/final.json.

Rule (pre-registered, per part): the best finite half-life is chosen on
campaign-27 validation (mean over seeds). Forgetting is detected when
d = NLL(inf) - NLL(best finite), paired by seed, has mean > 2 x sd(d) on
validation AND on holdout, with the same sign on all 3 seeds. Otherwise
"not identified from purchase likelihood", with the full profile reported.
B3 then runs at B2's best half-life on validation over the whole grid, inf
included.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Dict

HEADLINE = "outsample_users_holdout_period"
CELLS = [HEADLINE, "insample_users_holdout_period", "outsample_users_calib_period"]
GRID = ["1", "2", "5", "10", "30", "100", "inf"]
SEEDS = [1, 2, 3]
SLOT = {"b1": "full", "b2": "stock_only"}


def load(root: str, part: str) -> Dict[str, Dict[int, dict]]:
    runs: Dict[str, Dict[int, dict]] = {}
    for hl in GRID:
        runs[hl] = {}
        for s in SEEDS:
            fin = glob.glob(os.path.join(root, f"*_r47{part}_hl{hl}_s{s}", "**", "final.json"),
                            recursive=True)
            if len(fin) != 1:
                raise SystemExit(f"{part} hl{hl} s{s}: expected one final.json, found {len(fin)}")
            j = json.load(open(fin[0]))
            c = j["cfg"]
            want = {"seed": s, "slot_state": SLOT[part], "camp_decay": 1.0, "window": 384,
                    "decay": "none" if hl == "inf" else "exp"}
            if hl != "inf":
                want.update(decay_init=float(hl), decay_freeze=True)
            bad = {k: c.get(k) for k, v in want.items() if c.get(k) != v}
            if bad:
                raise SystemExit(f"{part} hl{hl} s{s}: cfg {bad}, expected {want}")
            if j.get("stopped_for_time_budget"):
                raise SystemExit(f"{part} hl{hl} s{s}: stopped for the time budget")
            runs[hl][s] = j
    return runs


def val(r: dict) -> float:
    return r["best_val_nll"]


def hold(r: dict, cell: str = HEADLINE) -> float:
    return r["test_cells"][cell]["nll"]


def rule(d) -> dict:
    m, sd = st.mean(d), st.stdev(d)
    same = all(x > 0 for x in d) or all(x < 0 for x in d)
    return {"d": d, "mean": m, "sd": sd, "same_sign": same,
            "beyond": abs(m) > 2 * sd}


def show(label: str, r: dict) -> None:
    print(f"  {label:<9}" + "  ".join(f"{x:+.4f}" for x in r["d"])
          + f"   mean {r['mean']:+.4f}  sd {r['sd']:.4f}  2sd {2 * r['sd']:.4f}"
          + f"   same sign {'yes' if r['same_sign'] else 'no'}")


def part_read(part: str, runs: Dict[str, Dict[int, dict]]) -> dict:
    print(f"===== {part.upper()} (slot_state {SLOT[part]}) -- HOLDOUT OPENED")
    print(f"{'HL':>5} {'val mean':>9} {'sd':>7} {'hold mean':>10} {'sd':>7}   "
          f"{'val per seed':<26} {'hold per seed':<26} best ep")
    vm = {}
    for hl in GRID:
        v = [val(runs[hl][s]) for s in SEEDS]
        h = [hold(runs[hl][s]) for s in SEEDS]
        vm[hl] = st.mean(v)
        eps = [runs[hl][s]["best_epoch"] for s in SEEDS]
        print(f"{hl:>5} {st.mean(v):>9.4f} {st.stdev(v):>7.4f} {st.mean(h):>10.4f} "
              f"{st.stdev(h):>7.4f}   {', '.join(f'{x:.4f}' for x in v):<26} "
              f"{', '.join(f'{x:.4f}' for x in h):<26} {eps}")
    best_fin = min(GRID[:-1], key=lambda k: vm[k])
    best_all = min(GRID, key=lambda k: vm[k])
    print(f"best finite half-life on validation: {best_fin}; "
          f"best over the grid, inf included: {best_all}")
    rv = rule([val(runs["inf"][s]) - val(runs[best_fin][s]) for s in SEEDS])
    rh = rule([hold(runs["inf"][s]) - hold(runs[best_fin][s]) for s in SEEDS])
    print(f"d = NLL(inf) - NLL(hl {best_fin}), paired by seed (positive = forgetting helps):")
    show("val", rv)
    show("holdout", rh)
    detected = all(r["same_sign"] and r["beyond"] and r["mean"] > 0 for r in (rv, rh))
    reverse = all(r["same_sign"] and r["beyond"] and r["mean"] < 0 for r in (rv, rh))
    print("  VERDICT: " + ("FORGETTING DETECTED" if detected else
                           "not identified from purchase likelihood"
                           + ("; no decay is better by the same rule" if reverse else "")))
    print("  other cells (holdout, no decision):")
    for cell in CELLS[1:]:
        d = [hold(runs["inf"][s], cell) - hold(runs[best_fin][s], cell) for s in SEEDS]
        print(f"    {cell:<31} " + "  ".join(f"{x:+.4f}" for x in d) + f"   mean {st.mean(d):+.4f}")
    print()
    return {"best_finite": best_fin, "best_all": best_all, "detected": detected,
            "val": rv, "holdout": rh}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    a = ap.parse_args()
    out = {p: part_read(p, load(a.runs_root, p)) for p in ("b1", "b2")}
    print("Prediction 2 (B1 flat, inf included) and 3 (B2 best finite beats inf, best > 5):")
    for p, r in out.items():
        print(f"  {p}: best finite {r['best_finite']}, best overall {r['best_all']}, "
              f"forgetting detected: {r['detected']}")
    print(f"B3 runs at B2's best half-life on validation: {out['b2']['best_all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
