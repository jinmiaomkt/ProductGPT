"""R44 stage 4 -- the final, paired comparison of the two frozen families.

Opens the HOLDOUT. Run it once, after every stage-4 run has finished
(EXPERIMENTS.md R44). The runs come from three batches, so each family is
given as explicit TAG:SEEDS pieces instead of one tag prefix:

    python3 scripts/r44_stage4.py \
        --g5 b40_g5_c06:1 --g5 b42_g5_c06:2,3,4,5 \
        --g6 b39_g6w384:1,2,3,4,5

A run is ~/ProductGPT/runs/*_<TAG>_s<SEED>/**/final.json. The decision rule
(pre-registered) is applied to G6 - G5 on the headline cell, paired by seed:
better if the mean < -0.02 and at least 4 of 5 seeds are negative; worse if
the mean > +0.02 and at least 4 of 5 are positive; otherwise a tie.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Dict, List

HEADLINE = "outsample_users_holdout_period"
CELLS = [HEADLINE, "insample_users_holdout_period", "outsample_users_calib_period"]
MARGIN, MIN_SEEDS = 0.02, 4


def load(root: str, pieces: List[str]) -> Dict[int, dict]:
    runs: Dict[int, dict] = {}
    for piece in pieces:
        tag, seeds = piece.split(":")
        for s in (int(x) for x in seeds.split(",")):
            fin = glob.glob(os.path.join(root, f"*_{tag}_s{s}", "**", "final.json"),
                            recursive=True)
            if len(fin) != 1:
                raise SystemExit(f"{tag} seed {s}: expected one final.json, found {len(fin)}")
            if s in runs:
                raise SystemExit(f"seed {s} given twice")
            j = json.load(open(fin[0]))
            hist = os.path.join(os.path.dirname(fin[0]), "history.json")
            j["_history"] = json.load(open(hist)) if os.path.exists(hist) else None
            j["_tag"] = tag
            runs[s] = j
    return runs


def verdict(d: List[float]) -> str:
    m = st.mean(d)
    if m < -MARGIN and sum(x < 0 for x in d) >= MIN_SEEDS:
        return "GEN 6 BETTER"
    if m > MARGIN and sum(x > 0 for x in d) >= MIN_SEEDS:
        return "GEN 6 WORSE"
    return "TIE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--g5", action="append", required=True, help="TAG:SEED,SEED,...")
    ap.add_argument("--g6", action="append", required=True, help="TAG:SEED,SEED,...")
    ap.add_argument("--sensitivity-drop", default="1",
                    help="seeds left out in the pre-stated sensitivity reading")
    a = ap.parse_args()

    fam = {"G5": load(a.runs_root, a.g5), "G6": load(a.runs_root, a.g6)}
    seeds = sorted(fam["G5"])
    if seeds != sorted(fam["G6"]):
        raise SystemExit(f"seeds differ: G5 {seeds}, G6 {sorted(fam['G6'])}")
    for name, runs in fam.items():
        revs = {r["cfg"].get("model_rev") for r in runs.values()}
        if revs != {2}:
            raise SystemExit(f"{name}: model_rev {revs}, expected 2")

    def nll(name, s, cell=HEADLINE):
        return fam[name][s]["test_cells"][cell]["nll"]

    print(f"R44 stage 4 -- seeds {seeds}. HOLDOUT OPENED.\n")
    for cell in CELLS:
        print(f"{cell}")
        for name in fam:
            xs = [nll(name, s, cell) for s in seeds]
            print(f"  {name}  " + "  ".join(f"s{s} {x:.4f}" for s, x in zip(seeds, xs))
                  + f"   mean {st.mean(xs):.4f}  sd {st.stdev(xs):.4f}")
        d = [nll("G6", s, cell) - nll("G5", s, cell) for s in seeds]
        print(f"  G6-G5 " + "  ".join(f"{x:+.4f}" for x in d)
              + f"   mean {st.mean(d):+.4f}  negative {sum(x < 0 for x in d)}/{len(d)}\n")

    d = [nll("G6", s) - nll("G5", s) for s in seeds]
    print(f"DECISION (headline, margin {MARGIN}, {MIN_SEEDS} of {len(d)} seeds): {verdict(d)}")
    drop = {int(x) for x in a.sensitivity_drop.split(",") if x}
    keep = [s for s in seeds if s not in drop]
    dk = [nll("G6", s) - nll("G5", s) for s in keep]
    print(f"Sensitivity, seeds {keep} only (not in the selection pool): "
          f"mean {st.mean(dk):+.4f}, negative {sum(x < 0 for x in dk)}/{len(dk)}\n")

    print("Reported whatever the NLL says:")
    for name, runs in fam.items():
        r = runs[seeds[0]]
        c = r["cfg"]
        eps = [x["best_epoch"] for x in runs.values()]
        secs, peaks = [], []
        for x in runs.values():
            for h in x["_history"] or []:
                if isinstance(h, dict):
                    if h.get("secs") is not None:
                        secs.append(h["secs"])
                    if h.get("peak_mem") is not None:
                        peaks.append(h["peak_mem"])
        print(f"  {name}: tags {sorted({x['_tag'] for x in runs.values()})}, "
              f"params {r['params']:,}, window {c.get('window', c.get('max_events'))}, "
              f"best epoch {st.mean(eps):.1f}"
              + (f", {st.mean(secs) / 60:.2f} min/epoch" if secs else "")
              + (f", peak {max(peaks) / 2**30:.1f} GB" if peaks else ", peak not logged"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
