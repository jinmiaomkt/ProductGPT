"""
Rank tuning runs by VALIDATION only (R30 search stages).

`summarize_batch.py` reports the four test cells. That is right for a finished
comparison and wrong for a hyperparameter search: choosing a configuration on
the holdout is how a model comparison quietly becomes a test-set fit. This
script reads `history.json` and reports the best campaign-27 validation NLL and
the epoch it came from. **It never opens final.json and never reports a test
cell.** Open the holdout once, with `summarize_batch.py`, after the
configuration is frozen.

USAGE
    python3 scripts/summarize_search.py --prefix b15_          # rank a stage
    python3 scripts/summarize_search.py --prefix b15_ --group 2  # pool seeds,
        # grouping tags as <config>_s<seed> and averaging over seeds
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--prefix", required=True, help="tag prefix, e.g. b15_")
    ap.add_argument("--group", action="store_true",
                    help="pool <config>_s<seed> tags and report mean +/- sd over seeds")
    ap.add_argument("--top", type=int, default=0, help="show only the best N")
    a = ap.parse_args()

    runs = defaultdict(list)
    for hist in Path(a.runs_root).glob(f"*{a.prefix}*/**/history.json"):
        tag = re.sub(r".*?" + re.escape(a.prefix), a.prefix, hist.parent.parent.parent.name)
        try:
            rows = json.loads(hist.read_text())
        except json.JSONDecodeError:
            continue
        rows = rows if isinstance(rows, list) else rows.get("epochs", [])
        vals = [(r.get("nll"), r.get("epoch")) for r in rows
                if isinstance(r, dict) and r.get("nll") is not None]
        if not vals:
            continue
        best_nll, best_ep = min(vals, key=lambda t: t[0])
        if not (hist.parent / "final.json").exists():
            tag += "  (RUNNING)"   # still training: partial curve, not comparable yet
        key = re.sub(r"_s\d+$", "", tag) if a.group else tag
        runs[key].append((best_nll, best_ep, len(rows)))

    if not runs:
        print(f"no runs found under {a.runs_root} matching {a.prefix!r}")
        return

    print("Ranked by BEST CAMPAIGN-27 VALIDATION NLL. Holdout cells are not read here.")
    print(f"{'configuration':<42} {'n':>2} {'val NLL':>16} {'best ep':>8} {'epochs':>7}")
    rows = []
    for key, v in runs.items():
        nlls = [x[0] for x in v]
        mean = st.mean(nlls)
        sd = st.stdev(nlls) if len(nlls) > 1 else None
        rows.append((mean, key, len(v), sd, st.mean(x[1] for x in v), st.mean(x[2] for x in v)))
    rows.sort()
    for i, (mean, key, n, sd, ep, eps) in enumerate(rows):
        if a.top and i >= a.top:
            break
        shown = f"{mean:.4f}" + (f" +/-{sd:.4f}" if sd is not None else "")
        print(f"{key:<42} {n:>2} {shown:>16} {ep:>8.1f} {eps:>7.1f}")
    print(f"\n{len(rows)} configurations; best {rows[0][1]} at {rows[0][0]:.4f} validation NLL.")
    print("Confirm the top configurations with fresh seeds before opening the holdout.")


if __name__ == "__main__":
    main()
