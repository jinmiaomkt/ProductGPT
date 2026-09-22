"""
R30 stage 2: read stage-1 (batch 14) validation results, pick each family's
best and runner-up (width, depth), and sample the pre-registered random search
around them. Prints one `qsub -v` variable string per run; submit_batch15.sh
submits them.

Selection uses CAMPAIGN-27 VALIDATION NLL only (history.json); final.json is
opened only to confirm a run finished. The holdout is never read.

Refuses to emit anything unless all 48 stage-1 runs have finished, so it can be
called safely from a watcher.

Sampling (pre-registered in EXPERIMENTS.md, R30 stage 2):
    (d, N)       12 configs at the family's stage-1 best, 12 at its runner-up
    lr           log-uniform [1e-4, 1.2e-3]
    dropout      {0.05, 0.1, 0.2, 0.3}
    weight decay {0, 0.01, 0.05, 0.1}
    grad accum   {2, 4, 8}  (batch 4 -> effective batch 8, 16, 32)
    d_ff ratio   {2, 3, 4}
    warmup frac  {0.02, 0.05, 0.1}
The RNG is seeded per family, so the draw is reproducible.

USAGE
    python3 scripts/r30_stage2_sample.py            # print the plan + qsub strings
    python3 scripts/r30_stage2_sample.py --check    # exit 0 only if stage 1 is complete
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

FAMILIES = {
    "gru": "ARCH=gru",
    "gru_enc": "ENCODER=gru,INVENTORY=slots,SAT_LAYERS=2",
    "tf": "ALIBI=1,INVENTORY=slots,SAT_LAYERS=2",
    "hyb": "ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2",
}
COMMON = "MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,SEED=1"
STAGE1_PREFIX, STAGE1_RUNS = "b14_", 48
PER_FAMILY = 24
FAMILY_SEED = {"gru": 3001, "gru_enc": 3002, "tf": 3003, "hyb": 3004}


def stage1(runs_root: Path):
    """tag -> (best validation NLL, finished?)"""
    out = {}
    for hist in runs_root.glob(f"*{STAGE1_PREFIX}*/**/history.json"):
        tag = re.sub(r".*?" + STAGE1_PREFIX, "", hist.parent.parent.parent.name)
        try:
            rows = json.loads(hist.read_text())
        except json.JSONDecodeError:
            continue
        rows = rows if isinstance(rows, list) else rows.get("epochs", [])
        vals = [r["nll"] for r in rows if isinstance(r, dict) and r.get("nll") is not None]
        if vals:
            out[tag] = (min(vals), (hist.parent / "final.json").exists())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    res = stage1(Path(a.runs_root))
    done = {t: v for t, (v, fin) in res.items() if fin}
    if len(done) < STAGE1_RUNS:
        print(f"stage 1 incomplete: {len(done)}/{STAGE1_RUNS} finished", file=sys.stderr)
        sys.exit(1)
    if a.check:
        return

    for fam, flags in FAMILIES.items():
        cells = []
        for tag, v in done.items():
            m = re.fullmatch(rf"{fam}_d(\d+)_N(\d+)", tag)
            if m:
                cells.append((v, int(m.group(1)), int(m.group(2))))
        cells.sort()
        (v1, d1, n1), (v2, d2, n2) = cells[0], cells[1]
        print(f"# {fam}: best d{d1} N{n1} ({v1:.4f}), runner-up d{d2} N{n2} ({v2:.4f})", file=sys.stderr)
        rng = random.Random(FAMILY_SEED[fam])
        for k in range(PER_FAMILY):
            d, n = (d1, n1) if k % 2 == 0 else (d2, n2)
            lr = math.exp(rng.uniform(math.log(1e-4), math.log(1.2e-3)))
            drop = rng.choice([0.05, 0.1, 0.2, 0.3])
            wd = rng.choice([0, 0.01, 0.05, 0.1])
            acc = rng.choice([2, 4, 8])
            ffr = rng.choice([2, 3, 4])
            warm = rng.choice([0.02, 0.05, 0.1])
            tag = f"b15_{fam}_c{k:02d}"
            print(f"{COMMON},{flags},D_MODEL={d},D_FF={ffr * d},N_LAYERS={n},"
                  f"LR={lr:.2e},DROPOUT={drop},WD={wd},GRAD_ACCUM={acc},WARMUP={warm},TAG={tag}")


if __name__ == "__main__":
    main()
