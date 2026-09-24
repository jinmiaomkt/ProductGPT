"""
R30 stage 4: the final comparison. THIS IS THE RUN THAT OPENS THE HOLDOUT.

Each family's configuration is frozen by its stage-3 validation MEAN over three
fresh seeds -- no further selection happens here. Every frozen configuration is
then run at five seeds (1-5) at BOTH vocabulary levels, plus a stock-path
ablation on the family that leads on validation.

    4 families x 5 seeds x 2 vocabulary levels      = 40
    winner x 3 stock variants x 5 seeds (level 7)   = 15
                                                      55 runs

Decision rules (pre-registered, EXPERIMENTS.md R30): family A beats B only if
the five-seed holdout mean differs by > 0.02 nats AND A wins at least 4 of 5
paired seeds. The vocabulary interaction counts only if it exceeds 0.02 with a
consistent sign in at least two families.

    python3 scripts/r30_stage4_select.py
"""
from __future__ import annotations

import argparse
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_stage3_select import (FAMILIES, family_of, finished_runs,  # noqa: E402
                               stage1_vars, stage2_vars)

SEEDS = [1, 2, 3, 4, 5]
# stock-path ablations for the leading family, at level 7
ABLATIONS = {
    "counts2": "INVENTORY=slots,SAT_LAYERS=2",     # the frozen setting (control)
    "counts0": "INVENTORY=slots,SAT_LAYERS=0",     # single attention step over slots
    "tokens": "",                                   # inventory GRU + cross-attention
    "nostock": "CROSS_ATTN=0",                      # no stock path at all
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--work", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()

    # stage 3: pool seeds, rank each family by validation MEAN
    res = finished_runs(Path(a.runs_root), "b16_")
    pooled = defaultdict(list)
    for tag, nll in res.items():
        pooled[re.sub(r"_s\d+$", "", tag)].append(nll)
    means = {k: st.mean(v) for k, v in pooled.items() if len(v) >= 3}
    if not means:
        sys.exit("no completed stage-3 configurations with 3 seeds")

    frozen = {}
    for fam in FAMILIES:
        cands = [(m, k) for k, m in means.items() if family_of(re.sub(r"^b16_", "", k)) == fam]
        if cands:
            frozen[fam] = min(cands)
    winner = min((m, fam) for fam, (m, _) in frozen.items())[1]

    s2 = stage2_vars(Path(a.work))
    for fam, (mean, b16tag) in frozen.items():
        orig = re.sub(r"^b16_", "", b16tag)
        base = s2.get(orig) if orig.startswith("b15_") else stage1_vars(orig)
        print(f"# {fam}: frozen at {orig} (stage-3 validation mean {mean:.4f})"
              + ("   <-- leads, gets the stock ablation" if fam == winner else ""),
              file=sys.stderr)
        for level in (7, 6):
            for seed in SEEDS:
                v = re.sub(r"SEED=\d+", f"SEED={seed}", base)
                v = re.sub(r"VOCAB_LEVEL=\d+", f"VOCAB_LEVEL={level}", v)
                v = re.sub(r"TAG=\S+", f"TAG=b17_{fam}_v{level}_s{seed}", v)
                print(v)
        if fam == winner:
            for name, flags in ABLATIONS.items():
                if name == "counts2":
                    continue                       # already covered by the v7 arm above
                for seed in SEEDS:
                    v = re.sub(r"SEED=\d+", f"SEED={seed}", base)
                    v = re.sub(r"INVENTORY=slots,SAT_LAYERS=2", "", v).replace(",,", ",")
                    v = v.rstrip(",")
                    if flags:
                        v = f"{v},{flags}"
                    v = re.sub(r"TAG=\S+", f"TAG=b17_{fam}_stock_{name}_s{seed}", v)
                    print(v)


if __name__ == "__main__":
    main()
