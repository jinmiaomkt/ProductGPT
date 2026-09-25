"""
R30 stage 3: take the top 3 configurations per family by CAMPAIGN-27 VALIDATION
across stages 1 and 2, and re-run each with three FRESH seeds (11, 12, 13).

Fresh seeds are the point. Every configuration ranked so far was trained once,
and single-seed leads in this project have evaporated three times (b9_tf_sat2,
hyb_stack, hyb_slots2). Stage 3 is what turns a search result into a finding;
the winner's curse is largest exactly at the top of a 144-configuration search.

Stage-1 tags carry default training hyperparameters; stage-2 tags carry the
sampled ones, recovered by re-running the (deterministic) sampler. The holdout
is still not opened -- that is stage 4.

    python3 scripts/r30_stage3_select.py            # print the qsub strings
    python3 scripts/r30_stage3_select.py --top 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

COMMON = "MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8"
STOCK = "INVENTORY=slots,SAT_LAYERS=2"
FAM_FLAGS = {
    "gru": "ARCH=gru",
    "gru_enc": f"ENCODER=gru,{STOCK}",
    "tf": f"ALIBI=1,{STOCK}",
    "hyb": f"ENCODER=gru_attn,FUSE=stack,ALIBI=1,{STOCK}",
}
FAMILIES = ["gru", "gru_enc", "tf", "hyb"]
SEEDS = [11, 12, 13]


def family_of(tag: str) -> str | None:
    body = re.sub(r"^b1[45]_", "", tag)
    for fam in sorted(FAMILIES, key=len, reverse=True):
        if body.startswith(fam + "_"):
            return fam
    return None


def finished_runs(runs_root: Path, prefix: str) -> dict:
    out = {}
    for hist in runs_root.glob(f"*{prefix}*/**/history.json"):
        tag = re.sub(r".*?(?=" + prefix + ")", "", hist.parent.parent.parent.name)
        if not (hist.parent / "final.json").exists():
            continue
        try:
            rows = json.loads(hist.read_text())
        except json.JSONDecodeError:
            continue
        rows = rows if isinstance(rows, list) else rows.get("epochs", [])
        vals = [r["nll"] for r in rows if isinstance(r, dict) and r.get("nll") is not None]
        if vals:
            out[tag] = min(vals)
    return out


def stage2_vars(work: Path) -> dict:
    """TAG -> qsub variable string, from the deterministic stage-2 sampler."""
    res = subprocess.run([sys.executable, str(work / "scripts" / "r30_stage2_sample.py")],
                         capture_output=True, text=True, cwd=work)
    vars_by_tag = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if "TAG=" in line:
            vars_by_tag[line.split("TAG=")[-1]] = line
    return vars_by_tag


def stage1_vars(tag: str) -> str:
    m = re.fullmatch(r"b14_(.+)_d(\d+)_N(\d+)", tag)
    fam, d, n = m.group(1), int(m.group(2)), int(m.group(3))
    return (f"{COMMON},{FAM_FLAGS[fam]},D_MODEL={d},D_FF={3 * d},N_LAYERS={n},"
            f"SEED=1,TAG={tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--work", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    runs = Path(a.runs_root)
    res = {**finished_runs(runs, "b14_"), **finished_runs(runs, "b15_")}
    if len(res) < 144:
        print(f"warning: only {len(res)} finished runs from stages 1-2 (expected 144)",
              file=sys.stderr)
    s2 = stage2_vars(Path(a.work))

    by_fam = defaultdict(list)
    for tag, nll in res.items():
        fam = family_of(tag)
        if fam:
            by_fam[fam].append((nll, tag))

    for fam in FAMILIES:
        picks = sorted(by_fam[fam])[:a.top]
        print(f"# {fam}: " + ", ".join(f"{t} {v:.4f}" for v, t in picks), file=sys.stderr)
        for _, tag in picks:
            base = s2.get(tag) if tag.startswith("b15_") else stage1_vars(tag)
            if base is None:
                print(f"# MISSING variables for {tag} -- skipped", file=sys.stderr)
                continue
            for seed in SEEDS:
                v = re.sub(r"SEED=\d+", f"SEED={seed}", base)
                v = re.sub(r"TAG=[^,]+", f"TAG=b16_{tag}_s{seed}", v)
                print(v)


if __name__ == "__main__":
    main()
