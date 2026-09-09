"""
Which validation metric should we select checkpoints and configurations on?

WHY THIS EXISTS
---------------
Across every comparison so far, validation NLL has ranked models in the
opposite order to holdout performance:

    jmr_flat  beats jmr_mix8 on val NLL, loses on the holdout period
    v2_emb    has the WORST val NLL of three arms and the BEST holdout

Hyperparameter tuning is a search that maximises whatever metric it is given.
If that metric mis-ranks models at n=4, running it 200 times finds the
configuration that games it hardest and returns a confident number pointing
the wrong way. So the selection metric has to be settled BEFORE tuning, not
after.

WHAT THIS MEASURES
------------------
Two things, both from artefacts that already exist -- no GPU time.

  1. WITHIN-RUN DISAGREEMENT. history.json holds per-epoch validation metrics.
     For each run, which epoch would each metric have selected? If NLL picks
     epoch 8 and AUPRC picks epoch 13, the choice of metric is already
     changing which weights we keep.

  2. CROSS-RUN PREDICTIVE POWER. For each finished run we have a validation
     metric at the selected epoch and the holdout metrics that resulted.
     Rank the runs by each validation metric, rank them by holdout
     performance, and see which validation metric orders them correctly.
     This is precisely the question tuning asks: "given these candidate
     configurations, which does my metric tell me to keep?"

WHAT IT CANNOT MEASURE
----------------------
Only best.pt and last.pt are kept, so alternative epochs cannot be re-scored
on the holdout. Settling (1) properly needs a run that checkpoints every
epoch. This script reports the disagreement so we know whether that run is
worth its GPU time.

Runs are pooled ONLY if they carry test_cells, i.e. were evaluated under the
current 2x2 design. Older runs are listed and skipped -- their validation sets
are different and pooling them would be meaningless.

USAGE
    python3 scripts/selection_metric_study.py
    python3 scripts/selection_metric_study.py --runs-root /path/to/runs
    python3 scripts/selection_metric_study.py --cell insample_users_holdout_period

Pure standard library: no numpy, no scipy, so it runs under the cluster's
system python3 without a venv.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Validation metrics, and whether lower is better.
VAL_METRICS = {
    "nll": True,
    "hit": False,
    "f1_macro": False,
    "auprc_macro": False,
    "rev_mae": True,
}
DEFAULT_CELL = "outsample_users_holdout_period"


def spearman(xs: List[float], ys: List[float]) -> float:
    """Rank correlation, average ranks for ties. Returns nan for n < 3."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load_runs(root: str, cell: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    runs, skipped = [], []
    for fin in sorted(glob.glob(os.path.join(root, "*", "**", "final.json"),
                                recursive=True)):
        rundir = fin.split(os.sep + "runs" + os.sep)
        tag = (rundir[1].split(os.sep)[0] if len(rundir) > 1
               else os.path.basename(os.path.dirname(fin)))
        try:
            d = json.load(open(fin))
        except Exception:
            continue
        cells = d.get("test_cells") or {}
        if cell not in cells:
            skipped.append(tag)
            continue
        hist_path = os.path.join(os.path.dirname(fin), "history.json")
        hist = []
        if os.path.exists(hist_path):
            try:
                hist = json.load(open(hist_path))
            except Exception:
                hist = []
        runs.append({
            "tag": tag,
            "cfg": d.get("cfg", {}),
            "best_epoch": d.get("best_epoch"),
            "history": hist,
            "holdout": cells[cell],
        })
    return runs, skipped


def best_epoch_by(hist: List[Dict[str, Any]], metric: str, lower: bool):
    vals = [(h.get(metric), h.get("epoch")) for h in hist if h.get(metric) is not None]
    if not vals:
        return None, None
    v, e = (min(vals) if lower else max(vals))
    return e, v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root",
                    default=os.environ.get("PRODUCTGPT_RUNS",
                                           os.path.expanduser("~/ProductGPT/runs")))
    ap.add_argument("--cell", default=DEFAULT_CELL,
                    help="which holdout cell counts as the target")
    a = ap.parse_args()

    runs, skipped = load_runs(a.runs_root, a.cell)
    if not runs:
        print(f"No runs with cell {a.cell!r} under {a.runs_root}", file=sys.stderr)
        return 1

    print(f"target cell: {a.cell}")
    print(f"runs pooled: {len(runs)}   skipped (no test_cells, pre-redesign): "
          f"{len(skipped)}")
    for s in skipped:
        print(f"    skipped: {s}")

    # ---- 1. within-run disagreement -----------------------------------
    print("\n" + "=" * 78)
    print("1. WHICH EPOCH WOULD EACH VALIDATION METRIC HAVE SELECTED?")
    print("=" * 78)
    hdr = f"{'run':<34}{'used':>6}" + "".join(f"{m[:9]:>10}" for m in VAL_METRICS)
    print(hdr)
    n_disagree = 0
    for r in runs:
        if not r["history"]:
            continue
        picks = []
        for m, lower in VAL_METRICS.items():
            e, _ = best_epoch_by(r["history"], m, lower)
            picks.append(e)
        used = r["best_epoch"]
        if len({p for p in picks if p is not None}) > 1:
            n_disagree += 1
        print(f"{r['tag'][:33]:<34}{used if used is not None else '?':>6}"
              + "".join(f"{(p if p is not None else '-'):>10}" for p in picks))
    print(f"\n  {n_disagree} of {len(runs)} runs: the five metrics do NOT agree on"
          " which epoch to keep.")
    print("  Re-scoring those epochs on the holdout needs per-epoch checkpoints,")
    print("  which are not saved. That is the argument for one diagnostic run.")

    # ---- 2. cross-run predictive power ---------------------------------
    print("\n" + "=" * 78)
    print("2. WHICH VALIDATION METRIC RANKS CONFIGURATIONS CORRECTLY?")
    print("=" * 78)

    def val_at_selected(r, metric):
        for h in r["history"]:
            if h.get("epoch") == r["best_epoch"]:
                return h.get(metric)
        return None

    print(f"\n{'run':<34}{'val nll':>9}{'val f1':>9}{'val auprc':>11}"
          f"{'HO nll':>9}{'HO f1':>9}")
    for r in sorted(runs, key=lambda x: x["holdout"]["nll"]):
        vn = val_at_selected(r, "nll")
        vf = val_at_selected(r, "f1_macro")
        va = val_at_selected(r, "auprc_macro")
        fmt = lambda v, w: (f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(f"{r['tag'][:33]:<34}{fmt(vn,9)}{fmt(vf,9)}{fmt(va,11)}"
              f"{r['holdout']['nll']:>9.4f}{r['holdout']['f1_macro']:>9.4f}")

    print("\n  Spearman rank correlation, validation metric vs holdout outcome.")
    print("  A metric that selects well should be strongly POSITIVE against")
    print("  holdout metrics where higher is better, and NEGATIVE against holdout")
    print("  NLL. A correlation near zero or of the wrong sign means selecting on")
    print("  that metric is no better than guessing.\n")
    ho_keys = ["nll", "f1_macro", "auprc_macro", "hit", "rev_mae"]
    print(f"  {'val metric':<14}" + "".join(f"{'HO ' + k[:8]:>13}" for k in ho_keys))
    for m in VAL_METRICS:
        xs, row = [], []
        for k in ho_keys:
            pairs = [(val_at_selected(r, m), r["holdout"].get(k)) for r in runs]
            pairs = [(x, y) for x, y in pairs
                     if isinstance(x, (int, float)) and isinstance(y, (int, float))]
            row.append(spearman([p[0] for p in pairs], [p[1] for p in pairs])
                       if len(pairs) >= 3 else float("nan"))
        print(f"  {m:<14}" + "".join(
            (f"{v:>13.2f}" if v == v else f"{'n/a':>13}") for v in row))

    print(f"\n  n = {len(runs)} runs. Treat these as directional only: rank")
    print("  correlation on this few points is noisy, and the runs are not")
    print("  independent draws -- they share an architecture and a seed.")

    # ---- 3. is it really the metric, or just training length? -----------
    print("\n" + "=" * 78)
    print("3. CONFOUND CHECK: HOW LONG DID EACH RUN TRAIN?")
    print("=" * 78)
    print("""
  Every validation metric being anti-predictive at once is suspicious. A
  simpler explanation: the runs differ in how many epochs they trained, and
  training longer fits the CALIBRATION period better while generalising worse
  to the HOLDOUT period. Validation lives in the calibration period, so it
  cannot see that happening -- it rewards exactly the overfitting that hurts.

  If epochs correlate with holdout outcome as strongly as the metrics do, the
  problem is not which metric we select on. It is that validation shares a
  time regime with training.
""")
    print(f"  {'run':<34}{'sel ep':>8}{'total':>8}{'HO nll':>10}{'HO f1':>9}")
    for r in sorted(runs, key=lambda x: (x["best_epoch"] if x["best_epoch"]
                                         is not None else -1)):
        tot = len(r["history"]) if r["history"] else 0
        print(f"  {r['tag'][:33]:<34}{r['best_epoch']:>8}{tot:>8}"
              f"{r['holdout']['nll']:>10.4f}{r['holdout']['f1_macro']:>9.4f}")

    eps = [r["best_epoch"] for r in runs if r["best_epoch"] is not None]
    for k in ("nll", "f1_macro"):
        ys = [r["holdout"][k] for r in runs if r["best_epoch"] is not None]
        print(f"\n  Spearman(selected epoch, holdout {k}) = "
              f"{spearman(eps, ys):+.2f}")
    print("""
  A strong POSITIVE correlation with holdout NLL means: the longer a run
  trained, the worse it did on the holdout period. If that is what we see, the
  fix is not a different validation metric -- it is a validation set that is
  temporally shifted from training, so early stopping can detect drift.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
