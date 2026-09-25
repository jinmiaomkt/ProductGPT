"""
R30 stage 5: turn the scored diagnostics into the three tables for the paper.

    python3 scripts/stage5_report.py --res-dir results/r30/stage5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

CLASSES = ["Buy1 Reg", "Buy10 Reg", "Buy1 FigA", "Buy10 FigA", "Buy1 FigB",
           "Buy10 FigB", "Buy1 Wep", "Buy10 Wep", "NotBuy"]


def mean_sd(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", required=True)
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(Path(a.res_dir).glob("*.json")):
        groups[re.sub(r"_s\d+$", "", f.stem)].append(json.loads(f.read_text()))
    if not groups:
        raise SystemExit(f"no results in {a.res_dir}")
    order = sorted(groups, key=lambda k: st.mean(r["nll"] for r in groups[k]))

    print("=" * 100)
    print("EFFICIENCY  (out-of-sample x holdout period; mean +/- sd over seeds)")
    print("=" * 100)
    print(f"{'configuration':<26} {'n':>2} {'NLL':>17} {'params':>12} {'s/epoch':>9} "
          f"{'ECE':>8} {'Brier':>8}")
    for k in order:
        g = groups[k]
        m, s = mean_sd([r["nll"] for r in g])
        print(f"{k:<26} {len(g):>2} {m:>9.4f}+/-{s:<7.4f} {g[0]['params']:>12,} "
              f"{st.mean(r['secs_per_epoch'] for r in g):>9.0f} "
              f"{st.mean(r['ece'] for r in g):>8.4f} {st.mean(r['brier'] for r in g):>8.4f}")

    print("\n" + "=" * 100)
    print("PER-CLASS NLL  (the aggregate tie can hide this)")
    print("=" * 100)
    print(f"{'configuration':<26} " + " ".join(f"{c.split()[0][:6]:>7}" for c in CLASSES))
    share = None
    for k in order:
        g = groups[k]
        rows = [st.mean(r["class_nll"][i] for r in g) for i in range(9)]
        if share is None:
            tot = sum(g[0]["class_n"])
            share = [n / tot for n in g[0]["class_n"]]
        print(f"{k:<26} " + " ".join(f"{v:>7.3f}" for v in rows))
    print(f"{'(share of occasions)':<26} " + " ".join(f"{v:>7.1%}" for v in share))

    print("\n  purchase classes (1-8) vs NotBuy, occasion-weighted:")
    print(f"  {'configuration':<26} {'purchases':>10} {'NotBuy':>10} {'gap':>8}")
    for k in order:
        g = groups[k]
        cn = g[0]["class_n"]
        buy_n = sum(cn[:8])
        buy = sum(st.mean(r["class_nll"][i] for r in g) * cn[i] for i in range(8)) / max(buy_n, 1)
        nb = st.mean(r["class_nll"][8] for r in g)
        print(f"  {k:<26} {buy:>10.4f} {nb:>10.4f} {buy - nb:>+8.4f}")

    print("\n" + "=" * 100)
    print("CALIBRATION  (reliability: confidence vs accuracy, pooled over seeds)")
    print("=" * 100)
    for k in order:
        g = groups[k]
        n = [sum(r["bin_n"][i] for r in g) for i in range(len(g[0]["bin_n"]))]
        cf = [sum(r["bin_conf"][i] for r in g) for i in range(len(n))]
        ac = [sum(r["bin_acc"][i] for r in g) for i in range(len(n))]
        parts = [f"{cf[i] / n[i]:.2f}->{ac[i] / n[i]:.2f}" for i in range(len(n)) if n[i] > 200]
        print(f"  {k:<26} " + "  ".join(parts[-6:]))
    print("\n  read as confidence->accuracy in the most populated bins; equal = calibrated.")


if __name__ == "__main__":
    main()
