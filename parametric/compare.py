"""R46 part A read-out: the ladder table, R6 selection, and paired differences.

    python -m parametric.compare --runs-dir D [--cell outsample_users_holdout_period]

R6 selection follows the pre-registration: within each K the start with the
best TRAINING likelihood; across K the best campaign-27 (`val`) NLL. The
holdout never enters selection.

Paired differences are per occasion, matched on (customer, position), with a
customer-cluster bootstrap (1,000 replicates, no refits). Every reported rung
is compared with the rung below it and with Markov-1 (M1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

CELLS = ("val", "insample_users_holdout_period", "outsample_users_calib_period",
         "outsample_users_holdout_period")
# R30 stage 4: out-of-sample x holdout, 5 seeds per cell (EXPERIMENTS.md)
NEURAL = [("plain GRU, level 6", 0.8856, 0.0067), ("GRU encoder, level 6", 0.8904, 0.0031),
          ("transformer + recency, level 7", 0.8925, 0.0058), ("GRU encoder, level 7", 0.8944, 0.0089),
          ("hybrid, level 6", 0.8958, 0.0076), ("hybrid, level 7", 0.8962, 0.0138),
          ("plain GRU, level 7", 0.9033, 0.0152), ("transformer + recency, level 6", 0.9049, 0.0100)]
LADDER = ("M0", "M1", "R0", "R1", "R1b", "R2", "R3", "R4", "R5", "R6")
STEPS = [("R1", "R0", "last decision"), ("R1b", "R1", "habit persistence"),
         ("R2", "R1", "run structure"), ("R3", "R2", "long memory"),
         ("R4", "R3", "holdings, pity, guarantee"), ("R5", "R4", "nested structure"),
         ("R6", "R5", "persistent taste")]


def occ(run: Path, cell: str):
    with np.load(run / f"occ_{cell}.npz") as z:
        uh, pos = z["uid_hash"].astype(np.int64), z["pos"].astype(np.int64)
        o = np.lexsort((pos, uh))
        P = z["prob"].astype(np.float64)[o]
        y = z["label"].astype(np.int64)[o]
        nll = -np.log(np.clip(P[np.arange(len(y)), y - 1] / P.sum(1), 1e-12, None))
        key = np.stack([uh[o], pos[o]], 1)
        return key, uh[o], nll, y


def paired(a: Path, b: Path, cell: str, reps: int = 1000, seed: int = 0) -> Dict:
    ka, ua, na, ya = occ(a, cell)
    kb, _, nb, yb = occ(b, cell)
    if not (np.array_equal(ka, kb) and np.array_equal(ya, yb)):
        raise SystemExit(f"{a.name} and {b.name} score different rows in {cell}")
    d = na - nb                                      # negative = a is better
    cust, inv = np.unique(ua, return_inverse=True)
    s = np.bincount(inv, weights=d)
    n = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.integers(0, len(cust), len(cust)), minlength=len(cust))
        boots[r] = (w * s).sum() / (w * n).sum()
    buy = ya <= 8
    return {"diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "diff_purchase": float(d[buy].mean()), "diff_notbuy": float(d[~buy].mean()),
            "customers": int(len(cust)), "rows": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--cell", default="outsample_users_holdout_period")
    a = ap.parse_args()
    root = Path(a.runs_dir)
    res = {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/result.json")}

    # R5: four nest starts, keep the best TRAINING likelihood (the same rule
    # the R6 warm start uses)
    r5 = [r for r in res.values() if r["rung"] == "R5"]
    r5_best = min(r5, key=lambda r: r["train_nll"]) if r5 else None
    if r5:
        print("R5 nest starts (kept: best training NLL):")
        for r in sorted(r5, key=lambda r: r["start"]):
            der = r.get("derived", {})
            print(f"  start {r['start']}: train {r['train_nll']:.6f}  lambda_top "
                  f"{der.get('lambda_top', {}).get('value', float('nan')):.3f}  lambda_banner "
                  f"{der.get('lambda_banner', {}).get('value', float('nan')):.3f}"
                  f"{'  <- kept' if r is r5_best else ''}")
    # R6 selection: start by training likelihood within K, K by campaign-27 NLL
    by_k: Dict[int, Dict] = {}
    for tag, r in res.items():
        if r["rung"] == "R6":
            k = r["K"]
            if k not in by_k or r["train_nll"] < by_k[k]["train_nll"]:
                by_k[k] = r
    sel = None
    if by_k:
        sel = min(by_k.values(), key=lambda r: r["cells"]["val"]["nll"])
        print("R6 selection (start by train NLL, K by campaign-27 NLL):")
        for k in sorted(by_k):
            r = by_k[k]
            print(f"  K={k}: start {r['start']}  train {r['train_nll']:.5f}  val {r['cells']['val']['nll']:.5f}"
                  f"{'  <- selected' if r is sel else ''}")
    tags = {rung: rung for rung in LADDER if rung in res}
    if r5_best is not None:
        tags["R5"] = r5_best["tag"]
    if sel is not None:
        tags["R6"] = sel["tag"]

    print(f"\nLadder on {a.cell} (NLL nats; purchase / NotBuy split; FigB classes 5, 6):")
    print(f"{'rung':<6}{'P':>5}{'train':>9}{'val':>9}{'NLL':>9}{'buy':>8}{'NotBuy':>8}{'B1FigB':>8}{'B10FigB':>8}"
          f"{'hit':>7}{'F1':>7}{'ECE':>7}  derived")
    table = {}
    for rung, tag in tags.items():
        r = res[tag]
        c = r["cells"][a.cell]
        der = {k: round(v["value"], 3) for k, v in r.get("derived", {}).items() if isinstance(v, dict)}
        table[rung] = {"tag": tag, "rare": r.get("rare", []), "ridge": r.get("ridge", []),
                       "negative_definite": r.get("se_diag", {}).get("negative_definite"),
                       "n_params": r["n_params"], "train_nll": r["train_nll"],
                       "val_nll": r["cells"]["val"]["nll"], "cell": c, "derived": r.get("derived", {}),
                       "all_cells": {k: r["cells"][k]["nll"] for k in CELLS if k in r["cells"]}}
        print(f"{rung:<6}{r['n_params']:>5}{r['train_nll']:>9.4f}{r['cells']['val']['nll']:>9.4f}{c['nll']:>9.4f}"
              f"{c['nll_purchase']:>8.4f}{c['nll_notbuy']:>8.4f}{c['nll_per_class'].get('5', float('nan')):>8.3f}"
              f"{c['nll_per_class'].get('6', float('nan')):>8.3f}{c['hit']:>7.3f}{c['f1_macro']:>7.3f}{c['ece']:>7.4f}  {der}")

    print(f"\nPaired differences on {a.cell} (a minus b; negative = the added information helps):")
    diffs = {}
    for hi, lo, what in STEPS + [(x, "M1", "vs Markov-1") for x in tags if x not in ("M0", "M1")]:
        if hi in tags and lo in tags:
            d = paired(root / tags[hi], root / tags[lo], a.cell)
            matters = d["ci95"][1] < 0 or d["ci95"][0] > 0
            d.update({"what": what, "matters": matters, "material": d["diff"] <= -0.02})
            diffs[f"{hi}-{lo}"] = d
            print(f"  {hi:>4} - {lo:<4} {what:<26} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  buy {d['diff_purchase']:+.4f}  NotBuy {d['diff_notbuy']:+.4f}"
                  f"{'  matters' if matters else ''}{'  materially' if d['material'] else ''}")

    best = min((v for k, v in table.items() if k not in ("M0", "M1")), key=lambda v: v["val_nll"], default=None)
    if best is not None and a.cell == "outsample_users_holdout_period":
        x = best["cell"]["nll"]
        ref = NEURAL[0][1]
        verdict = "wins" if x < ref - NEURAL[0][2] else "tie" if x <= ref + 0.02 else "loses"
        print(f"\nBest rung by campaign-27 NLL: {best['tag']} at {x:.4f}; plain GRU L6 {ref:.4f}: "
              f"{verdict} ({x - ref:+.4f})")
        for name, m, s in NEURAL:
            print(f"  {name:<32} {m:.4f} ± {s:.4f}   parametric {x - m:+.4f}")
    (root / f"compare_{a.cell}.json").write_text(json.dumps({"table": table, "diffs": diffs,
                                                             "r6_by_k": {k: v["tag"] for k, v in by_k.items()},
                                                             "neural": NEURAL}, indent=1))


if __name__ == "__main__":
    main()
