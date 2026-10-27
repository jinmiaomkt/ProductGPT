"""R48 step 3: gate and summary of the scored shards of one tag.

Gate (pre-registered): the shards' rows are exactly the scored customers' cells,
row for row, and the per-cell counts equal G1's (parametric.data.REFERENCE_CELLS).
    - no occasion is scored twice
    - the scored customers are exactly the --who group of the table
    - every row's label and cell agree with the table at (customer, pos)

Then the pre-registered metrics, aggregates only:
    per cell: n, NLL, hit rate, letter mass
    headline cell: per-class NLL and a customer bootstrap of the NLL
    references on the same rows: uniform (log 9), and the constant model that
    predicts the train cell's class shares, with a paired customer bootstrap of
    the LLM-minus-constant gap (H2)
Exploratory (not pre-registered): NLL by position in the 1,024-occasion window,
per cell, for the LLM and the constant.

Nothing about individual customers is printed.

USAGE (HPCC)
    python scripts/r48_summarise.py --table <r46>/occasions_v6_S1024.npz \
        --run-dir <runs>/r48 --tag qwen3_8b_out --n-shards 2 --who out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_zeroshot.score import CELL_NAMES, HEADLINE  # noqa: E402
from parametric.data import REFERENCE_CELLS, load, roles  # noqa: E402

POS_BINS = (0, 64, 256, 512, 768, 1024)


def cell_metrics(p, m):
    hit = (p["logp9"][m].argmax(1) + 1) == p["label"][m]
    return {"n": int(m.sum()), "nll": float(p["nll"][m].astype(np.float64).mean()),
            "hit_rate": float(hit.mean()),
            "letter_mass": float(np.exp(p["mass"][m].astype(np.float64)).mean())}


def bootstrap(uid, x, draws, seed):
    """Mean of x and its 95% interval from a bootstrap over customers."""
    _, inv = np.unique(uid, return_inverse=True)
    s, c = np.bincount(inv, x), np.bincount(inv).astype(np.float64)
    idx = np.random.default_rng(seed).integers(0, len(s), (draws, len(s)))
    reps = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"mean": float(s.sum() / c.sum()), "ci95": [float(lo), float(hi)],
            "customers": int(len(s)), "draws": draws}


def main() -> None:
    ap = argparse.ArgumentParser(description="R48: gate and summary of the scored shards")
    ap.add_argument("--table", required=True, help="R46 occasions_v6_S1024.npz")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--who", choices=("out", "in"), default="out")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run_dir)
    parts = [load(run / f"{a.tag}_shard{k}.npz") for k in range(a.n_shards)]
    p = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    tab = load(Path(a.table))
    r = roles(tab)

    # ---- gate
    code = np.full(tab["label"].shape, -1, np.int8)
    for k, name in enumerate(CELL_NAMES):
        code[r[name]] = k
    tuid = tab["uid_hash"]
    order = np.argsort(tuid)
    row = order[np.minimum(np.searchsorted(tuid[order], p["uid_hash"]), len(tuid) - 1)]
    pos = p["pos"].astype(np.int64)
    pairs = np.stack([p["uid_hash"].astype(np.int64), pos], 1)
    want = set(tuid[(tab["is_out"] == 1) == (a.who == "out")].tolist())
    got = set(np.unique(p["uid_hash"]).tolist())
    counts = {name: int((p["cell"] == k).sum()) for k, name in enumerate(CELL_NAMES)}
    gate = {
        "duplicate_rows": int(len(pairs) - len(np.unique(pairs, axis=0))),
        "customers": len(got), "customers_match_table": got == want,
        "uid_found": bool((tuid[row] == p["uid_hash"]).all()),
        "labels_match_table": bool((tab["label"][row, pos] == p["label"]).all()),
        "cells_match_table": bool((code[row, pos] == p["cell"]).all()),
        "cells": {n: {"scored": counts[n], "G1": g} for n, g in REFERENCE_CELLS.items()
                  if counts[n]},
    }
    gate["pass"] = bool(gate["duplicate_rows"] == 0 and gate["customers_match_table"]
                        and gate["uid_found"] and gate["labels_match_table"]
                        and gate["cells_match_table"] and gate["cells"]
                        and all(v["scored"] == v["G1"] for v in gate["cells"].values()))

    # ---- the constant reference: the train cell's class shares
    shares = np.bincount(tab["label"][r["train"]].astype(np.int64), minlength=10)[1:]
    shares = shares / shares.sum()
    lab = p["label"].astype(np.int64)
    const_nll = -np.log(shares[lab - 1])

    out = {"tag": a.tag, "gate": gate, "uniform_nll": math.log(9),
           "train_class_shares": [round(float(s), 5) for s in shares], "cells": {}}
    for k, name in enumerate(CELL_NAMES):
        m = p["cell"] == k
        if m.any():
            out["cells"][name] = dict(cell_metrics(p, m),
                                      constant_nll=float(const_nll[m].mean()))

    h = p["cell"] == HEADLINE
    out["headline"] = {
        "nll": bootstrap(p["uid_hash"][h], p["nll"][h].astype(np.float64), a.draws, a.seed),
        "llm_minus_constant": bootstrap(p["uid_hash"][h],
                                        p["nll"][h].astype(np.float64) - const_nll[h],
                                        a.draws, a.seed),
        "per_class": {str(c): {"n": int((lab[h] == c).sum()),
                               "nll": float(p["nll"][h][lab[h] == c].astype(np.float64).mean())}
                      for c in range(1, 10) if (lab[h] == c).any()},
    }

    expl = {}
    for k, name in enumerate(CELL_NAMES):
        m = p["cell"] == k
        if not m.any():
            continue
        expl[name] = []
        for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
            b = m & (pos >= lo) & (pos < hi)
            if b.any():
                expl[name].append({"pos": [lo, hi], "n": int(b.sum()),
                                   "nll": float(p["nll"][b].astype(np.float64).mean()),
                                   "constant_nll": float(const_nll[b].mean())})
    out["exploratory_nll_by_position"] = expl

    (run / f"{a.tag}_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not gate["pass"]:
        raise SystemExit("GATE FAILED")


if __name__ == "__main__":
    main()
