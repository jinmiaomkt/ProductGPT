"""R49: pair larger zero-shot LLMs with R48's Qwen3-8B, occasion by occasion.

Each --arm and the --ref are tags of scored shards from llm_zeroshot.score, each
already gated by scripts/r48_summarise.py.  Each is given as TAG:DIR:NSHARDS.

Gate: every arm holds exactly the reference's occasions in the two
out-of-sample cells -- the same customers, positions, labels and cells, each
once.  (Same table, same customers; only the model differs.)

The statistic is d = NLL(arm) - NLL(ref): negative = the arm is better than
Qwen3-8B.  Between two arms, d = NLL(second) - NLL(first).  Cluster bootstrap
over customers with one set of draws for every statistic (r48_pair_gru.Bootstrap).

Reported (aggregates only):
    pre-registered: headline cell: each model's NLL, each arm's d vs the
    reference, and d between the arms
    exploratory: the same by position in the 1,024-occasion window, per cell;
    d per decision in the headline cell; letter mass per model

USAGE (HPCC)
    python scripts/r49_pair_llms.py --ref qwen3_8b_out:<runs>/r48:2 \
        --arm qwen3_32b_out:<runs>/r49:4 --arm qwen36_27b_out:<runs>/r49:4 \
        --out <runs>/r49/r49_vs_qwen3_8b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm_zeroshot.score import CELL_NAMES, HEADLINE  # noqa: E402
from parametric.data import load  # noqa: E402
from r48_pair_gru import CELLS, POS_BINS, Bootstrap, sort_rows, summ  # noqa: E402


def read_tag(spec: str):
    tag, rest = spec.split(":", 1)
    d, n = rest.rsplit(":", 1)                   # DIR may itself hold a colon (C:\...)
    parts = [load(Path(d) / f"{tag}_shard{k}.npz") for k in range(int(n))]
    L = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    keep = np.isin(L["cell"], CELLS)
    L = {k: v[keep] for k, v in L.items()}
    o = sort_rows(L["uid_hash"], L["pos"])
    return tag, {k: v[o] for k, v in L.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="R49: pair larger LLMs with Qwen3-8B")
    ap.add_argument("--ref", required=True, help="TAG:DIR:NSHARDS of the reference (R48)")
    ap.add_argument("--arm", action="append", required=True, help="TAG:DIR:NSHARDS, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ref_tag, R = read_tag(a.ref)
    arms = [read_tag(s) for s in a.arm]
    gate = {}
    for tag, A in arms:
        g = {"rows": int(len(A["nll"])), "rows_ref": int(len(R["nll"]))}
        same = g["rows"] == g["rows_ref"]
        if same:
            g["same_occasions"] = bool((A["uid_hash"] == R["uid_hash"]).all()
                                       and (A["pos"] == R["pos"]).all())
            g["same_labels"] = bool((A["label"] == R["label"]).all())
            g["same_cells"] = bool((A["cell"] == R["cell"]).all())
        g["pass"] = bool(same and g["same_occasions"] and g["same_labels"] and g["same_cells"])
        gate[tag] = g
    out = {"ref": ref_tag, "arms": [t for t, _ in arms], "gate": gate,
           "gate_pass": all(g["pass"] for g in gate.values())}
    if not out["gate_pass"]:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        raise SystemExit("GATE FAILED: an arm does not hold the reference's occasions")

    nll = {ref_tag: R["nll"].astype(np.float64)}
    mass = {ref_tag: np.exp(R["mass"].astype(np.float64))}
    for tag, A in arms:
        nll[tag] = A["nll"].astype(np.float64)
        mass[tag] = np.exp(A["mass"].astype(np.float64))
    contrasts = {f"{t}_minus_{ref_tag}": nll[t] - nll[ref_tag] for t, _ in arms}
    for (t1, _), (t2, _) in combinations(arms, 2):
        contrasts[f"{t2}_minus_{t1}"] = nll[t2] - nll[t1]

    lab = R["label"].astype(np.int64)
    pos = R["pos"].astype(np.int64)
    _, cid = np.unique(R["uid_hash"], return_inverse=True)
    B = Bootstrap(cid, a.draws, a.seed)
    out.update(customers=int(B.C), draws=a.draws)

    h = R["cell"] == HEADLINE
    out["headline"] = {"n": int(h.sum()),
                       "nll": {t: summ(*B.reps(h, x)) for t, x in nll.items()},
                       "d": {k: summ(*B.reps(h, x)) for k, x in contrasts.items()},
                       "letter_mass": {t: float(x[h].mean()) for t, x in mass.items()},
                       "d_per_class": {}}
    for c in range(1, 10):
        m = h & (lab == c)
        if m.any():
            out["headline"]["d_per_class"][str(c)] = {
                "n": int(m.sum()), **{k: summ(*B.reps(m, x)) for k, x in contrasts.items()}}

    expl = {}
    for c in CELLS:
        rows = []
        for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
            m = (R["cell"] == c) & (pos >= lo) & (pos < hi)
            if m.any():
                rows.append({"pos": [lo, hi], "n": int(m.sum()),
                             "nll": {t: summ(*B.reps(m, x)) for t, x in nll.items()},
                             "d": {k: summ(*B.reps(m, x)) for k, x in contrasts.items()}})
        expl[CELL_NAMES[c]] = rows
    out["exploratory_by_position"] = expl

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
