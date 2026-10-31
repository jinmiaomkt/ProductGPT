"""R45 part C -- the real left-censored players: the pre-registered reading.

Opens the HOLDOUT. Run it once, after the evaluation job (scripts/r45c_eval.pbs)
has scored all 15 runs (EXPERIMENTS.md, R45 part C):

    python3 scripts/r45c_read.py

Written on 31 Oct 2026 with the plan, before any part C run started.

Inputs: results/r45/partC_eval.json (r45c_eval.py: every arm on the main and
the censored cells, the same partition, each run's final.json reproduced), and
each run's final.json for its cfg. Arm F is R44's stage-4 gen-6 runs,
*_b39_g6w384_s<seed>; I0 and I are *_r45c_I0_s<seed> and *_r45c_I_s<seed>.

Differences, NLL(a) - NLL(b) paired by seed 1-5; positive = a is worse.
Rule (the adoption rule): "hurts" if mean(a - b) > 0.02 and a - b > 0 on at
least 4 of 5 seeds; "helps" if mean(a - b) < -0.02 and a - b < 0 on at least
4 of 5; otherwise "no detectable difference".
  1. headline        I - F  on outsample_users_holdout_period
  2. censored cell   I - F  on censored_outsample_holdout_period
  3. flag            I - I0 on both
  4. the inclusive sample is eligible as the primary sample unless rule 1
     says "hurts"
Descriptive, no decision: I0 - F, the censored in-sample cell, the main
in-sample cell.

Predictions checked:
  1. headline: mean(I - F) in (-0.02, 0) and I - F < 0 on at least 3 of 5
  2. censored cell: rule 2 says "helps"
  3. flag: rule 3 says "no detectable difference" on both cells
  4. in every arm, both censored cells score a higher mean NLL than the headline
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics as st

HEAD = "outsample_users_holdout_period"
INS = "insample_users_holdout_period"
COUT = "censored_outsample_holdout_period"
CIN = "censored_insample_holdout_period"
CELLS = (HEAD, INS, COUT, CIN)
N_HEAD = 217303
SEEDS = [1, 2, 3, 4, 5]
TAG = {"F": "b39_g6w384", "I0": "r45c_I0", "I": "r45c_I"}
EXTRA = "perproduct/censored_list_int_wide4_simple7_IPT.json"
COST = 0.02
FROZEN = {"window": 384, "d_model": 96, "d_ff": 288, "N": 6, "lr": 2.19e-04, "dropout": 0.1,
          "weight_decay": 0.05, "grad_accum": 2, "warmup_frac": 0.1, "num_heads": 8,
          "chunk": 512, "batch_size": 4, "val_from": 27, "gen": 6, "model_rev": 2,
          "split_seed": 33, "max_users": None, "chop": "none",
          "data_file": "perproduct/clean_list_int_wide4_simple7_IPT.json"}
ARM_CFG = {"F": {"extra_data_file": None, "censor_flag": False},
           "I0": {"extra_data_file": EXTRA, "extra_train": True, "censor_flag": False},
           "I": {"extra_data_file": EXTRA, "extra_train": True, "censor_flag": True}}


def find(root: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(root, pattern, "**", "final.json"), recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"{pattern}: expected one final.json, found {len(hits)}")
    return hits[0]


def check_cfg(label: str, path: str, arm: str, seed: int) -> dict:
    j = json.load(open(path))
    c = j["cfg"]
    bad = {}
    for k, v in dict(FROZEN, seed=seed, **ARM_CFG[arm]).items():
        # R44's runs predate the R45 options, so absent means off for these
        default = {"extra_data_file": None, "chop": "none", "censor_flag": False}
        got = c.get(k, default.get(k, "<absent>"))
        same = math.isclose(got, v, rel_tol=1e-9) if (
            isinstance(v, float) and isinstance(got, (int, float))) else got == v
        if not same:
            bad[k] = got
    if bad:
        raise SystemExit(f"{label}: cfg {bad} differs from the plan")
    if j.get("stopped_for_time_budget"):
        raise SystemExit(f"{label}: stopped for the time budget")
    return j


def load(root: str, eval_path: str):
    ev = json.load(open(eval_path))
    by = {}
    for r in ev:
        key = (r["tag"], r["seed"])
        if key in by:
            raise SystemExit(f"{key} appears twice in {eval_path}")
        by[key] = r
    runs, finals = {}, {}
    for arm, tag in TAG.items():
        for s in SEEDS:
            if (tag, s) not in by:
                raise SystemExit(f"{arm} seed {s} ({tag}) is missing from {eval_path}")
            r = by[(tag, s)]
            for c in CELLS:
                if c not in r:
                    raise SystemExit(f"{arm} s{s}: no {c} in {eval_path}")
                if r[c].get("check_ok") is False:
                    raise SystemExit(f"{arm} s{s} {c}: did not reproduce final.json")
            if arm != "F" and not all("check_ok" in r[c] for c in CELLS):
                raise SystemExit(f"{arm} s{s}: its final.json lacks a censored cell")
            finals[(arm, s)] = check_cfg(f"{arm} s{s}", find(root, f"*_{tag}_s{s}"), arm, s)
            runs[(arm, s)] = r
    for c in CELLS:
        ns = {runs[k][c]["n"] for k in runs}
        if len(ns) != 1:
            raise SystemExit(f"{c}: the runs score different event counts {sorted(ns)}")
    if runs[("F", 1)][HEAD]["n"] != N_HEAD:
        raise SystemExit(f"{HEAD} scores {runs[('F', 1)][HEAD]['n']} events, not {N_HEAD}")
    seen = {}
    for key, j in finals.items():
        v = j["best_val_nll"]
        if v in seen:
            raise SystemExit(f"{key} and {seen[v]} have identical validation NLL {v}: a bug")
        seen[v] = key
    return runs, finals


def diff(runs, a: str, b: str, cell: str) -> list:
    return [runs[(a, s)][cell]["nll"] - runs[(b, s)][cell]["nll"] for s in SEEDS]


def rule(d: list) -> dict:
    m, sd = st.mean(d), st.stdev(d)
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    verdict = ("hurts" if m > COST and pos >= 4 else
               "helps" if m < -COST and neg >= 4 else "no detectable difference")
    return {"d": d, "mean": m, "sd": sd, "pos": pos, "neg": neg, "verdict": verdict}


def show(label: str, r: dict) -> None:
    print(f"    {label:<26}" + "  ".join(f"{x:+.4f}" for x in r["d"])
          + f"   mean {r['mean']:+.4f}  sd {r['sd']:.4f}  worse on {r['pos']}/5"
          + f"   -> {r['verdict']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--eval", default="results/r45/partC_eval.json")
    ap.add_argument("--out", default="results/r45/partC_read.json")
    a = ap.parse_args()
    print(f"reader sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
    runs, finals = load(a.runs_root, a.eval)
    print("gates passed: 15 runs (F, I0, I x seeds 1-5) in the evaluation, every final.json "
          "cell reproduced, cfgs as planned, none stopped for the time budget, every run "
          f"scores the same events in every cell ({N_HEAD:,} headline), no two validation "
          "NLLs equal")
    print()
    print("===== R45 part C -- HOLDOUT OPENED (mean NLL over seeds 1-5, seed sd)")
    print(f"{'arm':<4}" + "".join(f"{c:>36}" for c in CELLS) + "   best epochs")
    table = {}
    for arm in TAG:
        row = {c: [runs[(arm, s)][c]["nll"] for s in SEEDS] for c in CELLS}
        table[arm] = row
        eps = [finals[(arm, s)]["best_epoch"] for s in SEEDS]
        print(f"{arm:<4}" + "".join(f"{st.mean(row[c]):>28.4f} ({st.stdev(row[c]):.4f})"
                                    for c in CELLS) + f"   {eps}")
    print(f"events per cell: " + ", ".join(f"{c} {runs[('F', 1)][c]['n']:,}" for c in CELLS))
    print()

    r1 = rule(diff(runs, "I", "F", HEAD))
    r2 = rule(diff(runs, "I", "F", COUT))
    r3h = rule(diff(runs, "I", "I0", HEAD))
    r3c = rule(diff(runs, "I", "I0", COUT))
    print("RULES (paired by seed 1-5; positive = the first arm is worse)")
    show("1. I - F, headline", r1)
    show("2. I - F, censored out", r2)
    show("3. I - I0, headline", r3h)
    show("3. I - I0, censored out", r3c)
    eligible = r1["verdict"] != "hurts"
    print()
    print("DECISION: " + (
        "the inclusive sample (13,838 players) is ELIGIBLE as the primary training sample; "
        "the switch is Jin's and the coauthors' call" if eligible else
        "the censored players stay OUT; the headline cost is the reason"))
    if r3h["verdict"] != "no detectable difference" or r3c["verdict"] != "no detectable difference":
        print("  rule 3 separates I from I0: report both arms")
    print()

    desc = {"I0 - F, headline": rule(diff(runs, "I0", "F", HEAD)),
            "I0 - F, censored out": rule(diff(runs, "I0", "F", COUT)),
            "I - F, censored in": rule(diff(runs, "I", "F", CIN)),
            "I - F, main in-sample": rule(diff(runs, "I", "F", INS))}
    print("DESCRIPTIVE, no decision")
    for k, v in desc.items():
        show(k, v)
    print()

    p1 = -COST < r1["mean"] < 0 and r1["neg"] >= 3
    p2 = r2["verdict"] == "helps"
    p3 = r3h["verdict"] == r3c["verdict"] == "no detectable difference"
    p4 = all(st.mean(table[arm][c]) > st.mean(table[arm][HEAD]) for arm in TAG for c in (COUT, CIN))
    print("PREDICTIONS")
    print(f"  1. headline I - F in (-{COST}, 0), I ahead on >= 3 of 5: {'holds' if p1 else 'fails'} "
          f"(mean {r1['mean']:+.4f}, I ahead on {r1['neg']})")
    print(f"  2. censored cell: I helps: {'holds' if p2 else 'fails'} ({r2['verdict']})")
    print(f"  3. flag not detectable on both cells: {'holds' if p3 else 'fails'}")
    print(f"  4. censored cells worse than the headline in every arm: {'holds' if p4 else 'fails'}")

    out = {"table": table, "rules": {"1": r1, "2": r2, "3_head": r3h, "3_cens": r3c},
           "eligible": eligible, "descriptive": desc,
           "predictions": {"1": p1, "2": p2, "3": p3, "4": p4},
           "events": {c: runs[("F", 1)][c]["n"] for c in CELLS}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
