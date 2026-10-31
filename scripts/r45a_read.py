"""R45 part A -- censored training and testing: the pre-registered reading.

Opens the HOLDOUT. Run it once, after all 39 part A runs have finished
(EXPERIMENTS.md R45):

    python3 scripts/r45a_read.py

The plan fixes the rules but named no script. This one was written on 31 Oct
2026, after the runs finished and before any holdout number was read, and
it implements the plan's rules unchanged.

Runs: ~/ProductGPT/runs/*_r45a_<arm>_k<k>_s<seed>/**/final.json for arms L, T,
Z, ZF at calendar depth k in {1, 2, 4}, and *_r45a_Zcount_d128_s<seed>. Arm F
(no chop) is R44's stage-4 gen-6 runs, *_b39_g6w384_s<seed>, on the same
seeds 1-3. Every cfg is checked against R44's frozen gen-6 configuration and
its arm's chop settings, and every run must score the same holdout events.

Headline: NLL on outsample_users_holdout_period, paired by seed. The trainer
selected each run's epoch on campaign-27 validation.

Differences, NLL(a) - NLL(b) paired by seed; positive = a is worse:
  Z - L    the cost of a missing start, net of lost training volume
  Z - T    the part of that cost from the state
  T - L    the part from the context
  Z - ZF   how much of the cost the fix recovers
  L - F    the lost training volume alone (no rule in the plan; descriptive)

Rules (pre-registered, 3 seeds):
  - A cost exists at depth k if Z - L has the same sign on all 3 seeds and
    |mean| > 0.02; otherwise "not detectable". An existing cost with Z
    BETTER than L is printed as such and does not count against including
    the censored players. The same test is printed for Z - T, T - L and
    L - F as labelled descriptive lines; only Z - L carries the decision.
  - The fix recovers at depth k if mean(Z - ZF) > 1/2 mean(Z - L) and
    Z - ZF > 0 on all 3 seeds.
  - Including the 10,212: if no cost exists at any depth, or the fix
    recovers it at every depth where it exists, the next step is a
    pre-registered run that adds them with the flag on. Otherwise they stay
    out and the censoring cost is the reason.

Predictions checked:
  1. mean Z - L < 0.02 at every depth.
  2. mean Z - L is non-decreasing in k, and its 2 -> 4 step is smaller than
     its 1 -> 2 step (read here as "flattening by k = 4").
  3. mean Z - T >= mean T - L at every depth.
  Predictions 4 (part B) and 5 (generation, step 2) are not tested here.
Count cross-check, descriptive: Zcount_d128 - Z_k2, paired.
Other cell, no decision: insample_users_holdout_period.

Not computed here: the computed-state error the plan asks every table to
report. It is a property of the chop, not of the trained model (the counts
and pity depth are fixed functions of the rows), so it needs the data, not
these final.json files: r45_eval_chopped.state_error computes it for Z on the
same headline cell. T recovers the state exactly (test_r45_censor, 7.5e-08),
and L and F keep the whole history.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics as st

HEADLINE = "outsample_users_holdout_period"
OTHER = "insample_users_holdout_period"
N_EVENTS = {HEADLINE: 217303, OTHER: 222889}
SEEDS = [1, 2, 3]
DEPTHS = [1, 2, 4]
ARMS = ["L", "T", "Z", "ZF"]
STATE = {"L": "labels", "T": "true", "Z": "zero", "ZF": "zero"}
COST = 0.02
# R44's frozen gen-6 configuration (~/ProductGPT/r44_frozen_g6.env) and the
# data and split every R45 arm shares
FROZEN = {"window": 384, "d_model": 96, "d_ff": 288, "N": 6, "lr": 2.19e-04, "dropout": 0.1,
          "weight_decay": 0.05, "grad_accum": 2, "warmup_frac": 0.1, "num_heads": 8,
          "chunk": 512, "batch_size": 4, "val_from": 27, "gen": 6, "model_rev": 2,
          "split_seed": 33, "max_users": None,
          "data_file": "perproduct/clean_list_int_wide4_simple7_IPT.json"}


def find(root: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(root, pattern, "**", "final.json"), recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"{pattern}: expected one final.json, found {len(hits)}")
    return hits[0]


def check(label: str, path: str, seed: int, chop) -> dict:
    j = json.load(open(path))
    c = j["cfg"]
    bad = {}
    for k, v in dict(FROZEN, seed=seed).items():
        got = c.get(k, "<absent>")
        if isinstance(v, float) and isinstance(got, (int, float)):
            same = math.isclose(got, v, rel_tol=1e-9)
        else:
            same = got == v
        if not same:
            bad[k] = got
    if chop is None:
        if c.get("chop", "none") != "none":
            bad["chop"] = c.get("chop")
    else:
        bad.update({k: c.get(k) for k, v in chop.items() if c.get(k) != v})
    if bad:
        raise SystemExit(f"{label}: cfg {bad} differs from the plan")
    if j.get("stopped_for_time_budget"):
        raise SystemExit(f"{label}: stopped for the time budget")
    for cell, n in N_EVENTS.items():
        if int(j["test_cells"][cell]["n"]) != n:
            raise SystemExit(f"{label}: {cell} scored {j['test_cells'][cell]['n']} events, not {n}")
    return j


def load(root: str) -> dict:
    runs = {}
    for s in SEEDS:
        runs[("F", 0, s)] = check(f"F s{s}", find(root, f"*_b39_g6w384_s{s}"), s, None)
        for k in DEPTHS:
            for arm in ARMS:
                chop = {"chop": "calendar", "chop_depth": k, "chop_state": STATE[arm],
                        "chop_where": "both", "censor_flag": arm == "ZF"}
                runs[(arm, k, s)] = check(f"{arm} k{k} s{s}", find(root, f"*_r45a_{arm}_k{k}_s{s}"),
                                          s, chop)
        chop = {"chop": "count", "chop_depth": 128, "chop_state": "zero", "chop_where": "both",
                "censor_flag": False}
        runs[("Zcount", 128, s)] = check(f"Zcount s{s}", find(root, f"*_r45a_Zcount_d128_s{s}"),
                                         s, chop)
    seen = {}
    for key, j in runs.items():
        v = j["best_val_nll"]
        if v in seen:
            raise SystemExit(f"{key} and {seen[v]} have identical validation NLL {v}: a bug")
        seen[v] = key
    return runs


def hold(j: dict, cell: str = HEADLINE) -> float:
    return j["test_cells"][cell]["nll"]


def diff(runs: dict, a, b, cell: str = HEADLINE) -> list:
    return [hold(runs[a + (s,)], cell) - hold(runs[b + (s,)], cell) for s in SEEDS]


def rule(d: list) -> dict:
    m, sd = st.mean(d), st.stdev(d)
    same = all(x > 0 for x in d) or all(x < 0 for x in d)
    return {"d": d, "mean": m, "sd": sd, "same_sign": same, "beyond": abs(m) > COST,
            "exists": same and abs(m) > COST}


def show(label: str, r: dict, tail: str = "") -> None:
    print(f"    {label:<8}" + "  ".join(f"{x:+.4f}" for x in r["d"])
          + f"   mean {r['mean']:+.4f}  sd {r['sd']:.4f}"
          + f"   same sign {'yes' if r['same_sign'] else 'no '}"
          + f"  |mean|>{COST} {'yes' if r['beyond'] else 'no '}{tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--out", default="results/r45/partA_read.json")
    a = ap.parse_args()
    print(f"reader sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
    runs = load(a.runs_root)
    print(f"gates passed: 42 runs (39 part A + F on seeds 1-3), cfgs as planned, none stopped "
          f"for the time budget, every run scores {N_EVENTS[HEADLINE]:,} headline and "
          f"{N_EVENTS[OTHER]:,} in-sample holdout events, no two validation NLLs equal")
    print()
    print(f"===== R45 part A -- HOLDOUT OPENED ({HEADLINE})")
    print(f"{'arm':<7}{'k':>4} {'hold mean':>10} {'sd':>7}   {'hold per seed':<26} "
          f"{'val mean':>9}   best epochs")
    keys = [("F", 0)] + [(arm, k) for k in DEPTHS for arm in ARMS] + [("Zcount", 128)]
    table = {}
    for arm, k in keys:
        h = [hold(runs[(arm, k, s)]) for s in SEEDS]
        v = [runs[(arm, k, s)]["best_val_nll"] for s in SEEDS]
        eps = [runs[(arm, k, s)]["best_epoch"] for s in SEEDS]
        table[f"{arm}_{k}"] = {"hold": h, "val": v, "best_epoch": eps}
        print(f"{arm:<7}{k:>4} {st.mean(h):>10.4f} {st.stdev(h):>7.4f}   "
              f"{', '.join(f'{x:.4f}' for x in h):<26} {st.mean(v):>9.4f}   {eps}")
    print()

    per_k = {}
    for k in DEPTHS:
        zl = rule(diff(runs, ("Z", k), ("L", k)))
        zt = rule(diff(runs, ("Z", k), ("T", k)))
        tl = rule(diff(runs, ("T", k), ("L", k)))
        lf = rule(diff(runs, ("L", k), ("F", 0)))
        zzf = rule(diff(runs, ("Z", k), ("ZF", k)))
        recovers = zzf["mean"] > 0.5 * zl["mean"] and all(x > 0 for x in zzf["d"])
        cost = ("EXISTS (Z worse)" if zl["exists"] and zl["mean"] > 0 else
                "exists with Z BETTER than L" if zl["exists"] else "not detectable")
        print(f"k = {k}  (paired by seed 1, 2, 3)")
        show("Z - L", zl, f"   -> cost {cost}")
        show("Z - T", zt, "   (state part; descriptive)")
        show("T - L", tl, "   (context part; descriptive)")
        show("L - F", lf, "   (volume alone; descriptive)")
        show("Z - ZF", zzf, f"   -> fix recovers: {'YES' if recovers else 'no'} "
             f"(needs mean > {0.5 * zl['mean']:+.4f} and 3 of 3 positive)")
        print()
        per_k[k] = {"Z-L": zl, "Z-T": zt, "T-L": tl, "L-F": lf, "Z-ZF": zzf,
                    "cost_exists": zl["exists"] and zl["mean"] > 0,
                    "cost_exists_negative": zl["exists"] and zl["mean"] < 0,
                    "fix_recovers": recovers}

    costly = [k for k in DEPTHS if per_k[k]["cost_exists"]]
    unrecovered = [k for k in costly if not per_k[k]["fix_recovers"]]
    include = not unrecovered
    print("DECISION (including the 10,212 left-censored players):")
    print(f"  depths where a cost exists: {costly or 'none'}; of those the fix does not "
          f"recover: {unrecovered or 'none'}")
    print("  -> " + ("next step: a pre-registered run that adds them with the flag on" if include
                     else "they stay out; the censoring cost is reported as the reason"))
    print()

    m = {k: per_k[k]["Z-L"]["mean"] for k in DEPTHS}
    p1 = all(m[k] < COST for k in DEPTHS)
    mono = m[1] <= m[2] <= m[4]
    flat = (m[4] - m[2]) < (m[2] - m[1])
    p3 = {k: per_k[k]["Z-T"]["mean"] >= per_k[k]["T-L"]["mean"] for k in DEPTHS}
    print("PREDICTIONS")
    print(f"  1. mean Z - L < {COST} at every depth: {'holds' if p1 else 'fails'} "
          f"({', '.join(f'k{k} {m[k]:+.4f}' for k in DEPTHS)})")
    print(f"  2. monotone in depth: {'yes' if mono else 'no'}; flattening by k = 4 "
          f"(2->4 step {m[4] - m[2]:+.4f} < 1->2 step {m[2] - m[1]:+.4f}): {'yes' if flat else 'no'}")
    print("  3. Z - T >= T - L: " + ", ".join(
        f"k{k} {per_k[k]['Z-T']['mean']:+.4f} vs {per_k[k]['T-L']['mean']:+.4f} "
        f"{'holds' if p3[k] else 'fails'}" for k in DEPTHS))
    print("  4 (part B) and 5 (generation): not tested by part A")
    print()

    cc = rule(diff(runs, ("Zcount", 128), ("Z", 2)))
    print("COUNT CROSS-CHECK (Z at 128 occasions against Z at calendar k = 2; descriptive)")
    show("Zc - Z2", cc)
    print()
    print(f"OTHER CELL, no decision ({OTHER}):")
    other = {}
    for k in DEPTHS:
        o = {name: diff(runs, x, y, OTHER) for name, x, y in
             (("Z-L", ("Z", k), ("L", k)), ("Z-T", ("Z", k), ("T", k)),
              ("T-L", ("T", k), ("L", k)), ("Z-ZF", ("Z", k), ("ZF", k)))}
        other[k] = o
        print(f"  k = {k}  " + "   ".join(
            f"{name} " + " ".join(f"{x:+.4f}" for x in d) + f" (mean {st.mean(d):+.4f})"
            for name, d in o.items()))

    out = {"table": table, "per_k": {str(k): v for k, v in per_k.items()},
           "decision": {"costly": costly, "unrecovered": unrecovered, "include_10212": include},
           "predictions": {"1": p1, "2_monotone": mono, "2_flattening": flat,
                           "3": {str(k): v for k, v in p3.items()}},
           "count_check": cc, "other_cell": {str(k): v for k, v in other.items()}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
