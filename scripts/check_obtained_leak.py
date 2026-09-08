"""
Test whether a data file's `obtained` block leaks the current decision.

THE PROBLEM
-----------
AggregateInput is built from 15-token events:

    [ LTO x4 | ObtainedProducts x10 | PreviousDecision x1 ]

If the ObtainedProducts block at event t records what the user obtained AT t
rather than at t-1, it leaks the label: a user who did not draw obtained
nothing, so an all-zero block identifies y_t == 9 (NotBuy) exactly. This was
found in clean_list_int_wide4_simple6_IPT.json, where a gen-5 model scored
precision = recall = F1 = AUPRC = 1.000 on NotBuy.

Every generation consumes the same AggregateInput layout from the same R
generator, so every data file is worth checking.

USAGE
    python scripts/check_obtained_leak.py                       # default: the IPT file
    python scripts/check_obtained_leak.py --data-file clean_list_int_wide4_simple6.json
    python scripts/check_obtained_leak.py --data-file X.json --shift   # test the fixed path
    python scripts/check_obtained_leak.py --all                 # every file in PRODUCTGPT_DATA

Reports ONLY aggregate contingency counts -- never token values or records.

HOW TO READ IT
    P(y_t=9 | obtained all zero) == 1.0  and no counter-examples
        -> the block describes t. The label is readable from the input.
           Any model trained on this file has inflated NotBuy metrics.
    P(y_(t-1)=9 | obtained all zero) high instead
        -> the block describes t-1, as intended. No leak from this source.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

import paths
from dataset_multistream import TransformerDataset, load_json_dataset


def analyse(path: Path, n_users: int, shift: bool) -> dict:
    raw = load_json_dataset(str(path))
    if n_users:
        raw = raw[:n_users]
    if not raw:
        return {}
    if "AggregateInput" not in raw[0] or "Decision" not in raw[0]:
        print(f"  skipped: no AggregateInput/Decision field")
        return {}

    ds = TransformerDataset(raw, ai_rate=15, lto_len=4, obtained_len=10,
                            prev_dec_len=1, max_events=1024,
                            shift_obtained=shift)
    cur, prev = Counter(), Counter()
    n = 0
    for i in range(len(ds)):
        it = ds[i]
        obt, lab, pv = it["obtained"], it["label"], it["prev_decision"]
        S = min(obt.size(0), lab.size(0), pv.size(0))
        if S == 0:
            continue
        obt, lab, pv = obt[:S], lab[:S], pv[:S]
        valid = (lab >= 1) & (lab <= 9)
        zero = (obt == 0).all(dim=1)
        n += int(valid.sum())
        cur["z1"] += int((zero & (lab == 9) & valid).sum())
        cur["z0"] += int((zero & (lab != 9) & valid).sum())
        cur["n1"] += int((~zero & (lab == 9) & valid).sum())
        cur["n0"] += int((~zero & (lab != 9) & valid).sum())
        prev["z1"] += int((zero & (pv == 9) & valid).sum())
        prev["z0"] += int((zero & (pv != 9) & valid).sum())
        prev["n1"] += int((~zero & (pv == 9) & valid).sum())
        prev["n0"] += int((~zero & (pv != 9) & valid).sum())
    return {"n": n, "cur": cur, "prev": prev, "users": len(raw)}


def show(title: str, t: Counter) -> tuple[float, bool]:
    a, b, c, d = t["z1"], t["z0"], t["n1"], t["n0"]
    p_zero = a / (a + b) if (a + b) else float("nan")
    determines = (b == 0 and a > 0)
    print(f"    {title}")
    print(f"      all-zero block : {a:>10,} positive   {b:>10,} negative")
    print(f"      has products   : {c:>10,} positive   {d:>10,} negative")
    print(f"      P(positive | all-zero) = {p_zero:.4f}"
          f"{'   <-- DETERMINES THE LABEL' if determines else ''}")
    return p_zero, determines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="clean_list_int_wide4_simple6_IPT.json")
    ap.add_argument("--all", action="store_true",
                    help="check every clean_list*.json in PRODUCTGPT_DATA")
    ap.add_argument("--n-users", type=int, default=400,
                    help="users to sample (0 = all; 400 is plenty)")
    ap.add_argument("--shift", action="store_true",
                    help="apply the shift_obtained fix before measuring")
    args = ap.parse_args()

    if args.all:
        files = sorted(p.name for p in paths.data_dir().glob("clean_list*.json"))
    else:
        files = [args.data_file]

    print(f"shift_obtained = {args.shift}\n")
    verdicts = []
    for name in files:
        try:
            path = paths.data_file(name)
        except Exception as e:
            print(f"{name}\n  skipped: {e}\n")
            continue
        print(f"{name}")
        r = analyse(path, args.n_users, args.shift)
        if not r:
            print()
            continue
        print(f"  {r['users']} users, {r['n']:,} scored events")
        p_cur, leak = show("vs CURRENT decision y_t == 9", r["cur"])
        p_prev, _ = show("vs PREVIOUS decision y_(t-1) == 9", r["prev"])
        verdict = ("LEAK" if leak else
                   "ok (block describes t-1)" if p_prev > p_cur else "ok")
        print(f"  VERDICT: {verdict}\n")
        verdicts.append((name, verdict))

    if len(verdicts) > 1:
        print("=" * 60)
        for name, v in verdicts:
            print(f"  {v:<28} {name}")


if __name__ == "__main__":
    main()
