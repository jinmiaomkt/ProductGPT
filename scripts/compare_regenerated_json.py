"""
Prove a regenerated JSON differs from the original ONLY where the R26 fix says it should.

The R26 fix changes how obtained items are decoded (version-2 code -> ProductID ->
NewProductIndex6, instead of reading the version-2 code as a version-6 key).
So, comparing old and new files customer by customer:
  - the uid set, and every field other than AggregateInput and Item, must be identical;
  - inside AggregateInput, the 4 offer tokens and the previous-decision token must be identical;
  - every obtained token that changed must be a pair (old, new) = (buggy_id, fixed_id)
    produced by the SAME version-2 code.

Aggregate counts only; no customer records are printed.

USAGE
    python scripts/compare_regenerated_json.py --meta-dir <Data folder> --old <file> --new <file>
Relative file paths resolve against PRODUCTGPT_DATA.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402

R, LTO_W, OBT_W = 15, 4, 10


def uid_of(r) -> str:
    u = r["uid"]
    return str(u[0] if isinstance(u, (list, tuple)) else u)


def ipt_rounding_only(a, b) -> bool:
    a = (a[0] if isinstance(a, list) else a).split()
    b = (b[0] if isinstance(b, list) else b).split()
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= 0.0100001 for x, y in zip(a, b))


def resolve(p: str) -> str:
    return p if os.path.isabs(p) else str(Path(os.environ["PRODUCTGPT_DATA"]) / p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    meta = Path(a.meta_dir)
    v2 = pd.read_excel(meta / "FigureWeaponIndex2.xlsx")
    v6 = pd.read_excel(meta / "FigureWeaponIndex6.xlsx")
    lv6 = dict(zip(v6.NewProductIndex.astype(int), v6.NewProductIndex6.astype(int)))
    pid6 = dict(zip(v6.ProductID.astype(str), v6.NewProductIndex6.astype(int)))
    allowed = {(lv6[int(c)], pid6[str(p)]) for c, p in zip(v2.NewProductIndex, v2.ProductID)}

    old = {uid_of(r): r for r in load_json_dataset(resolve(a.old))}
    new = {uid_of(r): r for r in load_json_dataset(resolve(a.new))}
    print(f"customers: old {len(old):,}, new {len(new):,}, identical uid sets: {set(old) == set(new)}")

    fields = Counter()
    ipt_rounding = 0
    offers_diff = prev_diff = len_diff = 0
    pair = Counter()
    for uid in set(old) & set(new):
        o, n = old[uid], new[uid]
        for k in set(o) | set(n):
            if k in ("AggregateInput", "Item") or o.get(k) == n.get(k):
                continue
            if k == "IPT" and ipt_rounding_only(o[k], n[k]):
                ipt_rounding += 1   # sprintf("%.2f") rounds halfway values differently across platforms
                continue
            fields[k] += 1
        ao, an = parse_token_ids(o["AggregateInput"]), parse_token_ids(n["AggregateInput"])
        if len(ao) != len(an):
            len_diff += 1
            continue
        for t in range(len(ao) // R):
            b = t * R
            offers_diff += ao[b:b + LTO_W] != an[b:b + LTO_W]
            prev_diff += ao[b + R - 1] != an[b + R - 1]
            for x, y in zip(ao[b + LTO_W:b + LTO_W + OBT_W], an[b + LTO_W:b + LTO_W + OBT_W]):
                pair["unchanged" if x == y else ("changed as predicted" if (x, y) in allowed else "changed UNEXPECTEDLY")] += 1
    print(f"other fields that differ (customers affected): {dict(fields) if fields else 'none'}")
    print(f"customers whose IPT differs only by 0.01 h rounding: {ipt_rounding:,}")
    print(f"customers with different sequence length: {len_diff}")
    print(f"rows whose offer tokens differ: {offers_diff}; rows whose previous decision differs: {prev_diff}")
    tot = sum(pair.values())
    for k in ("unchanged", "changed as predicted", "changed UNEXPECTEDLY"):
        print(f"obtained tokens {k}: {pair[k]:,} ({pair[k] / max(tot, 1):.4f})")
    ok = (set(old) == set(new) and not fields and not len_diff and not offers_diff and not prev_diff
          and pair["changed UNEXPECTEDLY"] == 0)
    print("VERDICT:", "only the predicted obtained-token corrections" if ok else "UNEXPECTED DIFFERENCES - investigate")


if __name__ == "__main__":
    main()
