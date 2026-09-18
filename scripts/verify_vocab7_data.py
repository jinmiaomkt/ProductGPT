"""
Verify the per-product (vocab 7) data against the level-6 file (R28).

Every level-7 product token maps to exactly one level-6 token (its pooled
group). Collapsing level 7 that way must reproduce level 6 token for token --
that is what makes the two files comparable. Also reports how much token
resolution the new vocabulary buys, as entropy of the obtained stream.

Aggregate statistics only.

USAGE
    python scripts/verify_vocab7_data.py --v6 clean_list_int_wide4_simple6_IPT.json \
        --v7 perproduct/clean_list_int_wide4_simple7_IPT.json
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402

R, LTO_W, OBT_W = 15, 4, 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", default="clean_list_int_wide4_simple6_IPT.json")
    ap.add_argument("--v7", default="perproduct/clean_list_int_wide4_simple7_IPT.json")
    ap.add_argument("--vocab-table", default="perproduct/ProductVocab7.xlsx")
    a = ap.parse_args()
    root = Path(os.environ["PRODUCTGPT_DATA"])
    tab = pd.read_excel(root / a.vocab_table)
    collapse = {int(k): int(v) for k, v in zip(tab.NewProductIndex7, tab.NewProductIndex6)}
    last7 = int(tab.NewProductIndex7.max())
    collapse.update({0: 0, last7 + 1: 57, last7 + 2: 58, last7 + 3: 59})  # specials
    collapse[12] = 12  # [UNK]: the R export writes literal NA for a handful of items

    uid = lambda r: str(r["uid"][0] if isinstance(r["uid"], list) else r["uid"])
    d6 = {uid(r): r for r in load_json_dataset(str(root / a.v6))}
    d7 = {uid(r): r for r in load_json_dataset(str(root / a.v7))}
    print(f"customers: v6 {len(d6):,}, v7 {len(d7):,}, identical uid sets: {set(d6) == set(d7)}")

    bad_collapse = other_fields = len_diff = 0
    h6, h7 = Counter(), Counter()
    unmapped = Counter()
    for k in set(d6) & set(d7):
        r6, r7 = d6[k], d7[k]
        for f in ("Decision", "CampaignID", "IPT", "IsInserted", "WhetherDraw", "HowManyDraw"):
            if f in r6 and r6.get(f) != r7.get(f):
                other_fields += 1
                break
        a6, a7 = parse_token_ids(r6["AggregateInput"]), parse_token_ids(r7["AggregateInput"])
        if len(a6) != len(a7):
            len_diff += 1
            continue
        for i, (x, y) in enumerate(zip(a6, a7)):
            if i % R >= LTO_W + OBT_W:
                continue          # previous-decision slot: a decision id, not a product
            c = collapse.get(y)
            if c is None:
                unmapped[y] += 1
            elif c != x:
                bad_collapse += 1
        for t in range(len(a6) // R):
            b = t * R
            for x in a6[b + LTO_W:b + LTO_W + OBT_W]:
                if x: h6[x] += 1
            for y in a7[b + LTO_W:b + LTO_W + OBT_W]:
                if y: h7[y] += 1

    ent = lambda c: -sum(v / sum(c.values()) * math.log(v / sum(c.values())) for v in c.values())
    print(f"customers with a different sequence length: {len_diff}")
    print(f"customers whose other fields differ: {other_fields}")
    print(f"tokens that do not collapse back to level 6: {bad_collapse:,}")
    print(f"level-7 tokens with no mapping: {dict(unmapped) if unmapped else 'none'}")
    print(f"obtained stream: {len(h6)} ids, entropy {ent(h6):.3f} nats (level 6)")
    print(f"                 {len(h7)} ids, entropy {ent(h7):.3f} nats (level 7)")
    ok = not (bad_collapse or unmapped or len_diff or other_fields) and set(d6) == set(d7)
    print("VERDICT:", "level 7 collapses exactly onto level 6" if ok else "MISMATCH - investigate")


if __name__ == "__main__":
    main()
