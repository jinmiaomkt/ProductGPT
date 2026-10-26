"""R46 step 0: token conventions the parametric state builder relies on.

Prints AGGREGATE counts only -- no uid, no record, no per-customer value.
Run once on each machine before trusting parametric/occasions.py:

    python scripts/r46_inspect_tokens.py

What it settles:
  1. the value range of the previous-decision token, and what the first event
     of a history carries there;
  2. what each of the four offer slots holds when nothing is on offer;
  3. whether any Figure-B purchase happens while slot 1 is empty (if not, the
     Figure-B alternatives can be masked rather than learned as unavailable);
  4. how the level-6 product table maps ids to rarity / figure / limited.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths  # noqa: E402
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402


def main() -> None:
    raw = load_json_dataset(str(paths.data_file("clean_list_int_wide4_simple6_IPT.json")))
    print(f"users {len(raw)}")
    prev_first, prev_rest = Counter(), Counter()
    slot = [Counter() for _ in range(4)]
    obt = Counter()
    figb_buy_empty = figb_buy = 0
    rows = 0
    prev_mismatch = 0
    fields = Counter()
    for rec in raw:
        fields.update(rec.keys())
        ai = np.asarray(parse_token_ids(rec["AggregateInput"]), dtype=np.int64)
        dec = np.asarray(parse_token_ids(rec["Decision"]), dtype=np.int64)
        n = min(len(ai) // 15, len(dec))
        blk = ai[: n * 15].reshape(n, 15)
        dec = dec[:n]
        rows += n
        prev_first[int(blk[0, 14])] += 1
        prev_rest.update(blk[1:, 14].tolist())
        # does the prev-decision token equal the previous label?
        prev_mismatch += int((blk[1:, 14] != dec[:-1]).sum())
        for s in range(4):
            v = blk[:, s]
            slot[s].update(np.where(v == 0, -1, np.where(v >= 13, 1, v)).tolist())
        o = blk[:, 4:14].ravel()
        obt.update(np.where(o == 0, -1, np.where((o >= 13) & (o <= 56), 1, o)).tolist())
        is_b = (dec == 5) | (dec == 6)
        figb_buy += int(is_b.sum())
        figb_buy_empty += int((is_b & (blk[:, 1] < 13)).sum())
    print(f"rows {rows}")
    print("fields", sorted(fields))
    print("prev-decision token at the FIRST event:", dict(sorted(prev_first.items())))
    print("prev-decision token elsewhere:", dict(sorted(prev_rest.items())))
    print(f"prev-decision token != previous label (non-first rows): {prev_mismatch}")
    for s in range(4):
        print(f"offer slot {s}: -1=zero, 1=product 13+, other=raw:", dict(sorted(slot[s].items())))
    print("obtained tokens: -1=zero, 1=product 13..56, other=raw:", dict(sorted(obt.items())))
    print(f"Figure-B purchases {figb_buy}; with slot 1 empty {figb_buy_empty}")

    tab = pd.read_excel(paths.product_feature_xlsx())
    print("product table rows", len(tab), "unique NewProductIndex6", tab["NewProductIndex6"].nunique())
    cols = [c for c in ("Rarity", "type_figure", "LTO") if c in tab.columns]
    g = tab.groupby("NewProductIndex6")[cols].nunique()
    print("ids whose rows disagree on", {c: int((g[c] > 1).sum()) for c in cols})
    agg = tab.groupby("NewProductIndex6")[cols].first()
    print("rarity x figure x LTO over level-6 ids:")
    print(agg.value_counts().sort_index().to_string())
    eth = [c for c in tab.columns if c.startswith("Ethnicity")]
    wt = [c for c in tab.columns if c.startswith("WeaponType")]
    print("ethnicity columns", eth)
    print("weapon-type columns", wt)


if __name__ == "__main__":
    main()
