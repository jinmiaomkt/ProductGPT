"""
How diluted is the inventory memory the satiation cross-attention reads?

The offer-inventory cross-attention in gen 5 takes a softmax over EVERY product
token obtained so far: S x 10 slots, duplicates and low-rarity items included.
If most of those tokens are repeats of common items, a single rare acquisition
competes with hundreds of trivial ones for attention mass. The inventory GRU
has a second problem: every inserted NotBuy row feeds it an empty input, so its
state is updated -- and can decay -- on each quiet day.

This reports, at the end of each customer's calibration period (campaign <= 27):
  - obtained product tokens accumulated (what the attention softmax spans)
  - distinct products among them (what an additive inventory would hold)
  - the share of tokens by rarity tier
and across all rows, the share whose obtained block is empty.

Aggregate statistics only -- never uids, token values or per-customer records.

USAGE
    python scripts/inventory_dilution_check.py
"""
from __future__ import annotations

import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402
from shared.features import FEATURE_COLS, load_feature_tensor  # noqa: E402
import config5  # noqa: E402

PROD_LO, PROD_HI = 13, 56
LTO, OBT = 4, 10


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    root = os.environ.get("PRODUCTGPT_DATA")
    if not root:
        sys.exit("PRODUCTGPT_DATA is not set")
    feat = load_feature_tensor(config5.feature_path())
    rarity_col = FEATURE_COLS.index("Rarity")
    rarity = {t: round(float(feat[t, rarity_col]), 3) for t in range(PROD_LO, PROD_HI + 1)}
    tiers = sorted(set(rarity.values()))
    print(f"rarity values in the feature table (as stored): {tiers}")

    recs = load_json_dataset(str(Path(root) / "clean_list_int_wide4_simple6_IPT.json"))
    n_tokens, n_distinct, tier_share_rows = [], [], []
    empty_rows = total_rows = 0
    for rec in recs:
        ai = parse_token_ids(rec.get("AggregateInput"))
        camp = parse_token_ids(rec.get("CampaignID"))
        rate = LTO + OBT + 1
        n = min(len(ai) // rate, len(camp))
        toks = []
        for t in range(n):
            block = ai[t * rate + LTO: t * rate + LTO + OBT]
            prods = [x for x in block if PROD_LO <= x <= PROD_HI]
            total_rows += 1
            empty_rows += not prods
            if camp[t] <= 27:
                toks.extend(prods)
        if not toks:
            continue
        n_tokens.append(len(toks))
        n_distinct.append(len(set(toks)))
        c = Counter(rarity[x] for x in toks)
        tier_share_rows.append({k: c[k] / len(toks) for k in tiers})

    print(f"customers with any calibration-period acquisitions: {len(n_tokens):,}")
    print(f"rows with an empty obtained block: {empty_rows / total_rows:.1%} of {total_rows:,}")
    for name, xs in (("obtained tokens (attention span)", n_tokens),
                     ("distinct products (additive inventory)", n_distinct)):
        print(f"{name:<40} median {st.median(xs):>6.0f}   p90 {pct(xs, .9):>6}   max {max(xs):>6}")
    ratio = [a / b for a, b in zip(n_tokens, n_distinct)]
    print(f"{'tokens per distinct product':<40} median {st.median(ratio):>6.1f}   p90 {pct(ratio, .9):>6.1f}")
    for k in tiers:
        print(f"share of tokens at rarity {k}: mean {st.mean(r[k] for r in tier_share_rows):.1%}")


if __name__ == "__main__":
    main()
