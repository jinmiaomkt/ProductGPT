"""
Extract the REAL banner rotation, for R34 s4.

The simulation so far used a stylised rotation (the same products offered for
N consecutive occasions). The remaining question is what OUR calendar does, so
the identification statement is about this dataset rather than a stylisation.

AGGREGATE ONLY. This writes, per campaign: which product ids were offered, how
many distinct products, and the median number of occasions a customer spends in
that campaign. No customer identifier, no per-customer row, and no decision ever
leaves this script -- the output is a property of the game's calendar, not of
any consumer.

    python scripts/extract_offer_schedule.py --out results/r34/offer_schedule.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import config5
from dataset_multistream import parse_token_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-level", type=int, default=7, choices=[6, 7])
    ap.add_argument("--out", default="results/r34/offer_schedule.json")
    ap.add_argument("--lto-len", type=int, default=4)
    ap.add_argument("--ai-rate", type=int, default=15)
    a = ap.parse_args()

    cfg = config5.apply_vocab_level({}, a.vocab_level)
    data_file = Path(os.environ["PRODUCTGPT_DATA"]) / cfg["data_file"]
    print(f"[schedule] reading {data_file.name} (aggregates only)")
    records = json.loads(data_file.read_text())
    if isinstance(records, dict):                      # {key: [records]} or columnar
        records = next(v for v in records.values() if isinstance(v, list))

    offers = defaultdict(set)          # campaign -> {product id}
    spans = defaultdict(list)          # campaign -> occasions per customer
    first, last = cfg["first_prod_id"], cfg["last_prod_id"]
    n_users = 0
    for rec in records:
        if "CampaignID" not in rec or "AggregateInput" not in rec:
            continue
        n_users += 1
        ai = parse_token_ids(rec["AggregateInput"])
        camp = parse_token_ids(rec["CampaignID"])
        n_ev = min(len(camp), len(ai) // a.ai_rate)
        per_user = defaultdict(int)
        for t in range(n_ev):
            c = int(camp[t])
            if c <= 0:
                continue
            per_user[c] += 1
            for tok in ai[t * a.ai_rate: t * a.ai_rate + a.lto_len]:
                tok = int(tok)
                if first <= tok <= last:
                    offers[c].add(tok)
        for c, n in per_user.items():
            spans[c].append(n)

    out = {}
    for c in sorted(offers):
        out[str(c)] = {
            "products": sorted(offers[c]),
            "n_products": len(offers[c]),
            "median_occasions_per_customer": st.median(spans[c]) if spans[c] else 0,
            "n_customers": len(spans[c]),
        }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    ns = [v["n_products"] for v in out.values()]
    ms = [v["median_occasions_per_customer"] for v in out.values()]
    allp = sorted({p for v in out.values() for p in v["products"]})
    print(f"[schedule] {n_users:,} customers, {len(out)} campaigns, "
          f"{len(allp)} distinct products offered overall")
    print(f"[schedule] products per campaign: min {min(ns)} median {st.median(ns)} max {max(ns)}")
    print(f"[schedule] occasions per customer per campaign: median {st.median(ms)} "
          f"(min {min(ms)}, max {max(ms)})")
    print(f"[schedule] wrote {dest}")


if __name__ == "__main__":
    main()
