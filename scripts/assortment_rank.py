"""
Can the substitution kernel be identified from THIS calendar? A rank condition.

Under a block rotation the likelihood sees the stock only through
assortment-level aggregates (technical note, eq. 7):

    K(j, A_c) = sum_p kappa(j,p) q_p 1[p in A_c]        for c = 1..C

For each product j this is a linear system K_j = M kappa_j, where M is the
(campaigns x products) incidence matrix weighted by acquisition rates. So:

    kappa_j is identified from campaign-level variation ONLY IF rank(M) = P.

A necessary condition is C >= P: at least as many campaigns as products. That
single inequality explains every simulation result -- no sample size can help,
because the constraint is on the design matrix, not the noise.

Two features can rescue it, and both are visible in the calendar:
  * RERUNS give products distinct campaign profiles instead of one appearance
    each, so columns of M stop being near-duplicates;
  * STAGGERED ENTRY means customers carry different subsets of past campaigns,
    which adds independent row combinations (each customer effectively supplies
    their own weighting of the rows).

This script measures all of it from the extracted schedule, and (with
--data) the entry-campaign distribution. Aggregates only.

    python3 scripts/assortment_rank.py --schedule results/r34/offer_schedule.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--data", action="store_true", help="also read entry campaigns")
    ap.add_argument("--level", choices=["campaign", "banner"], default="campaign",
                    help="granularity of the incidence matrix: a campaign pools its "
                         "banners, but customers CHOOSE a banner, so the banner cell is "
                         "the finer and more favourable unit")
    ap.add_argument("--ai-rate", type=int, default=15)
    a = ap.parse_args()

    raw = json.loads(Path(a.schedule).read_text())
    if "campaign" in raw and "banner" in raw:
        sched, banner = raw["campaign"], raw["banner"]
    else:
        sched, banner = raw, None
    camps = sorted(sched, key=lambda c: int(c))
    products = sorted({p for v in sched.values() for p in v["products"]})
    P = len(products)
    idx = {p: i for i, p in enumerate(products)}
    if banner and a.level == "banner":
        rows_src = [(k, v) for k, v in banner.items() if v]
        label = "campaign x banner cells"
    else:
        rows_src = [(c, sched[c]["products"]) for c in camps]
        label = "campaigns"
    C = len(rows_src)
    M = np.zeros((C, P))
    for r, (_, prods) in enumerate(rows_src):
        for p in prods:
            M[r, idx[p]] = 1.0
    appearances_rows = M.sum(0)

    appearances = appearances_rows
    rank = int(np.linalg.matrix_rank(M))
    sv = np.linalg.svd(M, compute_uv=False)
    print("=" * 78)
    print("THE RANK CONDITION FOR IDENTIFYING kappa FROM CAMPAIGN VARIATION")
    print("=" * 78)
    print(f"  rows ({label})   {C}")
    print(f"  products offered P               {P}")
    print(f"  rank(M)                          {rank}")
    print(f"  identified?                      "
          f"{'YES' if rank >= P else 'NO -- rank ' + str(rank) + ' < P = ' + str(P)}")
    print(f"  rows needed (at least)           {P}")
    if rank >= 1:
        nz = sv[sv > 1e-9]
        print(f"  condition number of M            {nz.max() / nz.min():.1f}")

    print("\n  reruns (a product offered in more than one campaign):")
    hist = Counter(int(x) for x in appearances)
    for k in sorted(hist):
        print(f"    offered in {k} campaign(s): {hist[k]} products")
    print(f"    products appearing more than once: {int((appearances > 1).sum())} of {P}")

    # how distinguishable are the products' campaign profiles?
    prof = M.T                                            # (P, C)
    dup = 0
    pairs_same = []
    for i in range(P):
        for j in range(i + 1, P):
            if np.array_equal(prof[i], prof[j]):
                dup += 1
                pairs_same.append((products[i], products[j]))
    print(f"\n  product pairs with IDENTICAL campaign profiles: {dup} "
          f"of {P * (P - 1) // 2}")
    print("    (these pairs can never be separated by campaign variation alone)")

    # pairs that are sometimes together, sometimes apart -- the useful contrast
    both = M.T @ M                                        # co-occurrence counts
    only_i = appearances[:, None] - both
    useful = int(((both > 0) & (only_i > 0) & (only_i.T > 0)).sum() / 2)
    print(f"  pairs offered together AND separately:         {useful}")
    print("    (these are the pairs whose relative kernel weight is learnable)")

    if a.data:
        import config5
        from dataset_multistream import parse_token_ids
        cfg = config5.apply_vocab_level({}, 7)
        f = Path(os.environ["PRODUCTGPT_DATA"]) / cfg["data_file"]
        print(f"\n  reading entry campaigns from {f.name} (aggregates only)")
        recs = json.loads(f.read_text())
        if isinstance(recs, dict):
            recs = next(v for v in recs.values() if isinstance(v, list))
        entries = []
        for rec in recs:
            if "CampaignID" not in rec:
                continue
            cp = [int(x) for x in parse_token_ids(rec["CampaignID"]) if int(x) > 0]
            if cp:
                entries.append(min(cp))
        e = np.asarray(entries)
        print(f"  customers                        {len(e):,}")
        print(f"  distinct entry campaigns         {len(np.unique(e))}")
        print(f"  entry campaign quartiles         {np.percentile(e, [25, 50, 75])}")
        share_first = float((e == e.min()).mean())
        print(f"  share entering in the first campaign present  {share_first:.1%}")
        print("\n  Staggered entry adds independent ROW combinations: each entry cohort")
        print("  weights the campaign history differently. It cannot raise rank(M)")
        print("  above C, so it conditions the system rather than rescuing it.")


if __name__ == "__main__":
    main()
