"""
Verify the product-name -> integer mappings behind the offer and obtained streams (R26).

Background. The R pipeline encodes products twice:
  1. GenerateDecisionSequence*.R turns game item ids into integers per row
     (ItemsJustGotIndex, and the campaign offer columns).
  2. GenerateJSON.R / InsertNotBuy_GenerateJSON_IPT.R turn those integers into
     the 60-token vocabulary (NewProductIndex6) with FigureWeaponIndex6.xlsx:
       offers   -> map_vec_lv6B, keyed by ProductIndex
       obtained -> map_vec_lv6,  keyed by NewProductIndex
ProductIndex is stable across FigureWeaponIndex versions. NewProductIndex is
NOT: version 3 inserted two standard 5-star characters, renumbering 89 of 122
products. If step 1 wrote version-2 NewProductIndex codes, step 2 decodes 45
products to the wrong id -- including two 3-star weapons onto the ids of two
limited-time 5-star characters.

Checks (lookup tables are game metadata; the JSON is read for AGGREGATES only):
  A. every lookup table: no duplicate / missing ProductID, ProductIndex, Name
  B. each NewProductIndex6 id 18-56 holds exactly one product, named as in the
     feature table SelectedFigureWeaponEmbeddingIndex.xlsx
  C. CampaignWideIndex: offer names agree with their *Index columns
  D. offers in the JSON reproduce CampaignWideIndex exactly
  E. obtained stream: which encoding explains the data?
       - any value > 59 would mean ProductIndex codes passed through unmapped
       - version-2 codes predict ids 36 and 37 dominate the limited-time ids

USAGE
    python scripts/verify_product_index.py --meta-dir <folder with the xlsx lookup tables>
        [--data-file clean_list_int_wide4_simple6_IPT.json]
The data file is resolved against PRODUCTGPT_DATA.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402

R, LTO_W, OBT_W = 15, 4, 10
OFFER_COLS = ("Figure5A", "Figure5B", "Weapon5A", "Weapon5B")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--data-file", default="clean_list_int_wide4_simple6_IPT.json")
    a = ap.parse_args()
    meta = Path(a.meta_dir)
    tabs = {k: pd.read_excel(meta / f"FigureWeaponIndex{'' if k == 1 else k}.xlsx") for k in range(1, 7)}
    v2, v6 = tabs[2], tabs[6]
    feat = pd.read_excel(meta / "SelectedFigureWeaponEmbeddingIndex.xlsx")
    camp = pd.read_excel(meta / "CampaignWideIndex.xlsx")

    print("A. lookup-table integrity")
    for k, d in tabs.items():
        issues = {c: (int(d[c].isna().sum()), int(d[c].dropna().duplicated().sum()))
                  for c in ("ProductID", "ProductIndex", "NewProductIndex", "Name") if c in d}
        print(f"   v{k}: {len(d)} rows; (missing, duplicate) per column {issues}")
    b6 = v6.set_index("ProductID")
    for k in (1, 2, 3, 4, 5):
        d = tabs[k].set_index("ProductID")
        c = d.index.intersection(b6.index)
        n_pi = int((d.loc[c, "ProductIndex"] != b6.loc[c, "ProductIndex"]).sum())
        n_npi = int((d.loc[c, "NewProductIndex"] != b6.loc[c, "NewProductIndex"]).sum()) if "NewProductIndex" in d else "-"
        print(f"   v{k} vs v6 on {len(c)} shared products: ProductIndex differs {n_pi}, NewProductIndex differs {n_npi}")

    print("B. limited-time ids 18-56 vs the feature table")
    fname = dict(zip(feat.NewProductIndex6.astype(int), feat.Name.astype(str)))
    g = v6.groupby("NewProductIndex6").Name.apply(list)
    bad = [i for i in range(18, 57) if len(g.get(i, [])) != 1 or fname.get(i) != str(g[i][0])]
    print(f"   ids failing: {bad if bad else 'none'}")

    print("C. CampaignWideIndex names vs *Index columns")
    pidx = dict(zip(v6.ProductID, v6.ProductIndex))
    for col in OFFER_COLS:
        wrong = [int(cid) for cid, n, i in zip(camp.CampaignID, camp[col], camp[col + "Index"])
                 if n != "<PAD>" and pidx.get(n) != i]
        print(f"   {col}: campaigns whose index disagrees with the name: {wrong if wrong else 'none'}")

    lv6B = dict(zip(v6.ProductIndex.astype(int), v6.NewProductIndex6.astype(int)))
    lv6 = dict(zip(v6.NewProductIndex.astype(int), v6.NewProductIndex6.astype(int)))
    correct = dict(zip(v6.ProductID, v6.NewProductIndex6.astype(int)))
    receivers = {lv6[int(n)] for p, n in zip(v2.ProductID, v2.NewProductIndex)
                 if str(p).startswith("W3") and 18 <= lv6.get(int(n), 0) <= 56}
    miscoded = sum(lv6.get(int(n)) != correct.get(p) for p, n in zip(v2.ProductID, v2.NewProductIndex))
    exp = {int(r.CampaignID): [lv6B.get(int(r[c + "Index"]), 0) for c in OFFER_COLS] for _, r in camp.iterrows()}

    root = os.environ.get("PRODUCTGPT_DATA")
    if not root:
        sys.exit("PRODUCTGPT_DATA is not set")
    recs = load_json_dataset(str(Path(root) / a.data_file))
    offers_ok = offers_n = 0
    over59 = total = 0
    lto_ids = Counter()
    for rec in recs:
        ai = parse_token_ids(rec["AggregateInput"])
        cp = parse_token_ids(rec.get("CampaignID", []))
        for t in range(len(ai) // R):
            if t < len(cp) and cp[t] in exp:
                offers_n += 1
                offers_ok += ai[t * R:t * R + LTO_W] == exp[cp[t]]
            for x in ai[t * R + LTO_W:t * R + LTO_W + OBT_W]:
                if x:
                    total += 1
                    over59 += x > 59
                    if 18 <= x <= 56:
                        lto_ids[x] += 1
    print(f"D. offers in {a.data_file}: "
          + (f"{offers_ok / offers_n:.4f} of {offers_n:,} rows match CampaignWideIndex" if offers_n else "no CampaignID field"))
    n_lto = sum(lto_ids.values())
    share = sum(lto_ids[i] for i in receivers) / max(n_lto, 1)
    print(f"E. obtained stream: {total:,} tokens; values > 59: {over59:,}")
    print(f"   version-2 codes read through version 6 would mis-code {miscoded} of {len(v2)} products and send 3-star "
          f"weapons to ids {sorted(receivers)}")
    print(f"   share of limited-time-id tokens on those ids: {share:.3f} "
          f"(about {len(receivers) / 39:.3f} if acquisitions were spread evenly)")
    verdict = ("VERSION-2 CODES DECODED WITH VERSION 6 -- obtained stream corrupted"
               if over59 == 0 and share > 0.5 else "signature absent -- obtained stream consistent")
    print(f"   verdict: {verdict}")


if __name__ == "__main__":
    main()
