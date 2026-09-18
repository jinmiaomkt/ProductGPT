"""
Build the per-product vocabulary and feature table for R28.

Today every 3-star weapon shares one token, as do the 4-stars and the standard
5-stars: 44 product tokens for 118 distinct products. R27 showed that token
resolution matters (correcting the ids cut the obtained stream's entropy from
1.050 to 0.601 nats and cost every model about 0.02 nats), so R28 gives every
product its own token and its own embedding.

Layout (`vocab7`), chosen to keep the special tokens where the code expects
them and to make the product block contiguous:

    0        [PAD]
    1-9      decisions
    10       [SOS] decisions      11  [EOS] decisions     12  [UNK]
    13..     products, ordered by ProductID (F4-*, F5-*, W3-*, W4-*, W5-*)
    last 3   [EOS] products, [SOS-clean], [SOS-unclean]

Only products that the data can actually encode are included: the 118 real
products in FigureWeaponIndex2.xlsx, the table the per-user files were written
with. F5-014 and F5-036 (added in version 3) cannot appear and are excluded.

Outputs, written next to the data (never into OneDrive):
    <PRODUCTGPT_DATA>/perproduct/ProductVocab7.xlsx
        ProductID, Name, NewProductIndex7, v2 code, plus the 34 FEATURE_COLS
    <PRODUCTGPT_DATA>/perproduct/vocab7_map.csv
        v2 code -> NewProductIndex7, consumed by the R generator

Feature columns come from FullFigureWeaponEmbeddingIndex.xlsx, which covers all
177 products. Three columns are derived, because that sheet names them
differently or does not carry them:
    GenderFemale/GenderMale <- EthnicityFemale/EthnicityMale
    type_figure             <- type == "figure"
    LTO                     <- product appears in an offer slot of CampaignWideIndex

USAGE
    python scripts/build_product_vocab7.py --meta-dir <Data folder>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.features import FEATURE_COLS  # noqa: E402

FIRST_PROD_ID7 = 13
OFFER_COLS = ("Figure5AIndex", "Figure5BIndex", "Weapon5AIndex", "Weapon5BIndex")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: $PRODUCTGPT_DATA/perproduct")
    a = ap.parse_args()
    meta = Path(a.meta_dir)
    out = Path(a.out_dir or (Path(os.environ["PRODUCTGPT_DATA"]) / "perproduct"))
    out.mkdir(parents=True, exist_ok=True)

    v2 = pd.read_excel(meta / "FigureWeaponIndex2.xlsx")
    full = pd.read_excel(meta / "FullFigureWeaponEmbeddingIndex.xlsx")
    camp = pd.read_excel(meta / "CampaignWideIndex.xlsx")
    v6 = pd.read_excel(meta / "FigureWeaponIndex6.xlsx")

    special = {"<PAD>", "<EOS>", "<SOS-clean>", "<SOS-unclean>"}
    real = v2[~v2.ProductID.isin(special)].copy()
    real = real.sort_values("ProductID").reset_index(drop=True)
    real["NewProductIndex7"] = range(FIRST_PROD_ID7, FIRST_PROD_ID7 + len(real))
    last_prod = int(real.NewProductIndex7.max())
    eos, sos_clean, sos_unclean = last_prod + 1, last_prod + 2, last_prod + 3

    missing = set(real.ProductID) - set(full.ProductID)
    if missing:
        sys.exit(f"features missing for {len(missing)} products: {sorted(missing)[:5]}")

    f = full.set_index("ProductID")
    lto_pids = set()
    pid_by_index = dict(zip(v6.ProductIndex, v6.ProductID))
    for c in OFFER_COLS:
        for i in camp[c].dropna():
            pid = pid_by_index.get(int(i))
            if pid is not None:
                lto_pids.add(pid)

    rows = []
    for _, r in real.iterrows():
        src = f.loc[r.ProductID]
        rec = {"ProductID": r.ProductID, "Name": r.Name,
               "NewProductIndex7": int(r.NewProductIndex7),
               "v2_code": int(r.NewProductIndex),
               "NewProductIndex6": int(v6.loc[v6.ProductID == r.ProductID, "NewProductIndex6"].iloc[0])}
        for col in FEATURE_COLS:
            if col == "GenderFemale":
                rec[col] = src["EthnicityFemale"]
            elif col == "GenderMale":
                rec[col] = src["EthnicityMale"]
            elif col == "type_figure":
                rec[col] = int(str(src["type"]).strip().lower() == "figure")
            elif col == "LTO":
                rec[col] = int(r.ProductID in lto_pids)
            elif col == "CountryRuiYue":
                rec[col] = src["CountryRuiYue"] if "CountryRuiYue" in src else src["CountryLiYue"]
            else:
                rec[col] = src[col]
        rows.append(rec)
    tab = pd.DataFrame(rows)

    tab.to_excel(out / "ProductVocab7.xlsx", index=False)
    pd.DataFrame({"v2_code": tab.v2_code, "NewProductIndex7": tab.NewProductIndex7}).to_csv(
        out / "vocab7_map.csv", index=False)

    grp = tab.assign(g=tab.ProductID.str[:2]).groupby("g").size().to_dict()
    print(f"products: {len(tab)}  ids {FIRST_PROD_ID7}-{last_prod}  by group {grp}")
    print(f"specials: EOS={eos} SOS-clean={sos_clean} SOS-unclean={sos_unclean}; vocab size {sos_unclean + 1}")
    print(f"limited-time products (LTO=1): {int(tab.LTO.sum())}; 5-star: {int((tab.Rarity == 5).sum())}")
    nonnum = tab[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").isna().sum()
    print("non-numeric feature cells:", {k: int(v) for k, v in nonnum.items() if v} or "none")
    print("wrote", out / "ProductVocab7.xlsx", "and", out / "vocab7_map.csv")


if __name__ == "__main__":
    main()
