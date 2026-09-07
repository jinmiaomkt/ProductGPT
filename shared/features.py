"""
Product attribute features.

IMPORTANT CHANGE FROM THE OLD MODEL FILES
-----------------------------------------
The model modules used to run

    df = pd.read_excel("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")

at *import time*, at module top level. That made `import model4_...` fail on any
machine that is not the original EC2 box — including this Windows laptop and the
SMU HPCC login nodes.

Here the spreadsheet is only read when you explicitly call `load_feature_tensor(path)`.
Importing this module touches no disk and needs no data.

The trainers already built their feature tensor this way and passed it into
`build_transformer(feature_tensor=...)`, so nothing changes at runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .vocab import FIRST_PROD_ID, LAST_PROD_ID, MAX_TOKEN_ID

# Column in the spreadsheet holding the product's token ID.
PRODUCT_ID_COLUMN = "NewProductIndex6"

# The 34 attribute columns, in the order the model expects them.
# Columns commented out in the original source (EthnicityGrass, CountryFengDan,
# CountryXuMi, NewProductIndex3) are intentionally excluded.
FEATURE_COLS: list[str] = [
    # base stats
    "Rarity",
    "MaxLife",
    "MaxOffense",
    "MaxDefense",
    # weapon type (one-hot)
    "WeaponTypeOneHandSword",
    "WeaponTypeTwoHandSword",
    "WeaponTypeArrow",
    "WeaponTypeMagic",
    "WeaponTypePolearm",
    # ethnicity (one-hot)
    "EthnicityIce",
    "EthnicityRock",
    "EthnicityWater",
    "EthnicityFire",
    "EthnicityThunder",
    "EthnicityWind",
    # gender (one-hot)
    "GenderFemale",
    "GenderMale",
    # country (one-hot)
    "CountryRuiYue",
    "CountryDaoQi",
    "CountryZhiDong",
    "CountryMengDe",
    # misc
    "type_figure",
    "MinimumAttack",
    "MaximumAttack",
    "MinSpecialEffect",
    "MaxSpecialEffect",
    "SpecialEffectEfficiency",
    "SpecialEffectExpertise",
    "SpecialEffectAttack",
    "SpecialEffectSuper",
    "SpecialEffectRatio",
    "SpecialEffectPhysical",
    "SpecialEffectLife",
    "LTO",
]

FEATURE_DIM = len(FEATURE_COLS)  # 34

# Known column renames between spreadsheet versions.
#
# The copy of SelectedFigureWeaponEmbeddingIndex.xlsx on this machine names the
# fourth country one-hot "CountryLiYue"; the code has always asked for
# "CountryRuiYue". They are the same slot: the sheet carries exactly four
# Country* one-hots (LiYue, DaoQi, ZhiDong, MengDe) and FEATURE_COLS expects
# exactly four (RuiYue, DaoQi, ZhiDong, MengDe), with three matching verbatim.
# "Liyue" is also the actual in-game region name, so RuiYue looks like the typo.
#
# Aliases are applied only when the canonical name is absent, and every
# substitution is printed -- a silent rename here would quietly change which
# column feeds the model.
COLUMN_ALIASES: dict[str, list[str]] = {
    "CountryRuiYue": ["CountryLiYue"],
}


def _resolve_columns(df_columns, wanted: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Map wanted column names onto what the sheet actually has."""
    have = set(df_columns)
    resolved: list[str] = []
    substitutions: list[tuple[str, str]] = []
    missing: list[str] = []
    for col in wanted:
        if col in have:
            resolved.append(col)
            continue
        alt = next((a for a in COLUMN_ALIASES.get(col, []) if a in have), None)
        if alt is not None:
            resolved.append(alt)
            substitutions.append((col, alt))
        else:
            missing.append(col)
    if missing:
        raise KeyError(
            f"Feature spreadsheet is missing required column(s): {missing}. "
            f"Present columns: {sorted(have)}"
        )
    return resolved, substitutions


def load_feature_tensor(xls_path: str | Path) -> torch.Tensor:
    """
    Build the product-level feature table.

    Returns a (MAX_TOKEN_ID + 1, FEATURE_DIM) float32 tensor, i.e. (60, 34),
    indexed by token ID. Rows for non-product tokens stay all-zero.

    This is the exact behaviour of `load_feature_tensor` in the generation-4
    trainers, which is the tensor actually passed to `build_transformer`.
    """
    df = pd.read_excel(xls_path, sheet_name=0)

    cols, subs = _resolve_columns(df.columns, FEATURE_COLS)
    for canonical, actual in subs:
        print(f"[features] column alias: {canonical!r} not found, using {actual!r}")

    # Some cells in the spreadsheet are not numeric (the local copy has one
    # stray text value in MinSpecialEffect). The model cannot consume a string,
    # so coerce to NaN and zero-fill -- but say so loudly and per column, since
    # this silently alters a product's feature vector.
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    dirty = (numeric.isna() & df[cols].notna()).sum()
    for col, n in dirty.items():
        if n:
            print(f"[features] WARNING: {col!r} has {int(n)} non-numeric cell(s); "
                  f"coerced to 0.0 -- fix the spreadsheet for production runs")
    numeric = numeric.fillna(0.0)

    arr = np.zeros((MAX_TOKEN_ID + 1, FEATURE_DIM), dtype=np.float32)
    for i, row_id in enumerate(df[PRODUCT_ID_COLUMN]):
        token_id = int(row_id)
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = numeric.iloc[i].to_numpy(dtype=np.float32)

    return torch.from_numpy(arr)
