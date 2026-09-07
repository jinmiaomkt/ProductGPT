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


def load_feature_tensor(xls_path: str | Path) -> torch.Tensor:
    """
    Build the product-level feature table.

    Returns a (MAX_TOKEN_ID + 1, FEATURE_DIM) float32 tensor, i.e. (60, 34),
    indexed by token ID. Rows for non-product tokens stay all-zero.

    This is the exact behaviour of `load_feature_tensor` in the generation-4
    trainers, which is the tensor actually passed to `build_transformer`.
    """
    df = pd.read_excel(xls_path, sheet_name=0)

    arr = np.zeros((MAX_TOKEN_ID + 1, FEATURE_DIM), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row[PRODUCT_ID_COLUMN])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)

    return torch.from_numpy(arr)
