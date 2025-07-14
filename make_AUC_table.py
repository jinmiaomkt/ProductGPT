#!/usr/bin/env python3
# =============================================================
#  make_AUC_table.py  –  AUC table + ROC curves
# =============================================================
import json, gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
import torch
import pandas as pd


# ------------------------------------------------------------------
# 0. EDIT YOUR FILE LOCATIONS
# ------------------------------------------------------------------
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedLP/LP_feature_predictions.jsonl')
LABEL_PATH = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json")
SEED       = 33
# ------------------------------------------------------------------

# ---------------------- helper: robust int-vector -----------------
def to_int_vec(x):
    """Return list[int] from str, list[str], list[int], or ['1 2 3']."""
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                out.extend(int(v) for v in item.split())
            else:
                raise TypeError(f"Unsupported inner type: {type(item)}")
        return out
    raise TypeError(f"Unsupported field type: {type(x)}")

def flat_uid(u):
    """uid may be 'abc' or ['abc']; always return str."""
    return str(u[0] if isinstance(u, list) else u)

# ---------------------- 1. load predictions -----------------------
pred_rows = []
with PRED_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        uid = flat_uid(rec["uid"])
        
        for t, p in enumerate(rec["probs"]):          # 9-way probs
            pred_rows.append(
                {"uid": uid, "t": t, **{f"p{i+1}": p[i] for i in range(9)}})
pred_df = pd.DataFrame(pred_rows)

# ---------------------- 2. load labels & explode ------------------
raw_json = json.loads(LABEL_PATH.read_text())

# raw_records: list‑of‑dicts (one per uid)
raw_records = (
    list(raw_json)
    if isinstance(raw_json, list)
    else [{k: raw_json[k][i] for k in raw_json} for i in range(len(raw_json["uid"]))]
)

# label_dict  uid -> dict of lists {label[], idx_h[], feat_h[]}
label_dict = {}
for rec in raw_records:
    uid = flat_uid(rec["uid"])
    label_dict[uid] = {
        "label"  : to_int_vec(rec["Decision"]),
        "idx_h"  : to_int_vec(rec["IndexBasedHoldout"]),
        "feat_h" : to_int_vec(rec["FeatureBasedHoldout"]),
    }

# ---------------------- 2. rebuild 80‑10‑10 split ---------------
train_sz   = int(0.8 * len(raw_records))
val_sz     = int(0.1 * len(raw_records))
test_sz    = len(raw_records) - train_sz - val_sz
g = torch.Generator().manual_seed(SEED)
train_idx, val_idx, test_idx = random_split(
    range(len(raw_records)), [train_sz, val_sz, test_sz], generator=g
)
train_uid = {flat_uid(raw_records[i]["uid"]) for i in train_idx.indices}
val_uid   = {flat_uid(raw_records[i]["uid"]) for i in val_idx.indices}
test_uid  = {flat_uid(raw_records[i]["uid"]) for i in test_idx.indices}

def which_split(uid):
    if uid in val_uid:
        return "val"
    if uid in test_uid:
        return "test"
    return "train"

# ---------------------- 3. bucket definitions --------------------
BIN_TASKS = {
    "BuyNone"   : [9],
    "BuyOne"    : [1, 3, 5, 7],
    "BuyTen"    : [2, 4, 6, 8],
    "BuyRegular": [1, 2],
    "BuyFigure" : [3, 4, 5, 6],
    "BuyWeapon" : [7, 8],
}
# fast membership lookup
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}

# ---------------------- 4. accumulators --------------------------
# scores[(task, group, split)] -> {"y":[], "p":[]}
scores = defaultdict(lambda: {"y": [], "p": []})

def period_group(idx_h, feat_h):
    if feat_h == 0:
        return "Calibration"
    if feat_h == 1 and idx_h == 0:
        return "HoldoutA"
    if idx_h == 1:
        return "HoldoutB"
    return "UNASSIGNED"

# ---------------------- 5. stream predictions --------------------
open_pred = gzip.open if PRED_PATH.suffix == ".gz" else open
with open_pred(PRED_PATH, "rt") as f:
    for line in f:
        rec = json.loads(line)
        uid = flat_uid(rec["uid"])
        if uid not in label_dict:
            continue  # skip if uid missing in labels

        lbl_info = label_dict[uid]
        split    = which_split(uid)

        for t, probs in enumerate(rec["probs"]):  # probs is length‑9 list
            try:
                y      = lbl_info["label"][t]
                idx_h  = lbl_info["idx_h"][t]
                feat_h = lbl_info["feat_h"][t]
            except IndexError:
                continue  # mismatch in sequence length → ignore

            group = period_group(idx_h, feat_h)

            for task, pos in BIN_TASKS.items():
                y_bin = int(y in TASK_POSSETS[task])
                p_bin = sum(probs[i - 1] for i in pos)  # probs are 1‑indexed
                key   = (task, group, split)
                scores[key]["y"].append(y_bin)
                scores[key]["p"].append(p_bin)

# ---------------------- 6. compute AUC table ---------------------
rows = []
for task in BIN_TASKS:
    for group in ["Calibration", "HoldoutA", "HoldoutB"]:
        for split in ["val", "test"]:
            key = (task, group, split)
            y   = scores[key]["y"]
            p   = scores[key]["p"]
            auc_val = (
                np.nan
                if len(set(y)) < 2
                else roc_auc_score(y, p)
            )
            rows.append(
                {"Task": task, "Group": group, "Split": split, "AUC": auc_val}
            )

auc_tbl = (
    pd.DataFrame(rows)
    .pivot(index=["Task", "Group"], columns="Split", values="AUC")
    .round(4)
    .sort_index()
)

print("\n=============  BINARY ROC‑AUC TABLE  =======================")
print(auc_tbl.fillna(" NA"))
print("============================================================")