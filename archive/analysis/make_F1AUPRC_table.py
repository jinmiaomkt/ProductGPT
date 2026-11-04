#!/usr/bin/env python3
# =============================================================
#  make_F1AUPRC_table.py  –  robust F1 & AUPRC for 6 buckets
# =============================================================
from __future__ import annotations
import json, gzip, re, os, itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 0. EDIT FILE LOCATIONS
# ------------------------------------------------------------------
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedFull/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_predictions.jsonl')
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/LSTM/lstm_predictions.jsonl')
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/GRU/gru_predictions.jsonl')
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedLP/LP_feature_predictions.jsonl')
LABEL_PATH = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json')
SEED       = 33
# ------------------------------------------------------------------

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid MKL/OpenBLAS clash on macOS

# -------------------- helpers --------------------------------------
def to_int_vec(x):
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(int(v) if isinstance(item, str) else item for v in str(item).split())
        return out
    raise TypeError(type(x))

flat_uid = lambda u: str(u[0] if isinstance(u, list) else u)

# -------------------- 1. load labels -------------------------------
raw = json.loads(LABEL_PATH.read_text())
records = list(raw) if isinstance(raw, list) else [
    {k: raw[k][i] for k in raw} for i in range(len(raw["uid"]))
]

def explode_record(rec):
    uid   = flat_uid(rec["uid"])
    y     = to_int_vec(rec["Decision"])
    idx_h = to_int_vec(rec["IndexBasedHoldout"])
    feat_h= to_int_vec(rec["FeatureBasedHoldout"])
    for t in range(len(y)):
        yield {
            "uid": uid, "t": t,
            "label": y[t],
            "idx_h": idx_h[t],
            "feat_h": feat_h[t],
        }

label_df = pd.DataFrame(itertools.chain.from_iterable(explode_record(r) for r in records))

# -------------------- 2. load predictions robustly -----------------
brace_re  = re.compile(r"\{.*\}")
open_pred = gzip.open if PRED_PATH.suffix == ".gz" else open

pred_rows, length_note = [], Counter()
with open_pred(PRED_PATH, "rt", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line or line[0] not in "{[":
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            m = brace_re.search(line)
            if not m:
                continue
            try:
                rec = json.loads(m.group())
            except json.JSONDecodeError:
                continue

        uid        = flat_uid(rec.get("uid", ""))
        probs_data = rec["probs"]

        # old = list‑of‑lists, new = single list
        preds = probs_data if isinstance(probs_data[0], list) else [probs_data]
        L_pred = len(preds)

        # we don't know label length yet → store whole list; trim after merge
        for t, vec in enumerate(preds):
            pred_rows.append({"uid": uid, "t": t, **{f"p{i+1}": vec[i] for i in range(9)}})

pred_df = pd.DataFrame(pred_rows)

# -------------------- 3. merge & align lengths ---------------------
data = pred_df.merge(label_df, on=["uid", "t"], how="inner")

# Rebuild 80‑10‑10 split on uids
all_uids = [flat_uid(r["uid"]) for r in records]
train_u, temp_u = train_test_split(all_uids, test_size=0.2, random_state=SEED)
val_u,  test_u  = train_test_split(temp_u,  test_size=0.5, random_state=SEED)
data["Split"] = data.uid.apply(lambda u: "train" if u in train_u else "val" if u in val_u else "test")

data["Group"] = np.select(
    [
        data.feat_h == 0,
        (data.feat_h == 1) & (data.idx_h == 0),
        data.idx_h == 1,
    ],
    ["Calibration", "HoldoutA", "HoldoutB"],
    default="UNASSIGNED"
)

# -------------------- 4. define binary tasks -----------------------
BIN_TASKS = {
    "BuyNone"   : [9],
    "BuyOne"    : [1, 3, 5, 7],
    "BuyTen"    : [2, 4, 6, 8],
    "BuyRegular": [1, 2],
    "BuyFigure" : [3, 4, 5, 6],
    "BuyWeapon" : [7, 8],
}

for task, pos in BIN_TASKS.items():
    data[f"y_{task}"] = data.label.isin(pos).astype(int)
    data[f"p_{task}"] = data[[f"p{i}" for i in pos]].sum(axis=1)

# -------------------- 5. compute F1 & AUPRC ------------------------
rows = []
for task in BIN_TASKS:
    for grp in ["Calibration", "HoldoutA", "HoldoutB"]:
        for spl in ["val", "test"]:
            sdf = data[(data.Group == grp) & (data.Split == spl)]
            if sdf.empty or sdf[f"y_{task}"].nunique() < 2:
                continue
            y_true = sdf[f"y_{task}"].values
            y_prob = sdf[f"p_{task}"].values
            y_pred = (y_prob > 0.5).astype(int)

            rows.append({
                "Metric": task,
                "Group": grp,
                "Split": spl,
                "F1":     round(f1_score(y_true, y_pred), 4),
                "AUPRC":  round(average_precision_score(y_true, y_prob), 4),
            })

df_out = (
    pd.DataFrame(rows)
      .pivot(index=["Metric","Group"], columns="Split", values=["F1","AUPRC"])
      .round(4)
)

print("\n====== F1 Score & AUPRC (6 binary buckets) ======")
print(df_out.fillna("—"))
print("===============================================")
