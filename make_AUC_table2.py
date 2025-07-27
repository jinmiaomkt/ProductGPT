#!/usr/bin/env python3
# =============================================================
#  make_AUC_table.py  –  ROC‑AUC, Hit‑rate, F1, AUPRC tables
#                       + aggregate macro metrics
# =============================================================
from __future__ import annotations
import json, gzip, re, os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ------------------------------------------------------------------
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/GRU/gru_predictions.jsonl')
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/LSTM/lstm_predictions.jsonl')
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedFull/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_predictions.jsonl')
LABEL_PATH = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json')
SEED       = 33
open_pred  = gzip.open if PRED_PATH.suffix == ".gz" else open
# ------------------------------------------------------------------

# ---------------------- helpers ------------------------------------
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

# ---------------------- 1. labels ----------------------------------
raw = json.loads(LABEL_PATH.read_text())
records = list(raw) if isinstance(raw, list) else [
    {k: raw[k][i] for k in raw} for i in range(len(raw["uid"]))
]
label_dict = {
    flat_uid(rec["uid"]): {
        "label" : to_int_vec(rec["Decision"]),
        "idx_h" : to_int_vec(rec["IndexBasedHoldout"]),
        "feat_h": to_int_vec(rec["FeatureBasedHoldout"]),
    } for rec in records
}

# -------- 1b. reproduce train/val/test split -----------------------
g = torch.Generator().manual_seed(SEED)
n      = len(records)
tr, va = int(0.8*n), int(0.1*n)
tr_i, va_i, te_i = random_split(range(n), [tr, va, n-tr-va], generator=g)
val_uid  = {flat_uid(records[i]["uid"]) for i in va_i.indices}
test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}
which_split = lambda u: "val" if u in val_uid else "test" if u in test_uid else "train"

# ---------------------- 2. tasks -----------------------------------
BIN_TASKS = {
    "BuyNone":   [9],
    "BuyOne":    [1, 3, 5, 7],
    "BuyTen":    [2, 4, 6, 8],
    "BuyRegular":[1, 2],
    "BuyFigure": [3, 4, 5, 6],
    "BuyWeapon": [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}

def period_group(idx_h, feat_h):
    if feat_h == 0:               return "Calibration"
    if feat_h == 1 and idx_h == 0:return "HoldoutA"
    if idx_h == 1:                return "HoldoutB"
    return "UNASSIGNED"

# ---------------------- 3. stream predictions ----------------------
scores       = defaultdict(lambda: {"y": [], "p": []})
length_note  = Counter()
accept = reject = 0
brace_re  = re.compile(r"\{.*\}")

with open_pred(PRED_PATH, "rt", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line or line[0] not in "{[":
            reject += 1               # skip log/empty lines
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            m = brace_re.search(line)
            if not m:
                reject += 1
                continue
            try:
                rec = json.loads(m.group())
            except json.JSONDecodeError:
                reject += 1
                continue

        uid = flat_uid(rec.get("uid", ""))
        lbl_info = label_dict.get(uid)
        if lbl_info is None:
            reject += 1
            continue

        probs_field = rec["probs"]
        preds = probs_field if isinstance(probs_field[0], list) else [probs_field]
        L = min(len(preds), len(lbl_info["label"]))
        if len(preds) != len(lbl_info["label"]):
            length_note["length_mismatch"] += 1

        split_tag = which_split(uid)
        for t in range(L):
            y      = lbl_info["label"][t]
            idx_h  = lbl_info["idx_h"][t]
            feat_h = lbl_info["feat_h"][t]
            probs  = preds[t]

            group = period_group(idx_h, feat_h)
            for task, pos in BIN_TASKS.items():
                y_bin = int(y in TASK_POSSETS[task])
                p_bin = sum(probs[i-1] for i in pos)  # 1‑indexed
                key   = (task, group, split_tag)
                scores[key]["y"].append(y_bin)
                scores[key]["p"].append(p_bin)

        accept += 1

print(f"[INFO] parsed: {accept} users accepted, {reject} lines skipped.")
if length_note:
    print("[INFO] notes:", dict(length_note))

# ---------------------- 4. compute metrics --------------------------
rows = []
for task in BIN_TASKS:
    for grp in ["Calibration","HoldoutA","HoldoutB"]:
        for spl in ["val","test"]:
            y, p = scores[(task, grp, spl)]["y"], scores[(task, grp, spl)]["p"]
            if not y:
                continue
            auc   = np.nan if len(set(y)) < 2 else roc_auc_score(y, p)
            y_hat = [int(prob >= 0.5) for prob in p]
            acc   = accuracy_score(y, y_hat) if len(set(y)) > 1 else np.nan
            f1    = f1_score(y, y_hat)        if len(set(y)) > 1 else np.nan
            auprc = average_precision_score(y, p) if len(set(y)) > 1 else np.nan

            rows.append(
                {"Task": task, "Group": grp, "Split": spl,
                 "AUC": auc, "Hit": acc, "F1": f1, "AUPRC": auprc}
            )

metrics = pd.DataFrame(rows)

def pivot(metric: str) -> pd.DataFrame:
    return (metrics
            .pivot(index=["Task","Group"], columns="Split", values=metric)
            .reindex(columns=["val","test"])
            .round(4)
            .sort_index())

auc_tbl   = pivot("AUC")
hit_tbl   = pivot("Hit")
f1_tbl    = pivot("F1")
auprc_tbl = pivot("AUPRC")

# ---------- aggregate macro metrics ---------------------------------
# macro_tbl = (
#     metrics
#       .groupby("Split")[["AUC","Hit","F1","AUPRC"]]
#       .mean()
#       .reindex(["val","test"])
#       .round(4)
#       .rename_axis("Split")
# )

# ---------- aggregate macro metrics  (PER‑PERIOD × PER‑SUBSET) ------
macro_period_tbl = (
    metrics
      .groupby(["Group", "Split"])[["AUC", "Hit", "F1", "AUPRC"]]
      .mean()                       # avg across tasks
      .unstack("Split")             # cols: metric × split
      .round(4)
)

# put the split level in the desired order (val, test)
macro_period_tbl = macro_period_tbl.reorder_levels([1, 0], axis=1)  # split → outer, metric → inner
macro_period_tbl = macro_period_tbl.sort_index(axis=1, level=0)     # ensure val before test
macro_period_tbl = macro_period_tbl[['val', 'test']]                # final order


# ---------------------- 5. print tables -----------------------------
print("\n=============  BINARY ROC‑AUC TABLE  =======================")
print(auc_tbl.fillna(" NA"))
print("=============  HIT‑RATE (ACCURACY) TABLE  ==================")
print(hit_tbl.fillna(" NA"))
print("=============  MACRO‑F1 TABLE  =============================")
print(f1_tbl.fillna(" NA"))
print("=============  AUPRC TABLE  ================================")
print(auprc_tbl.fillna(" NA"))
print("=============  AGGREGATE MACRO METRICS  ====================")
print(macro_period_tbl.fillna(" NA"))
print("============================================================")
