#!/usr/bin/env python3
# =============================================================
#  make_AUC_table.py  –  tolerant ROC‑AUC table builder
# =============================================================
from __future__ import annotations
import json, gzip, warnings, os, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score

# Uncomment next two lines once; silences OpenMP clash crash
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------------
# 0. EDIT YOUR FILE LOCATIONS
# ------------------------------------------------------------------
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedLP/LP_feature_predictions.jsonl')
LABEL_PATH = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json")
SEED       = 33
# ------------------------------------------------------------------


# ---------------------- helper: robust int‑vector -----------------
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
    return str(u[0] if isinstance(u, list) else u)


# ---------------------- 1. load labels ---------------------------
raw_json = json.loads(LABEL_PATH.read_text())

raw_records = (
    list(raw_json)
    if isinstance(raw_json, list)
    else [{k: raw_json[k][i] for k in raw_json} for i in range(len(raw_json["uid"]))]
)

label_dict = {}
for rec in raw_records:
    uid = flat_uid(rec["uid"])
    label_dict[uid] = {
        "label":  to_int_vec(rec["Decision"]),
        "idx_h":  to_int_vec(rec["IndexBasedHoldout"]),
        "feat_h": to_int_vec(rec["FeatureBasedHoldout"]),
    }

# ------------- 1b. rebuild 80‑10‑10 split ------------------------
train_sz = int(0.8 * len(raw_records))
val_sz   = int(0.1 * len(raw_records))
test_sz  = len(raw_records) - train_sz - val_sz
g        = torch.Generator().manual_seed(SEED)
tr_i, va_i, te_i = random_split(range(len(raw_records)),
                                [train_sz, val_sz, test_sz], generator=g)
val_uid  = {flat_uid(raw_records[i]["uid"]) for i in va_i.indices}
test_uid = {flat_uid(raw_records[i]["uid"]) for i in te_i.indices}

which_split = lambda u: "val" if u in val_uid else "test" if u in test_uid else "train"

# ---------------------- 2. task / group defs ---------------------
BIN_TASKS = {
    "BuyNone":   [9],
    "BuyOne":    [1, 3, 5, 7],
    "BuyTen":    [2, 4, 6, 8],
    "BuyRegular": [1, 2],
    "BuyFigure":  [3, 4, 5, 6],
    "BuyWeapon":  [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}

def period_group(idx_h, feat_h):
    if feat_h == 0:
        return "Calibration"
    if feat_h == 1 and idx_h == 0:
        return "HoldoutA"
    if idx_h == 1:
        return "HoldoutB"
    return "UNASSIGNED"


# ---------------------- 3. stream predictions --------------------
scores = defaultdict(lambda: {"y": [], "p": []})
accept, reject = 0, 0                                   # counters

open_pred = gzip.open if PRED_PATH.suffix == ".gz" else open
brace_re  = re.compile(r"\{.*\}")                       # greedy, last '}' keeps most json

with open_pred(PRED_PATH, "rt", errors="replace") as f:
    for n, raw_line in enumerate(f, 1):
        line = raw_line.strip()
        if not line:
            continue
        if line[0] not in "{[":
            # not a JSON line – likely log text
            reject += 1
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # salvage attempt: take substring between first '{' and last '}'
            m = brace_re.search(line)
            if not m:
                warnings.warn(f"Line {n}: no JSON object found – skipped")
                reject += 1
                continue
            try:
                rec = json.loads(m.group())
            except json.JSONDecodeError:
                warnings.warn(f"Line {n}: corrupted JSON – skipped")
                reject += 1
                continue

        uid = flat_uid(rec.get("uid", ""))
        if uid not in label_dict:
            warnings.warn(f"uid {uid} not in labels – skipped")
            reject += 1
            continue

        lbl_info = label_dict[uid]
        split    = which_split(uid)

        probs_field = rec["probs"]
        # old (list‑of‑lists) or new (single list)
        probs_iter = probs_field if probs_field and isinstance(probs_field[0], list) \
                               else [probs_field] * len(lbl_info["label"])

        for t, probs in enumerate(probs_iter):
            try:
                y, idx_h, feat_h = (lbl_info["label"][t],
                                    lbl_info["idx_h"][t],
                                    lbl_info["feat_h"][t])
            except IndexError:
                warnings.warn(f"uid {uid}: shorter labels than preds – truncated")
                break

            group = period_group(idx_h, feat_h)

            for task, pos in BIN_TASKS.items():
                y_bin = int(y in TASK_POSSETS[task])
                p_bin = sum(probs[i - 1] for i in pos)  # 1‑indexed
                key   = (task, group, split)
                scores[key]["y"].append(y_bin)
                scores[key]["p"].append(p_bin)

        accept += 1

print(f"[INFO] Parsed prediction file: {accept} records accepted, {reject} skipped.")

# ---------------------- 4. compute AUC table ---------------------
rows = []
for task in BIN_TASKS:
    for group in ["Calibration", "HoldoutA", "HoldoutB"]:
        for split in ["val", "test"]:
            key = (task, group, split)
            y   = scores[key]["y"]
            p   = scores[key]["p"]
            auc = np.nan if len(set(y)) < 2 else roc_auc_score(y, p)
            rows.append({"Task": task, "Group": group, "Split": split, "AUC": auc})

auc_tbl = (
    pd.DataFrame(rows)
      .pivot(index=["Task", "Group"], columns="Split", values="AUC")
      .round(4)
      .sort_index()
)

print("\n=============  BINARY ROC‑AUC TABLE  =======================")
print(auc_tbl.fillna(" NA"))
print("============================================================")
