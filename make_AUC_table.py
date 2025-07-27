#!/usr/bin/env python3
# =============================================================
#  make_AUC_table.py  –  ROC‑AUC table (robust, no spam)
# =============================================================
from __future__ import annotations
import json, gzip, re, os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score

# Optional: silence Intel/LLVM OpenMP clash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ------------------------------------------------------------------
# EDIT YOUR FILE LOCATIONS
# ------------------------------------------------------------------
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/GRU/gru_predictions.jsonl')
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/LSTM/lstm_predictions.jsonl')
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/LSTM/lstm_predictions.jsonl')
LABEL_PATH = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json')
SEED       = 33
open_pred = gzip.open if PRED_PATH.suffix == ".gz" else open
# ------------------------------------------------------------------

# ---------------------- tiny helpers ---------------------------------
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

# ---------------------- 1. load labels -------------------------------
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

# ── peek at a single user ───────────────────────────────────────────
uid0, rec0 = next(iter(label_dict.items()))
print(uid0, rec0["label"][:5], len(rec0["label"]))

with open_pred(PRED_PATH, "rt") as f:
    for line in f:
        if uid0 in line:
            probs = json.loads(line)["probs"][0]
            print("probs len =", len(probs))
            print("arg‑max idx =", np.argmax(probs))
            break

# -------- 1b. 80‑10‑10 split (reproduce training split) --------------
g = torch.Generator().manual_seed(SEED)
n      = len(records)
tr, va = int(0.8*n), int(0.1*n)
tr_i, va_i, te_i = random_split(range(n), [tr, va, n-tr-va], generator=g)
val_uid  = {flat_uid(records[i]["uid"]) for i in va_i.indices}
test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}
which_split = lambda u: "val" if u in val_uid else "test" if u in test_uid else "train"

# ---------------------- 2. task definitions --------------------------
BIN_TASKS = {
    "BuyNone":   [9],
    "BuyOne":    [1, 3, 5, 7],
    "BuyTen":    [2, 4, 6, 8],
    "BuyRegular":[1, 2],
    "BuyFigure": [3, 4, 5, 6],
    "BuyWeapon": [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}

# ------------------------------------------------
#  C. drop the i‑1 correction when you sum probs
# ------------------------------------------------
def period_group(idx_h, feat_h):
    if feat_h == 0:               return "Calibration"
    if feat_h == 1 and idx_h == 0:return "HoldoutA"
    if idx_h == 1:                return "HoldoutB"
    return "UNASSIGNED"

# ---------------------- 3. stream predictions ------------------------
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

        # ------------- align lengths ---------------------------------
        probs_field = rec["probs"]
        # old format = list‑of‑lists, new = single list
        preds = probs_field if isinstance(probs_field[0], list) else [probs_field]
        L_pred, L_lbl = len(preds), len(lbl_info["label"])
        if L_pred != L_lbl:
            length_note["pred>lbl" if L_pred > L_lbl else "pred<label"] += 1
        L = min(L_pred, L_lbl)

        split = which_split(uid)
        for t in range(L):
            y      = lbl_info["label"][t]
            idx_h  = lbl_info["idx_h"][t]
            feat_h = lbl_info["feat_h"][t]
            probs  = preds[t]

            group = period_group(idx_h, feat_h)
            for task, pos in BIN_TASKS.items():
                y_bin = int(y in TASK_POSSETS[task])
                p_bin = sum(probs[i-1] for i in pos)  # 1‑indexed
                key   = (task, group, split)
                scores[key]["y"].append(y_bin)
                scores[key]["p"].append(p_bin)

        accept += 1

print(f"[INFO] parsed: {accept} users accepted, {reject} lines skipped.")
if length_note:
    print("[INFO] length mismatches:", dict(length_note))

# ---------------------- 4. compute AUC table -------------------------
rows = []
for task in BIN_TASKS:
    for grp in ["Calibration","HoldoutA","HoldoutB"]:
        for spl in ["val","test"]:
            key = (task, grp, spl)
            y, p = scores[key]["y"], scores[key]["p"]
            auc  = np.nan if len(set(y)) < 2 else roc_auc_score(y, p)
            rows.append({"Task": task, "Group": grp, "Split": spl, "AUC": auc})

auc_tbl = (
    pd.DataFrame(rows)
      .pivot(index=["Task","Group"], columns="Split", values="AUC")
      .round(4)
      .sort_index()
)

print("\n=============  BINARY ROC‑AUC TABLE  =======================")
print(auc_tbl.fillna(" NA"))
print("============================================================")
