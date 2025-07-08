#!/usr/bin/env python3
# =============================================================
#  Calculate F1 Score and AUPRC for 6 Binary Metrics
# =============================================================
import json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split

# --- FILE PATHS ---
PRED_PATH  = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/productgpt_predictions.jsonl")
LABEL_PATH = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json")
SEED = 33

# --- HELPER FUNCTIONS ---
def to_int_vec(x):
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                out.extend(int(v) for v in item.split())
        return out
    raise TypeError(f"Unsupported type: {type(x)}")

def flat_uid(u): return str(u[0] if isinstance(u, list) else u)

# --- LOAD PREDICTIONS ---
pred_rows = []
with PRED_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        uid = flat_uid(rec["uid"])
        for t, p in enumerate(rec["probs"]):
            pred_rows.append({"uid": uid, "t": t, **{f"p{i+1}": p[i] for i in range(9)}})
pred_df = pd.DataFrame(pred_rows)

# --- LOAD LABELS ---
raw_json = json.loads(LABEL_PATH.read_text())

def explode_record(rec):
    uid = flat_uid(rec["uid"])
    y      = to_int_vec(rec["Decision"])
    idx_h  = to_int_vec(rec["IndexBasedHoldout"])
    feat_h = to_int_vec(rec["FeatureBasedHoldout"])
    for t in range(len(y)):
        yield {
            "uid": uid, "t": t,
            "label": y[t],
            "IndexBasedHoldout": idx_h[t],
            "FeatureBasedHoldout": feat_h[t]
        }

if isinstance(raw_json, dict):
    n_rows = len(raw_json["uid"])
    raw_iter = ({k: raw_json[k][i] for k in raw_json} for i in range(n_rows))
else:
    raw_iter = raw_json

label_rows = list(itertools.chain.from_iterable(explode_record(r) for r in raw_iter))
label_df = pd.DataFrame(label_rows)

# --- MERGE DATA ---
data = pred_df.merge(label_df, on=["uid", "t"], how="inner")

# --- RECONSTRUCT SPLIT ---
uids = sorted({flat_uid(r["uid"]) for r in raw_json})
train_u, temp_u = train_test_split(uids, test_size=0.2, random_state=SEED)
val_u, test_u   = train_test_split(temp_u, test_size=0.5, random_state=SEED)
data["Split"] = data["uid"].apply(lambda x: "train" if x in train_u else "val" if x in val_u else "test")

# --- PERIOD LABELING ---
data["Group"] = np.select(
    [
        data.FeatureBasedHoldout == 0,
        (data.FeatureBasedHoldout == 1) & (data.IndexBasedHoldout == 0),
        data.IndexBasedHoldout == 1
    ],
    ["Calibration", "HoldoutA", "HoldoutB"],
    default="UNASSIGNED"
)

# --- DEFINE BINARY BUCKETS ---
BIN_TASKS = {
    "BuyNone"    : [9],
    "BuyOne"     : [1,3,5,7],
    "BuyTen"     : [2,4,6,8],
    "BuyRegular" : [1,2],
    "BuyFigure"  : [3,4,5,6],
    "BuyWeapon"  : [7,8],
}

for task, pos in BIN_TASKS.items():
    data[f"y_{task}"] = data.label.isin(pos).astype(int)
    data[f"p_{task}"] = data[[f"p{i}" for i in pos]].sum(axis=1)

# --- F1 and AUPRC TABLE ---
metrics = []
for task in BIN_TASKS:
    for grp in ["Calibration", "HoldoutA", "HoldoutB"]:
        for split in ["val", "test"]:
            sdf = data[(data.Group == grp) & (data.Split == split)]
            if sdf.empty or sdf[f"y_{task}"].nunique() < 2:
                continue
            y_true = sdf[f"y_{task}"]
            y_pred = sdf[f"p_{task}"]
            f1     = f1_score(y_true, y_pred > 0.5)
            auprc  = average_precision_score(y_true, y_pred)
            metrics.append({
                "Metric": task,
                "Group": grp,
                "Split": split,
                "F1": round(f1, 4),
                "AUPRC": round(auprc, 4)
            })

# --- OUTPUT TABLE ---
df_out = pd.DataFrame(metrics).pivot(index=["Metric", "Group"], columns="Split", values=["F1", "AUPRC"]).round(4)
print("\n====== F1 Score and AUPRC (Binary Buckets) ======")
print(df_out.fillna("—"))
print("=================================================")
