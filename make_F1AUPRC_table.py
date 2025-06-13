#!/usr/bin/env python3
# =============================================================
#  make_AUC_table.py  –  AUC table + ROC curves
# =============================================================
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import random_split
import torch, matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. EDIT YOUR FILE LOCATIONS
# ------------------------------------------------------------------
PRED_PATH  = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/productgpt_predictions.jsonl")
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

def explode_record(rec):
    uid = flat_uid(rec["uid"])
    y      = to_int_vec(rec["Decision"])
    idx_h  = to_int_vec(rec["IndexBasedHoldout"])
    feat_h = to_int_vec(rec["FeatureBasedHoldout"])
    assert len({len(y),len(idx_h),len(feat_h)}) == 1, "len mismatch"

    for t in range(len(y)):
        yield {"uid": uid, "t": t,
               "label": y[t],
               "IndexBasedHoldout": idx_h[t],
               "FeatureBasedHoldout": feat_h[t]}

# accept dict-of-lists OR list-of-dicts
if isinstance(raw_json, dict):
    n_rows = len(raw_json["uid"])
    raw_iter = (
        {k: raw_json[k][i] for k in raw_json} for i in range(n_rows)
    )
else:
    raw_iter = raw_json

label_rows = list(itertools.chain.from_iterable(
                    explode_record(r) for r in raw_iter))
label_df   = pd.DataFrame(label_rows)

# ---------------------- 3. merge ---------------------------------
data = pred_df.merge(label_df, on=["uid","t"], how="inner")
if data.empty:
    raise RuntimeError("No uid/t pairs matched!")

# ---------------------- 4. rebuild 80-10-10 split -----------------
raw_records = list(raw_json) if isinstance(raw_json, list) else \
              [{k: raw_json[k][i] for k in raw_json} for i in range(len(raw_json["uid"]))]

train_sz, val_sz = int(0.8*len(raw_records)), int(0.1*len(raw_records))
test_sz          = len(raw_records) - train_sz - val_sz

g = torch.Generator().manual_seed(SEED)
train_idx, val_idx, test_idx = random_split(
    range(len(raw_records)), [train_sz, val_sz, test_sz], generator=g)

train_uid = {flat_uid(raw_records[i]["uid"]) for i in train_idx.indices}
val_uid   = {flat_uid(raw_records[i]["uid"]) for i in val_idx.indices}
test_uid  = {flat_uid(raw_records[i]["uid"]) for i in test_idx.indices}

def which_split(u):
    if u in val_uid:  return "val"
    if u in test_uid: return "test"
    return "train"
data["Split"] = data["uid"].map(which_split)

# ---------------------- 5. tag periods ---------------------------
data["Group"] = np.select(
    [
        data.FeatureBasedHoldout == 0,
        (data.FeatureBasedHoldout == 1) & (data.IndexBasedHoldout == 0),
        data.IndexBasedHoldout == 1
    ],
    ["Calibration","HoldoutA","HoldoutB"],
    default="UNASSIGNED")

# ---------------------- 6. build binary buckets ------------------
BIN_TASKS = {
    "BuyNone"    : [9],
    "BuyOne"     : [1,3,5,7],
    "BuyTen"     : [2,4,6,8],
    "BuyRegular" : [1,2],
    "BuyFigure"  : [3,4,5,6],
    "BuyWeapon"  : [7,8],
}
for task,pos in BIN_TASKS.items():
    pos_set = set(pos)
    data[f"y_{task}"] = data.label.isin(pos_set).astype(int)
    data[f"p_{task}"] = data[[f"p{i}" for i in pos]].sum(axis=1)

# ---------------------- 7. compute AUC table ---------------------
def bin_auc(df, task):
    return np.nan if df[f"y_{task}"].nunique()<2 \
           else roc_auc_score(df[f"y_{task}"], df[f"p_{task}"])

rows=[]
for task in BIN_TASKS:
    for grp in ["Calibration","HoldoutA","HoldoutB"]:
        for split in ["val","test"]:
            sdf = data[(data.Group==grp)&(data.Split==split)]
            rows.append({"Task":task,"Group":grp,"Split":split,
                         "AUC":bin_auc(sdf,task)})
auc_tbl = (pd.DataFrame(rows)
             .pivot(index=["Task","Group"], columns="Split", values="AUC")
             .round(4)
             .sort_index())

print("\n=============  BINARY ROC-AUC TABLE  =======================")
print(auc_tbl.fillna(" NA"))
print("============================================================")

# ---------------------- 8. ROC curves (one bucket) ---------------
# TASK   = "Buy"                      # choose any key in BIN_TASKS
# GROUPS = ["Calibration","HoldoutA","HoldoutB"]

# fig, axes = plt.subplots(1,3,figsize=(14,5),sharey=True)
# for ax, grp in zip(axes, GROUPS):
#     gdf = data[data.Group==grp]

#     ins = gdf[gdf.Split=="train"]
#     oos = gdf[gdf.Split.isin(["val","test"])]

#     fpr_i,tpr_i,_ = roc_curve(ins[f"y_{TASK}"], ins[f"p_{TASK}"])
#     fpr_o,tpr_o,_ = roc_curve(oos[f"y_{TASK}"], oos[f"p_{TASK}"])

#     ax.plot(fpr_i,tpr_i,lw=2,label=f"In-sample  AUC={auc(fpr_i,tpr_i):.3f}")
#     ax.plot(fpr_o,tpr_o,lw=2,ls="--",
#             label=f"OOS  (val+test) AUC={auc(fpr_o,tpr_o):.3f}")
#     ax.plot([0,1],[0,1],"k:",lw=1)
#     ax.set_title(grp); ax.set_xlim(0,1); ax.set_ylim(0,1)
#     ax.set_xlabel("FPR"); ax.set_ylabel("TPR" if grp=="Calibration" else "")
#     ax.legend(loc="lower right")
# fig.suptitle(f"ROC curves – bucket: {TASK}",fontsize=14)
# fig.tight_layout(rect=[0,0,1,0.95]); plt.show()

# =============================================================
# 9.  Group-/Sub-group table: HitRate  •  F1  •  AUPRC
# =============================================================
from sklearn.metrics import f1_score, average_precision_score

# ----------  declare your models here  ------------------------
MODELS = {
    "ProductGPT"   : "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/productgpt_predictions.jsonl",
    "GRU"          : "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/gru_predictions.jsonl",
    "LSTM"         : "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/lstm_predictions.jsonl",
}

metric_rows = []

# helper to load a model’s predictions into a DF with uid/t/ p1..p9
def load_preds(jsonl_path):
    rows = []
    with open(jsonl_path) as f:
        for ln in f:
            rec  = json.loads(ln)
            uid  = flat_uid(rec["uid"])
            probs= rec["probs"]
            for t,p in enumerate(probs):
                rows.append({"uid":uid,"t":t,
                             **{f"p{i+1}":p[i] for i in range(9)}})
    return pd.DataFrame(rows)

for model_name, path in MODELS.items():
    print(f"→   scoring  {model_name}")
    pdf = load_preds(path)
    df  = data[["uid","t","label","Group","Split"]].merge(
              pdf, on=["uid","t"], how="inner")

    #   top-1 prediction (Hit-Rate)
    df["pred"] = df[[f"p{i}" for i in range(1,10)]].idxmax(axis=1).str[1:].astype(int)

    for grp in ["Calibration","HoldoutA","HoldoutB"]:
        for split in ["val","test"]:                      # sub-groups wanted
            sub = df[(df.Group==grp)&(df.Split==split)]
            if sub.empty:   continue

            hit_rate = np.mean(sub.pred == sub.label)

            f1  = f1_score(sub.label, sub.pred, average="macro")

            # Average class-wise AUPRC
            prc = []
            for cls in range(1,10):
                y_true = (sub.label==cls).astype(int)
                y_scr  = sub[f"p{cls}"]
                if y_true.nunique()==1:   # skip classes not present
                    continue
                prc.append(average_precision_score(y_true,y_scr))
            auprc = np.mean(prc) if prc else np.nan

            metric_rows.append({
                "Model":model_name, "Group":grp, "Sub":split,
                "HitRate":hit_rate, "F1":f1, "AUPRC":auprc
            })

# ------------  assemble pretty tables  ---------------------------
tbl = (pd.DataFrame(metric_rows)
         .set_index(["Group","Sub","Model"])
         .sort_index()
         .round(4))

for grp in ["Calibration","HoldoutA","HoldoutB"]:
    for sub in ["val","test"]:
        if (grp,sub) not in tbl.index: continue
        print(f"\n=== {grp}  –  {sub} set ===\n")
        print(tbl.loc[(grp,sub)]
                .rename(columns={"HitRate":"Hit Rate",
                                 "F1":"Avg. F1",
                                 "AUPRC":"Avg. AUPRC"})
                .to_string())
