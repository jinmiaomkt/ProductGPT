#!/usr/bin/env python3
# =============================================================
#  make_ROC_curve.py  –  robust ROC curves + per‑bucket AUC table
# =============================================================
from __future__ import annotations
import json, gzip, re, os, itertools
from pathlib import Path
from collections import Counter

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------------
# FILE LOCATIONS
# ------------------------------------------------------------------
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/LSTM/lstm_predictions.jsonl')
# PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/GRU/gru_predictions.jsonl')
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedFull/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_predictions.jsonl')
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedLP/LP_feature_predictions.jsonl')
LABEL_PATH = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json')
SEED       = 33
# ------------------------------------------------------------------

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS OpenMP clash

# -------------------------- helpers ---------------------------------
flat_uid = lambda u: str(u[0] if isinstance(u, list) else u)
def to_int_vec(x):
    if isinstance(x, str):   return [int(v) for v in x.split()]
    if isinstance(x, list):  return [int(v) for item in x for v in str(item).split()]
    raise TypeError(type(x))

# -------------------------- 1. labels -------------------------------
raw = json.loads(LABEL_PATH.read_text())
records = list(raw) if isinstance(raw, list) else [
    {k: raw[k][i] for k in raw} for i in range(len(raw["uid"]))
]

def explode(rec):
    uid = flat_uid(rec["uid"])
    y   = to_int_vec(rec["Decision"])
    idx = to_int_vec(rec["IndexBasedHoldout"])
    feat= to_int_vec(rec["FeatureBasedHoldout"])
    for t in range(len(y)):
        yield {"uid":uid,"t":t,"label":y[t],"idx_h":idx[t],"feat_h":feat[t]}

label_df = pd.DataFrame(itertools.chain.from_iterable(explode(r) for r in records))

# -------------------------- 2. predictions --------------------------
brace_re  = re.compile(r"\{.*\}")
open_pred = gzip.open if PRED_PATH.suffix == ".gz" else open
pred_rows, length_note = [], Counter()

with open_pred(PRED_PATH, "rt", errors="replace") as f:
    for line in f:
        line=line.strip()
        if not line or line[0] not in "{[":
            continue
        try:
            rec=json.loads(line)
        except json.JSONDecodeError:
            m=brace_re.search(line)
            if not m: continue
            try: rec=json.loads(m.group())
            except json.JSONDecodeError: continue

        uid=flat_uid(rec.get("uid",""))
        probs=rec["probs"]
        preds = probs if isinstance(probs[0], list) else [probs]
        for t,vec in enumerate(preds):
            pred_rows.append({"uid":uid,"t":t,**{f"p{i+1}":vec[i] for i in range(9)}})

pred_df = pd.DataFrame(pred_rows)

# -------------------------- 3. merge & split ------------------------
data = pred_df.merge(label_df, on=["uid","t"], how="inner")
if data.empty:
    raise RuntimeError("No uid/t pairs matched; check files.")

# reproduce 80‑10‑10 uid split
all_uids = [flat_uid(r["uid"]) for r in records]
train_u,temp_u = train_test_split(all_uids,test_size=0.2,random_state=SEED)
val_u,test_u   = train_test_split(temp_u,test_size=0.5,random_state=SEED)
data["Split"] = data.uid.apply(lambda u:"train" if u in train_u else
                                         "val"  if u in val_u   else "test")
data["Group"] = np.select(
    [data.feat_h==0,
     (data.feat_h==1)&(data.idx_h==0),
     data.idx_h==1],
    ["Calibration","HoldoutA","HoldoutB"],"UNASSIGNED")

# -------------------------- 4. binary buckets -----------------------
BIN_TASKS={
    "BuyNone":[9],
    "BuyOne":[1,3,5,7],
    "BuyTen":[2,4,6,8],
    "BuyRegular":[1,2],
    "BuyFigure":[3,4,5,6],
    "BuyWeapon":[7,8],
}
for task,pos in BIN_TASKS.items():
    pos_set=set(pos)
    data[f"y_{task}"]=data.label.isin(pos_set).astype(int)
    data[f"p_{task}"]=data[[f"p{i}" for i in pos]].sum(axis=1)

# -------------------------- 5. AUC table ----------------------------
rows=[]
for task in BIN_TASKS:
    for grp in ["Calibration","HoldoutA","HoldoutB"]:
        for spl in ["val","test"]:
            sdf=data[(data.Group==grp)&(data.Split==spl)]
            auc_val=np.nan if sdf[f"y_{task}"].nunique()<2 else \
                     roc_auc_score(sdf[f"y_{task}"],sdf[f"p_{task}"])
            rows.append({"Task":task,"Group":grp,"Split":spl,"AUC":auc_val})

auc_tbl=(pd.DataFrame(rows)
           .pivot(index=["Task","Group"],columns="Split",values="AUC")
           .round(4).sort_index())
print("\n========  Binary ROC‑AUC Table  =========")
print(auc_tbl.fillna(" NA"))
print("=========================================")

# -------------------------- 6. ROC curves ---------------------------
TASK   ="BuyNone"           # pick any key in BIN_TASKS
GROUPS =["Calibration","HoldoutA","HoldoutB"]

fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
for ax,grp in zip(axes,GROUPS):
    gdf=data[data.Group==grp]
    ins=gdf[gdf.Split=="train"]
    oos=gdf[gdf.Split.isin(["val","test"])]

    fpr_i,tpr_i,_=roc_curve(ins[f"y_{TASK}"],ins[f"p_{TASK}"])
    fpr_o,tpr_o,_=roc_curve(oos[f"y_{TASK}"],oos[f"p_{TASK}"])

    ax.plot(fpr_i,tpr_i,lw=2,label=f"In‑sample AUC={auc(fpr_i,tpr_i):.3f}")
    ax.plot(fpr_o,tpr_o,lw=2,ls="--",label=f"OOS AUC={auc(fpr_o,tpr_o):.3f}")
    ax.plot([0,1],[0,1],"k:",lw=1)
    ax.set_title(grp); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR" if grp=="Calibration" else "")
    ax.legend(loc="lower right")

fig.suptitle(f"ROC curves – bucket: {TASK}",fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
plt.show()
