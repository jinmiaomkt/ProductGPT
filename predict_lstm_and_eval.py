#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_lstm_and_eval_raw.py

Matches a raw-input LSTM checkpoint trained with input_dim=15 and an 'fc' head.
- Runs inference (per-timestep classes 1..9)
- Computes AUC / Hit / F1 / AUPRC by (Task, PeriodGroup, Split)
- Prints tables, saves CSVs locally, uploads to S3
"""

from __future__ import annotations
import argparse, json, gzip, os, sys, subprocess
from contextlib import nullcontext
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True, help="JSON array of user records (AggregateInput present)")
    p.add_argument("--ckpt",        required=True, help="LSTM *.pt (state_dict with lstm.* and fc.*)")
    p.add_argument("--hidden-size", type=int,     required=True, help="Hidden size used in training (match ckpt)")
    p.add_argument("--input-dim",   type=int,     default=15,    help="Feature dim per timestep (raw features)")
    p.add_argument("--batch-size",  type=int,     default=128)
    p.add_argument("--labels",      required=True, help="Labels JSON with Decision/IndexBasedHoldout/FeatureBasedHoldout")
    p.add_argument("--s3",          required=True, help="S3 prefix for uploads (s3://bucket/path/)")
    p.add_argument("--pred-out",    default="",    help="Optional local predictions path (.jsonl or .jsonl.gz)")
    p.add_argument("--thresh",      type=float,    default=0.5)
    p.add_argument("--seed",        type=int,      default=33)
    p.add_argument("--uids-val",    default="")
    p.add_argument("--uids-test",   default="")
    p.add_argument("--fold-id",     type=int,      default=-1)
    return p.parse_args()

# ---------- I/O helpers ----------
def smart_open_w(path: str | Path):
    if not path: return None
    path = str(path)
    if path == "-": return nullcontext(sys.stdout)
    if path.endswith(".gz"): return gzip.open(path, "wt")
    return open(path, "w")

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Invalid S3 uri: {uri}"
    no = uri[5:]
    return (no.split("/", 1) + [""])[:2] if "/" in no else (no, "")

def s3_join(prefix: str, filename: str) -> str:
    if not prefix.endswith("/"): prefix += "/"
    return prefix + filename

def s3_join_folder(prefix: str, folder: str) -> str:
    if not prefix.endswith("/"): prefix += "/"
    return prefix + folder.strip("/") + "/"

def s3_upload_file(local_path: str | Path, s3_uri_full: str):
    try:
        import boto3
        b, k = parse_s3_uri(s3_uri_full)
        boto3.client("s3").upload_file(str(local_path), b, k)
    except Exception:
        rc = os.system(f"aws s3 cp '{local_path}' '{s3_uri_full}'")
        if rc != 0:
            raise RuntimeError(f"Failed to upload {local_path} to {s3_uri_full}")

def s3_read_text(s3_uri: str) -> str:
    b, k = parse_s3_uri(s3_uri)
    try:
        import boto3
        obj = boto3.client("s3").get_object(Bucket=b, Key=k)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        return subprocess.check_output(["aws", "s3", "cp", s3_uri, "-"]).decode("utf-8")

def load_uid_set(path_or_s3: str) -> Set[str]:
    if not path_or_s3: return set()
    txt = s3_read_text(path_or_s3) if path_or_s3.startswith("s3://") else Path(path_or_s3).read_text()
    return {ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")}

# ---------- Labels & splits ----------
def to_int_vec(x):
    if isinstance(x, str): return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, str): out.extend(int(v) for v in item.split())
            else: out.append(int(item))
        return out
    raise TypeError(type(x))

def flat_uid(u): return str(u[0] if isinstance(u, list) else u)

def load_labels(path: Path) -> Tuple[Dict[str, Dict[str, List[int]]], List[dict]]:
    raw = json.loads(path.read_text())
    records = list(raw) if isinstance(raw, list) else [{k: raw[k][i] for k in raw} for i in range(len(raw["uid"]))]
    label_dict = {
        flat_uid(rec["uid"]): {
            "label":  to_int_vec(rec["Decision"]),
            "idx_h":  to_int_vec(rec["IndexBasedHoldout"]),
            "feat_h": to_int_vec(rec["FeatureBasedHoldout"]),
        } for rec in records
    }
    return label_dict, records

def build_splits(records, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(records); tr, va = int(.8*n), int(.1*n)
    tr_i, va_i, te_i = random_split(range(n), [tr, va, n-tr-va], generator=g)
    val_uid  = {flat_uid(records[i]["uid"]) for i in va_i.indices}
    test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}
    def which(u): return "val" if u in val_uid else "test" if u in test_uid else "train"
    return which

# ---------- Dataset ----------
class PredictDataset(Dataset):
    def __init__(self, json_path: Path, input_dim: int):
        raw = json.loads(json_path.read_text())
        if not isinstance(raw, list): raise ValueError("Input JSON must be an array of objects")
        self.rows = raw; self.input_dim = input_dim
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        rec = self.rows[i]
        uid = flat_uid(rec["uid"])
        feat_str = rec["AggregateInput"][0] if isinstance(rec["AggregateInput"], list) else rec["AggregateInput"]
        flat = [0.0 if t == "NA" else float(t) for t in str(feat_str).split()]
        T = len(flat) // self.input_dim
        x = torch.tensor(flat[:T*self.input_dim], dtype=torch.float32).view(max(T,1), self.input_dim)
        return {"uid": uid, "x": x, "T": T}

def collate(batch):
    uids = [b["uid"] for b in batch]
    Ts   = [b["T"] for b in batch]
    xs   = [b["x"] for b in batch]
    xpad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return {"uid": uids, "x": xpad, "T": Ts}

# ---------- Model ----------
class LSTMDecisionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)   # (B, T, H)
        return self.fc(out)     # (B, T, C=10)

# ---------- Metrics ----------
BIN_TASKS = {
    "BuyNone":   [9],
    "BuyOne":    [1,3,5,7],
    "BuyTen":    [2,4,6,8],
    "BuyRegular":[1,2],
    "BuyFigure": [3,4,5,6],
    "BuyWeapon": [7,8],
}
TASK_POSSETS = {k: set(v) for k,v in BIN_TASKS.items()}

def period_group(idx_h, feat_h):
    if feat_h == 0:               return "Calibration"
    if feat_h == 1 and idx_h == 0:return "HoldoutA"
    if idx_h == 1:                return "HoldoutB"
    return "UNASSIGNED"

# ---------- Main ----------
def main():
    args = parse_args()
    data_path  = Path(args.data)
    label_path = Path(args.labels)
    s3_prefix  = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    if args.fold_id is not None and args.fold_id >= 0:
        s3_prefix = s3_join_folder(s3_prefix, f"fold{args.fold_id}")
    print(f"[INFO] S3 upload prefix: {s3_prefix}")

    label_dict, records = load_labels(label_path)

    uids_val = load_uid_set(args.uids_val) if args.uids_val else set()
    uids_test= load_uid_set(args.uids_test) if args.uids_test else set()
    if uids_val or uids_test:
        if not (uids_val and uids_test): raise ValueError("Provide BOTH --uids-val and --uids-test (or neither).")
        if uids_val & uids_test: raise ValueError("UID overlap between val and test.")
        def which(u): return "val" if u in uids_val else "test" if u in uids_test else "train"
        print(f"[INFO] Using EXACT UID lists: val={len(uids_val)}, test={len(uids_test)}")
    else:
        which = build_splits(records, seed=args.seed)
        print(f"[INFO] Using fallback 80/10/10 split with seed={args.seed}")

    ds = PredictDataset(data_path, input_dim=args.input_dim)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMDecisionModel(input_dim=args.input_dim, hidden_size=args.hidden_size, num_classes=10).to(device).eval()

    # strict load of raw-feature LSTM checkpoint (strip "module." if present)
    state = torch.load(args.ckpt, map_location=device)
    sd = state.get("model_state_dict", state)
    sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)

    scores = defaultdict(lambda: {"y": [], "p": []})
    length_note = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    writer = smart_open_w(args.pred_out) if args.pred_out else None

    with torch.no_grad():
        for batch in loader:
            x, Ts, uids = batch["x"].to(device), batch["T"], batch["uid"]
            logits10 = model(x)                 # (B, T, 10)
            probs9   = F.softmax(logits10[...,1:], dim=-1)  # (B,T,9)

            for i, uid in enumerate(uids):
                T = Ts[i]
                probs_seq = probs9[i,:T].cpu().numpy()
                if writer:
                    writer.write(json.dumps({"uid": uid, "probs": np.round(probs_seq, 6).tolist()}) + "\n")

                lbl = label_dict.get(uid)
                if lbl is None: reject += 1; continue

                L = min(len(probs_seq), len(lbl["label"]))
                split = which(uid)
                accept_users.setdefault(split, set()).add(uid)
                for t in range(L):
                    y = lbl["label"][t]
                    idx_h, feat_h = lbl["idx_h"][t], lbl["feat_h"][t]
                    group = period_group(idx_h, feat_h)
                    for task, pos in BIN_TASKS.items():
                        y_bin = int(y in TASK_POSSETS[task])
                        p_bin = float(sum(probs_seq[t][j-1] for j in pos))
                        key = (task, group, split)
                        scores[key]["y"].append(y_bin)
                        scores[key]["p"].append(p_bin)
                if len(probs_seq) != len(lbl["label"]):
                    length_note["pred>lbl" if len(probs_seq) > len(lbl["label"]) else "pred<label"] += 1
                accept += 1

    if writer: writer.__exit__(None, None, None)
    print(f"[INFO] parsed: {accept} users accepted, {reject} users missing labels.")
    if length_note: print("[INFO] length mismatches:", dict(length_note))
    if args.uids_val and args.uids_test:
        print(f"[INFO] coverage: val={len(accept_users.get('val', set()))}/{len(uids_val)}, "
              f"test={len(accept_users.get('test', set()))}/{len(uids_test)}")

    # tables
    rows = []
    for task in BIN_TASKS:
        for grp in ["Calibration","HoldoutA","HoldoutB"]:
            for spl in ["val","test"]:
                y, p = scores[(task, grp, spl)]["y"], scores[(task, grp, spl)]["p"]
                if not y: continue
                if len(set(y)) < 2:
                    auc = acc = f1 = auprc = np.nan
                else:
                    auc = roc_auc_score(y, p)
                    yhat = [int(pp >= args.thresh) for pp in p]
                    acc = accuracy_score(y, yhat)
                    f1  = f1_score(y, yhat)
                    auprc = average_precision_score(y, p)
                rows.append({"Task": task, "Group": grp, "Split": spl, "AUC": auc, "Hit": acc, "F1": f1, "AUPRC": auprc})
    metrics = pd.DataFrame(rows)

    def pivot(metric): 
        return (metrics.pivot(index=["Task","Group"], columns="Split", values=metric)
                        .reindex(columns=["val","test"]).round(4).sort_index())

    auc_tbl, hit_tbl, f1_tbl, auprc_tbl = map(pivot, ["AUC","Hit","F1","AUPRC"])
    macro_tbl = (metrics.groupby(["Group","Split"])[["AUC","Hit","F1","AUPRC"]].mean()
                        .unstack("Split").round(4))
    macro_tbl = macro_tbl.reorder_levels([1,0], axis=1).sort_index(axis=1, level=0)[['val','test']]

    def _p(title, df):
        print(f"\n=============  {title}  =======================")
        print(df.fillna(" NA"))
        print("================================================")

    _p("BINARY ROC-AUC TABLE", auc_tbl)
    _p("HIT-RATE (ACCURACY) TABLE", hit_tbl)
    _p("MACRO-F1 TABLE", f1_tbl)
    _p("AUPRC TABLE", auprc_tbl)
    _p("AGGREGATE MACRO METRICS", macro_tbl)

    out_dir = Path("/tmp/predict_eval_outputs_lstm_raw"); out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {
        "auc_table.csv": auc_tbl,
        "hit_table.csv": hit_tbl,
        "f1_table.csv": f1_tbl,
        "auprc_table.csv": auprc_tbl,
        "macro_period_table.csv": macro_tbl,
    }.items():
        pth = out_dir / name
        df.to_csv(pth)
        s3_upload_file(pth, s3_join(s3_prefix, name))
        print(f"[S3] uploaded: {s3_join(s3_prefix, name)}")

    if args.pred_out:
        s3_upload_file(args.pred_out, s3_join(s3_prefix, Path(args.pred_out).name))
        print(f"[S3] uploaded: {s3_join(s3_prefix, Path(args.pred_out).name)}")

if __name__ == "__main__":
    main()
