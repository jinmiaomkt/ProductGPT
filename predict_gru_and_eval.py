#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_gru_and_eval.py

End-to-end for GRU:
  - Run inference for per-timestep 9-way decision probabilities (classes 1..9)
  - Compute AUC / Hit / F1 / AUPRC by (Task, PeriodGroup, Split)
  - Print AUC table(s) to console
  - Save CSV tables (and optional predictions) locally and upload to S3

Usage (example):
  python predict_gru_and_eval.py \
    --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
    --ckpt /home/ec2-user/tmp_gru/gru_h128_lr0.001_bs4.pt \
    --hidden-size 128 \
    --input-dim 15 \
    --labels '/home/ec2-user/data/clean_list_int_wide4_simple6.json' \
    --s3 's3://productgptbucket/experiments/gru/run_001/' \
    --pred-out /tmp/gru_preds.jsonl.gz \
    --uids-val s3://productgptbucket/FullProductGPT/performer/FeatureBased/folds/featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0_val_uids.txt \
    --uids-test s3://productgptbucket/FullProductGPT/performer/FeatureBased/folds/featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0_test_uids.txt \
    --fold-id 0
"""

from __future__ import annotations
import argparse, json, gzip, os, sys, math, subprocess
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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ===================== CLI ======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True, help="JSON array file (list of user records)")
    p.add_argument("--ckpt",        required=True, help="GRU *.pt state_dict checkpoint")
    p.add_argument("--hidden-size", type=int,     required=True, help="Hidden size used in training (must match ckpt)")
    p.add_argument("--input-dim",   type=int,     default=15,    help="Feature dim per timestep (default 15)")
    p.add_argument("--batch-size",  type=int,     default=128,   help="Batch size for inference")
    p.add_argument("--labels",      required=True, help="Label JSON (clean_list_int_wide4_simple6.json)")
    p.add_argument("--s3",          required=True, help="S3 URI prefix for uploads (e.g., s3://bucket/folder/)")
    p.add_argument("--pred-out",    default="",    help="Optional local predictions path (.jsonl or .jsonl.gz)")
    p.add_argument("--thresh",      type=float,    default=0.5,   help="Threshold for Hit/F1 on binary tasks")
    p.add_argument("--seed",        type=int,      default=33,    help="Seed for fallback 80/10/10 split")

    # EXACT-MATCH UID overrides (local path or s3://..., one UID per line)
    p.add_argument("--uids-val",    default="", help="Text file (or s3://...) with validation UIDs")
    p.add_argument("--uids-test",   default="", help="Text file (or s3://...) with test UIDs")
    p.add_argument("--fold-id",     type=int,  default=-1, help="If >=0, upload under .../fold{ID}/")
    return p.parse_args()

# ================== Utilities ===================
def smart_open_w(path: str | Path):
    """stdout if '-', gzip if *.gz, else normal text file (write)."""
    if isinstance(path, Path):
        path = str(path)
    if not path:
        raise ValueError("Empty path for smart_open_w")
    if path == "-":
        return nullcontext(sys.stdout)
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Invalid S3 uri: {uri}"
    no_scheme = uri[5:]
    if "/" in no_scheme:
        bucket, key = no_scheme.split("/", 1)
    else:
        bucket, key = no_scheme, ""
    return bucket, key

def s3_join(prefix: str, filename: str) -> str:
    if not prefix.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {prefix}")
    if not filename:
        raise ValueError("filename is empty")
    return (prefix if prefix.endswith("/") else prefix + "/") + filename

def s3_join_folder(prefix: str, folder: str) -> str:
    if not prefix.endswith("/"):
        prefix += "/"
    folder = folder.strip("/")
    return prefix + folder + "/"

def s3_upload_file(local_path: str | Path, s3_uri_full: str):
    assert not s3_uri_full.endswith("/"), f"S3 object key must not end with '/': {s3_uri_full}"
    try:
        import boto3
        bucket, key = parse_s3_uri(s3_uri_full)
        boto3.client("s3").upload_file(str(local_path), bucket, key)
    except Exception as e:
        rc = os.system(f"aws s3 cp '{local_path}' '{s3_uri_full}'")
        if rc != 0:
            raise RuntimeError(f"Failed to upload {local_path} to {s3_uri_full}: {e}")

def s3_read_text(s3_uri: str) -> str:
    bucket, key = parse_s3_uri(s3_uri)
    try:
        import boto3
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        data = subprocess.check_output(["aws", "s3", "cp", s3_uri, "-"])
        return data.decode("utf-8")

def load_uid_set(path_or_s3: str) -> Set[str]:
    if not path_or_s3:
        return set()
    text = s3_read_text(path_or_s3) if path_or_s3.startswith("s3://") else Path(path_or_s3).read_text()
    uids = set()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            uids.add(line)
    return uids

# ================= Data / Labels =================
def _json_array(path: str | Path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of objects")
    return raw

def to_int_vec(x):
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, str):
                out.extend(int(v) for v in item.split())
            else:
                out.append(int(item))
        return out
    raise TypeError(type(x))

def flat_uid(u):  # scalar or [scalar]
    return str(u[0] if isinstance(u, list) else u)

def load_labels(label_path: Path) -> Tuple[Dict[str, Dict[str, List[int]]], List[dict]]:
    raw = json.loads(label_path.read_text())
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
    return label_dict, records

def build_splits(records, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(records)
    tr, va = int(0.8*n), int(0.1*n)
    tr_i, va_i, te_i = random_split(range(n), [tr, va, n-tr-va], generator=g)
    val_uid  = {flat_uid(records[i]["uid"]) for i in va_i.indices}
    test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}
    def which_split(u):
        return "val" if u in val_uid else "test" if u in test_uid else "train"
    return which_split

# ================= Dataset & Collate (GRU) =============
class PredictDataset(Dataset):
    """
    Expects records with:
      - uid: scalar or [scalar]
      - AggregateInput: [ "t1 t2 ... tK" ]  (K multiple of input_dim)
    """
    def __init__(self, json_path: Path, input_dim: int):
        self.rows = _json_array(json_path)
        self.input_dim = input_dim

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        uid = flat_uid(rec["uid"])
        feat_str = rec["AggregateInput"][0] if isinstance(rec["AggregateInput"], list) else rec["AggregateInput"]
        flat = [0.0 if tok == "NA" else float(tok) for tok in str(feat_str).strip().split()]
        T = len(flat) // self.input_dim
        if T == 0:
            x = torch.zeros((1, self.input_dim), dtype=torch.float32)
        else:
            x = torch.tensor(flat[:T*self.input_dim], dtype=torch.float32).view(T, self.input_dim)
        return {"uid": uid, "x": x, "T": T}

def collate(batch):
    uids = [b["uid"] for b in batch]
    xs   = [b["x"] for b in batch]
    Ts   = [b["T"] for b in batch]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return {"uid": uids, "x": x_pad, "T": Ts}

# ================== GRU Model ==========================
class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, H)
        return self.fc(out)           # (B, T, C=10)

# =================== Metrics Setup =====================
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

# ======================= Main ==========================
def main():
    args = parse_args()
    data_path  = Path(args.data)
    ckpt_path  = Path(args.ckpt)
    label_path = Path(args.labels)
    s3_prefix  = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    pred_out   = args.pred_out

    # If fold-id provided, nest under that folder
    s3_prefix_effective = s3_join_folder(s3_prefix, f"fold{args.fold_id}") if args.fold_id is not None and args.fold_id >= 0 else s3_prefix
    print(f"[INFO] S3 upload prefix: {s3_prefix_effective}")

    # ---------- Labels ----------
    label_dict, records = load_labels(label_path)

    # ---------- EXACT MATCH OVERRIDE or fallback 80/10/10 ----------
    uids_val_override  = load_uid_set(args.uids_val)  if args.uids_val  else set()
    uids_test_override = load_uid_set(args.uids_test) if args.uids_test else set()

    if uids_val_override or uids_test_override:
        if not (uids_val_override and uids_test_override):
            raise ValueError("Provide BOTH --uids-val and --uids-test (or neither).")
        overlap = uids_val_override & uids_test_override
        if overlap:
            raise ValueError(f"UIDs present in BOTH val and test: {sorted(list(overlap))[:5]} ...")

        def which_split(u):
            return "val" if u in uids_val_override else "test" if u in uids_test_override else "train"
        print(f"[INFO] Using EXACT UID lists: val={len(uids_val_override)}, test={len(uids_test_override)}")
    else:
        which_split = build_splits(records, seed=args.seed)
        print(f"[INFO] Using fallback 80/10/10 split with seed={args.seed}")

    # ---------- DataLoader ----------
    ds = PredictDataset(data_path, input_dim=args.input_dim)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GRUClassifier(
        input_dim   = args.input_dim,
        hidden_size = args.hidden_size,
        num_classes = 10
    ).to(device).eval()

    # ---------- Load checkpoint ----------
    # Your GRU training saved state_dict directly: torch.save(model.state_dict(), ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        sd = state
    else:
        # fallback if someone saved {"model_state_dict": ...}
        sd = state.get("model_state_dict", state)
    # strip DistributedDataParallel prefix if any
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    model.load_state_dict(sd, strict=True)

    # ---------- Metric accumulators ----------
    scores       = defaultdict(lambda: {"y": [], "p": []})
    length_note  = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    # Optional predictions writer
    pred_writer = smart_open_w(pred_out) if pred_out else None

    # ---------- Inference + streaming eval ----------
    with torch.no_grad():
        for batch in loader:
            x    = batch["x"].to(device)   # (B, T, D)
            Ts   = batch["T"]
            uids = batch["uid"]

            logits10 = model(x)            # (B, T, 10)
            logits9  = logits10[..., 1:]   # (B, T, 9) drop PAD=0
            probs9   = F.softmax(logits9, dim=-1)

            for i, uid in enumerate(uids):
                T = Ts[i]
                probs_seq = probs9[i, :T].detach().cpu().numpy()  # (T, 9), slice off pad tail
                # write predictions line if requested
                if pred_writer:
                    pred_writer.write(json.dumps({"uid": uid, "probs": np.round(probs_seq, 6).tolist()}) + "\n")

                lbl_info = label_dict.get(uid)
                if lbl_info is None:
                    reject += 1
                    continue

                L_pred, L_lbl = len(probs_seq), len(lbl_info["label"])
                if L_pred != L_lbl:
                    length_note["pred>lbl" if L_pred > L_lbl else "pred<label"] += 1
                L = min(L_pred, L_lbl)

                split_tag = which_split(uid)
                accept_users.setdefault(split_tag, set()).add(uid)
                for t in range(L):
                    y      = lbl_info["label"][t]      # 1..9 (0 is pad; should not appear here)
                    idx_h  = lbl_info["idx_h"][t]
                    feat_h = lbl_info["feat_h"][t]
                    probs  = probs_seq[t]              # length 9, classes 1..9

                    group = period_group(idx_h, feat_h)
                    for task, pos_classes in BIN_TASKS.items():
                        y_bin = int(y in TASK_POSSETS[task])
                        p_bin = float(sum(probs[j-1] for j in pos_classes))  # j in {1..9} → index j-1
                        key   = (task, group, split_tag)
                        scores[key]["y"].append(y_bin)
                        scores[key]["p"].append(p_bin)

                accept += 1

    if pred_writer:
        pred_writer.__exit__(None, None, None)

    print(f"[INFO] parsed: {accept} users accepted, {reject} users missing labels.")
    if length_note:
        print("[INFO] length mismatches:", dict(length_note))
    if args.uids_val and args.uids_test:
        print(f"[INFO] coverage: val={len(accept_users.get('val', set()))} / {len(load_uid_set(args.uids_val))}, "
              f"test={len(accept_users.get('test', set()))} / {len(load_uid_set(args.uids_test))}")

    # ---------- Compute tables ----------
    rows = []
    for task in BIN_TASKS:
        for grp in ["Calibration","HoldoutA","HoldoutB"]:
            for spl in ["val","test"]:
                y, p = scores[(task, grp, spl)]["y"], scores[(task, grp, spl)]["p"]
                if not y:
                    continue
                if len(set(y)) < 2:
                    auc = acc = f1 = auprc = np.nan
                else:
                    auc   = roc_auc_score(y, p)
                    y_hat = [int(prob >= args.thresh) for prob in p]
                    acc   = accuracy_score(y, y_hat)
                    f1    = f1_score(y, y_hat)
                    auprc = average_precision_score(y, p)
                rows.append({"Task": task, "Group": grp, "Split": spl,
                             "AUC": auc, "Hit": acc, "F1": f1, "AUPRC": auprc})
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

    macro_period_tbl = (
        metrics
          .groupby(["Group", "Split"])[["AUC", "Hit", "F1", "AUPRC"]]
          .mean()
          .unstack("Split")
          .round(4)
    )
    macro_period_tbl = macro_period_tbl.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0)
    macro_period_tbl = macro_period_tbl[['val', 'test']]

    # ---------- Print ALL tables to console ----------
    def _p(title: str, df: pd.DataFrame):
        print(f"\n=============  {title}  =======================")
        print(df.fillna(" NA"))
        print("============================================================")

    _p("BINARY ROC-AUC TABLE", auc_tbl)
    _p("HIT-RATE (ACCURACY) TABLE", hit_tbl)
    _p("MACRO-F1 TABLE", f1_tbl)
    _p("AUPRC TABLE", auprc_tbl)
    _p("AGGREGATE MACRO METRICS", macro_period_tbl)

    # ---------- Save locally & upload to S3 ----------
    out_dir = Path("/tmp/predict_eval_outputs_gru")
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_csv   = out_dir / "auc_table.csv"
    hit_csv   = out_dir / "hit_table.csv"
    f1_csv    = out_dir / "f1_table.csv"
    auprc_csv = out_dir / "auprc_table.csv"
    macro_csv = out_dir / "macro_period_table.csv"

    auc_tbl.to_csv(auc_csv)
    hit_tbl.to_csv(hit_csv)
    f1_tbl.to_csv(f1_csv)
    auprc_tbl.to_csv(auprc_csv)
    macro_period_tbl.to_csv(macro_csv)

    for pth in [auc_csv, hit_csv, f1_csv, auprc_csv, macro_csv]:
        dest = s3_join(s3_prefix_effective, pth.name)
        s3_upload_file(pth, dest)
        print(f"[S3] uploaded: {dest}")

    if pred_out:
        dest = s3_join(s3_prefix_effective, Path(pred_out).name)
        s3_upload_file(pred_out, dest)
        print(f"[S3] uploaded: {dest}")

if __name__ == "__main__":
    main()
