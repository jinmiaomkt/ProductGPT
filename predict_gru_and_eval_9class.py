#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_gru_and_eval_new.py

End-to-end for GRU — evaluation methodology matches predict_productgpt_and_eval.py:
  - Run inference for per-timestep 9-way decision probabilities (classes 1..9)
  - End-align predictions with labels when lengths differ
  - Compute AUC / AUPRC by (Task, PeriodGroup, Split)  [binary tasks]
  - Compute 6 multiclass metrics by (PeriodGroup, Split):
      MacroOvR_AUC, MacroAUPRC, MacroF1, MCC, LogLoss, Top2Acc
  - Print paper-style tables to console
  - Save CSV tables (and optional predictions) locally and upload to S3

Usage (example):
  python predict_gru_and_eval_new.py \
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
    --ckpt /home/ec2-user/tmp_gru/gru_h128_lr0.001_bs4.pt \
    --hidden-size 128 \
    --input-dim 15 \
    --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
    --s3 s3://productgptbucket/experiments/gru/run_001/ \
    --pred-out /tmp/gru_preds.jsonl.gz \
    --uids-val s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \
    --uids-test s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \
    --fold-id 0
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

# ── CHANGED: added matthews_corrcoef, log_loss, label_binarize ──────────────
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
    matthews_corrcoef, log_loss,
)
from sklearn.preprocessing import label_binarize
# ────────────────────────────────────────────────────────────────────────────

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
    p.add_argument("--seed",        type=int,      default=33,    help="Seed for fallback 80/10/10 split")

    # EXACT-MATCH UID overrides (local path or s3://..., one UID per line)
    p.add_argument("--uids-val",    default="", help="Text file (or s3://...) with validation UIDs")
    p.add_argument("--uids-test",   default="", help="Text file (or s3://...) with test UIDs")
    p.add_argument("--fold-id",     type=int,  default=-1, help="If >=0, upload under .../fold{ID}/")
    return p.parse_args()

# ================== Utilities ===================
def smart_open_w(path: str | Path):
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

def flat_uid(u):
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
            T = 1
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
        out, _ = self.gru(x)
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

GROUP_ORDER = ["Calibration", "HoldoutA", "HoldoutB"]
SPLIT_ORDER = ["val", "test"]

# ── CHANGED: class labels for the 9-class problem ───────────────────────────
NINE_CLASSES = list(range(1, 10))   # [1, 2, ..., 9]
# ────────────────────────────────────────────────────────────────────────────

PAPER_AUC_TASKS = {
    "BuyOne": "Buy One",
    "BuyTen": "Buy Ten",
    "BuyFigure": "Character Event Wish",
    "BuyWeapon": "Weapon Event Wish",
    "BuyRegular": "Regular Wish",
}
PAPER_AUC_ORDER = [
    "Buy One",
    "Buy Ten",
    "Character Event Wish",
    "Weapon Event Wish",
    "Regular Wish",
]

def period_group(idx_h, feat_h):
    if feat_h == 0:               return "Calibration"
    if feat_h == 1 and idx_h == 0:return "HoldoutA"
    if idx_h == 1:                return "HoldoutB"
    return "UNASSIGNED"

# ── CHANGED: helper to compute all six 9-class metrics ──────────────────────
def compute_multiclass_metrics(
    y_arr: np.ndarray,
    p_arr: np.ndarray,
) -> dict:
    """
    Compute 6 multiclass metrics for a single (group, split) bucket.

    Parameters
    ----------
    y_arr : (N,) int array, true class labels in {1..9}
    p_arr : (N, 9) float array, predicted probabilities (must sum to ~1 per row)

    Returns
    -------
    dict with keys:
        MacroOvR_AUC, MacroAUPRC, MacroF1, MCC, LogLoss, Top2Acc
    """
    # Hard predictions (argmax → back to 1-indexed)
    y_hat = p_arr.argmax(axis=1) + 1                      # (N,)

    # ── 1. Macro One-vs-Rest AUC ──────────────────────────────────────────
    try:
        macro_ovr_auc = roc_auc_score(
            y_arr, p_arr,
            multi_class="ovr",
            average="macro",
            labels=NINE_CLASSES,
        )
    except ValueError:
        macro_ovr_auc = np.nan

    # ── 2. Macro AUPRC ────────────────────────────────────────────────────
    # Binarise labels into a (N, 9) matrix, then average per-class AP.
    # label_binarize with classes=[1..9] maps label k → column k-1.
    y_bin = label_binarize(y_arr, classes=NINE_CLASSES)    # (N, 9)
    per_class_ap = []
    for c in range(9):
        if y_bin[:, c].sum() == 0:
            continue                  # skip entirely absent classes
        per_class_ap.append(average_precision_score(y_bin[:, c], p_arr[:, c]))
    macro_auprc = float(np.mean(per_class_ap)) if per_class_ap else np.nan

    # ── 3. Macro-F1 ───────────────────────────────────────────────────────
    macro_f1 = f1_score(
        y_arr, y_hat,
        labels=NINE_CLASSES,
        average="macro",
        zero_division=0,
    )

    # ── 4. Matthews Correlation Coefficient ───────────────────────────────
    mcc = matthews_corrcoef(y_arr, y_hat)

    # ── 5. Log-Loss (Cross-Entropy) ───────────────────────────────────────
    p_clipped = np.clip(p_arr, 1e-7, 1.0)
    ll = log_loss(y_arr, p_clipped, labels=NINE_CLASSES)

    # ── 6. Top-2 Accuracy ─────────────────────────────────────────────────
    top2_indices = np.argsort(p_arr, axis=1)[:, -2:]      # (N, 2), 0-indexed
    top2_labels  = top2_indices + 1                        # convert to 1-indexed
    top2_acc = float(np.mean(
        [y_arr[i] in top2_labels[i] for i in range(len(y_arr))]
    ))

    return {
        "MacroOvR_AUC": round(float(macro_ovr_auc), 4),
        "MacroAUPRC":   round(float(macro_auprc),   4),
        "MacroF1":      round(float(macro_f1),       4),
        "MCC":          round(float(mcc),            4),
        "LogLoss":      round(float(ll),             4),
        "Top2Acc":      round(float(top2_acc),       4),
    }
# ────────────────────────────────────────────────────────────────────────────

# ======================= Main ==========================
def main():
    args = parse_args()
    data_path  = Path(args.data)
    ckpt_path  = Path(args.ckpt)
    label_path = Path(args.labels)
    s3_prefix  = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    pred_out   = args.pred_out

    s3_prefix_effective = (
        s3_join_folder(s3_prefix, f"fold{args.fold_id}")
        if args.fold_id is not None and args.fold_id >= 0
        else s3_prefix
    )
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
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        sd = state
    else:
        sd = state.get("model_state_dict", state)
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    model.load_state_dict(sd, strict=True)

    # ---------- Metric accumulators ----------
    scores       = defaultdict(lambda: {"y": [], "p": []})
    multi_scores = defaultdict(lambda: {"y": [], "p": []})
    length_note  = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    uid_to_pred_len = {}
    uid_to_lbl_len  = {}

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
                probs_seq_np = probs9[i, :T].detach().cpu().numpy()  # (T, 9)

                if pred_writer:
                    pred_writer.write(
                        json.dumps({"uid": uid, "probs": np.round(probs_seq_np, 6).tolist()}) + "\n"
                    )

                lbl_info = label_dict.get(uid)
                if lbl_info is None:
                    reject += 1
                    continue

                L_pred = len(probs_seq_np)
                L_lbl  = len(lbl_info["label"])
                uid_to_pred_len[uid] = L_pred
                uid_to_lbl_len[uid]  = L_lbl

                if L_pred != L_lbl:
                    length_note["pred>lbl" if L_pred > L_lbl else "pred<label"] += 1

                L          = min(L_pred, L_lbl)
                lbl_offset = L_lbl - L

                y_arr      = np.asarray(lbl_info["label"][lbl_offset : lbl_offset + L], dtype=np.int64)
                idx_h_arr  = np.asarray(lbl_info["idx_h"] [lbl_offset : lbl_offset + L], dtype=np.int64)
                feat_h_arr = np.asarray(lbl_info["feat_h"][lbl_offset : lbl_offset + L], dtype=np.int64)
                p_arr      = probs_seq_np[:L]   # (L, 9)

                group_idx = np.where(
                    feat_h_arr == 0, 0,
                    np.where((feat_h_arr == 1) & (idx_h_arr == 0), 1, 2)
                )

                split_tag = which_split(uid)
                accept_users.setdefault(split_tag, set()).add(uid)

                valid_mask = (y_arr >= 1) & (y_arr <= 9)
                for g_idx, g_name in enumerate(GROUP_ORDER):
                    mask = valid_mask & (group_idx == g_idx)
                    if not mask.any():
                        continue
                    mkey = (g_name, split_tag)
                    multi_scores[mkey]["y"].extend(y_arr[mask].tolist())
                    multi_scores[mkey]["p"].extend(p_arr[mask])

                for task, pos_classes in BIN_TASKS.items():
                    y_bin   = np.isin(y_arr, list(TASK_POSSETS[task])).astype(np.int8)
                    col_idx = [j - 1 for j in pos_classes]
                    p_bin   = p_arr[:, col_idx].sum(axis=1)

                    for g_idx, g_name in enumerate(GROUP_ORDER):
                        mask = (group_idx == g_idx)
                        if not mask.any():
                            continue
                        key = (task, g_name, split_tag)
                        scores[key]["y"].extend(y_bin[mask].tolist())
                        scores[key]["p"].extend(p_bin[mask].tolist())

                accept += 1

    # ── Debug output ───────────────────────────────────────────
    print("\n[DEBUG] group distribution in scored samples:")
    for grp in GROUP_ORDER:
        for spl in SPLIT_ORDER:
            n = len(scores[("BuyOne", grp, spl)]["y"])
            print(f"  BuyOne | {grp} | {spl}: {n} samples")

    print(f"\n[INFO] parsed: {accept} users accepted, {reject} users missing labels.")
    if length_note:
        print("[INFO] length mismatches:", dict(length_note))

    print("\n[DEBUG] pred>lbl breakdown by split:")
    for spl in ["val", "test", "train"]:
        users_in_split = accept_users.get(spl, set())
        n_mismatch = sum(
            1 for uid in users_in_split
            if uid_to_pred_len.get(uid, 0) > uid_to_lbl_len.get(uid, 0)
        )
        print(f"  {spl}: {n_mismatch} mismatch / {len(users_in_split)} total users")

    excess = [
        uid_to_pred_len[uid] - uid_to_lbl_len[uid]
        for uid in uid_to_pred_len
        if uid_to_pred_len[uid] > uid_to_lbl_len.get(uid, 0)
    ]
    if excess:
        import statistics
        print(f"[DEBUG] excess slots — min={min(excess)} max={max(excess)} "
              f"mean={statistics.mean(excess):.1f} median={statistics.median(excess):.1f}")

    if pred_writer:
        pred_writer.__exit__(None, None, None)

    if args.uids_val and args.uids_test:
        print(f"[INFO] coverage: val={len(accept_users.get('val', set()))} / {len(load_uid_set(args.uids_val))}, "
              f"test={len(accept_users.get('test', set()))} / {len(load_uid_set(args.uids_test))}")

    # ── Bucket stats & prevalence (unchanged) ─────────────────
    bucket_stats = []
    for task in BIN_TASKS:
        for grp in GROUP_ORDER:
            for spl in SPLIT_ORDER:
                y = scores[(task, grp, spl)]["y"]
                if not y:
                    continue
                n = len(y)
                pos = sum(y)
                neg = n - pos
                prev = pos / n if n else float('nan')
                bucket_stats.append({
                    "Task": task, "Group": grp, "Split": spl,
                    "N": n, "Pos": pos, "Neg": neg, "Prev": round(prev, 4)
                })

    stats_df = pd.DataFrame(bucket_stats).sort_values(["Split", "Group", "Task"])
    print("\n=============  BUCKET SIZES & PREVALENCE  =======================")
    print(stats_df.to_string(index=False))
    print("============================================================")

    # ---------- Compute binary task tables (unchanged) ----------
    rows = []
    for task in BIN_TASKS:
        for grp in GROUP_ORDER:
            for spl in SPLIT_ORDER:
                y, p = scores[(task, grp, spl)]["y"], scores[(task, grp, spl)]["p"]
                if not y:
                    continue
                if len(set(y)) < 2:
                    auc = np.nan
                    auprc = np.nan
                else:
                    auc   = roc_auc_score(y, p)
                    auprc = average_precision_score(y, p)
                rows.append({
                    "Task": task, "Group": grp, "Split": spl,
                    "AUC": auc, "AUPRC": auprc,
                })

    metrics = pd.DataFrame(rows)

    # ── CHANGED: compute all 6 multiclass metrics per (group, split) ─────────
    multi_rows = []
    for grp in GROUP_ORDER:
        for spl in SPLIT_ORDER:
            y = multi_scores[(grp, spl)]["y"]
            p = multi_scores[(grp, spl)]["p"]
            if not y:
                continue

            y_arr = np.asarray(y, dtype=np.int64)
            p_arr = np.vstack(p)                    # (N, 9)

            m = compute_multiclass_metrics(y_arr, p_arr)
            multi_rows.append({
                "Group":        grp,
                "Split":        spl,
                "MacroOvR_AUC": m["MacroOvR_AUC"],
                "MacroAUPRC":   m["MacroAUPRC"],
                "MacroF1":      m["MacroF1"],
                "MCC":          m["MCC"],
                "LogLoss":      m["LogLoss"],
                "Top2Acc":      m["Top2Acc"],
            })

    multiclass_metrics = pd.DataFrame(multi_rows)
    # ────────────────────────────────────────────────────────────────────────

    # ── CHANGED: pivot tables now cover all 6 metrics ────────────────────────
    MULTI_METRICS = ["MacroOvR_AUC", "MacroAUPRC", "MacroF1", "MCC", "LogLoss", "Top2Acc"]

    paper_multi_tbl = (
        multiclass_metrics
        .pivot(index="Group", columns="Split", values=MULTI_METRICS)
        .reindex(index=GROUP_ORDER)
        .reindex(columns=pd.MultiIndex.from_product([MULTI_METRICS, SPLIT_ORDER]))
        .round(4)
    )

    def make_multi_panel(group_name: str) -> pd.DataFrame:
        sub = multiclass_metrics[multiclass_metrics["Group"] == group_name]
        return (
            sub.pivot(index="Group", columns="Split", values=MULTI_METRICS)
            .reindex(columns=pd.MultiIndex.from_product([MULTI_METRICS, SPLIT_ORDER]))
            .round(4)
        )

    multi_calibration_tbl = make_multi_panel("Calibration")
    multi_holdoutA_tbl    = make_multi_panel("HoldoutA")
    multi_holdoutB_tbl    = make_multi_panel("HoldoutB")
    # ────────────────────────────────────────────────────────────────────────

    # Binary AUC panel helpers (unchanged)
    paper_auc = metrics[metrics["Task"].isin(PAPER_AUC_TASKS.keys())].copy()
    paper_auc["TaskPretty"] = paper_auc["Task"].map(PAPER_AUC_TASKS)

    def make_auc_panel(group_name: str) -> pd.DataFrame:
        sub = paper_auc[paper_auc["Group"] == group_name].copy()
        return (
            sub.pivot(index="TaskPretty", columns="Split", values="AUC")
            .reindex(index=PAPER_AUC_ORDER, columns=SPLIT_ORDER)
            .round(4)
        )

    auc_calibration_tbl = make_auc_panel("Calibration")
    auc_holdoutA_tbl    = make_auc_panel("HoldoutA")
    auc_holdoutB_tbl    = make_auc_panel("HoldoutB")

    # ── CHANGED: updated print titles ────────────────────────────────────────
    def _p(title: str, df: pd.DataFrame):
        print(f"\n=============  {title}  =======================")
        print(df.fillna(" NA"))
        print("============================================================")

    _p("9-CLASS METRICS (ALL GROUPS)", paper_multi_tbl)
    _p("9-CLASS METRICS - CALIBRATION",  multi_calibration_tbl)
    _p("9-CLASS METRICS - HOLDOUT A",    multi_holdoutA_tbl)
    _p("9-CLASS METRICS - HOLDOUT B",    multi_holdoutB_tbl)
    _p("SELECTED BINARY AUC TABLE - CALIBRATION", auc_calibration_tbl)
    _p("SELECTED BINARY AUC TABLE - HOLDOUT A", auc_holdoutA_tbl)
    _p("SELECTED BINARY AUC TABLE - HOLDOUT B", auc_holdoutB_tbl)
    # ────────────────────────────────────────────────────────────────────────

    # ---------- Save locally & upload to S3 ----------
    out_dir = Path("/tmp/predict_eval_outputs_gru")
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_multi_csv       = out_dir / "paper_multiclass_table.csv"
    multi_calibration_csv = out_dir / "paper_multi_calibration.csv"
    multi_holdoutA_csv    = out_dir / "paper_multi_holdoutA.csv"
    multi_holdoutB_csv    = out_dir / "paper_multi_holdoutB.csv"
    auc_calibration_csv   = out_dir / "paper_auc_calibration.csv"
    auc_holdoutA_csv      = out_dir / "paper_auc_holdoutA.csv"
    auc_holdoutB_csv      = out_dir / "paper_auc_holdoutB.csv"

    paper_multi_tbl.to_csv(paper_multi_csv)
    multi_calibration_tbl.to_csv(multi_calibration_csv)
    multi_holdoutA_tbl.to_csv(multi_holdoutA_csv)
    multi_holdoutB_tbl.to_csv(multi_holdoutB_csv)
    auc_calibration_tbl.to_csv(auc_calibration_csv)
    auc_holdoutA_tbl.to_csv(auc_holdoutA_csv)
    auc_holdoutB_tbl.to_csv(auc_holdoutB_csv)

    for pth in [paper_multi_csv, multi_calibration_csv, multi_holdoutA_csv,
                multi_holdoutB_csv, auc_calibration_csv, auc_holdoutA_csv,
                auc_holdoutB_csv]:
        dest = s3_join(s3_prefix_effective, pth.name)
        s3_upload_file(pth, dest)
        print(f"[S3] uploaded: {dest}")

    if pred_out:
        dest = s3_join(s3_prefix_effective, Path(pred_out).name)
        s3_upload_file(pred_out, dest)
        print(f"[S3] uploaded: {dest}")


if __name__ == "__main__":
    main()