#!/usr/bin/env python3
"""
predict_productgpt_and_eval.py

End-to-end:
  - Run inference for 9-way decision probabilities (every ai_rate steps)
  - Compute AUC / Hit / F1 / AUPRC by (Task, PeriodGroup, Split)
  - Print AUC table to console
  - Save CSV tables (and optional predictions) locally and upload to S3

Usage (example):
python3 predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /tmp/FullProductGPT_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/evals/best_calibrated_$(date +%F_%H%M%S)/ \
  --pred-out /tmp/preds_phaseB.jsonl.gz \
  --uids-val s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id 0

Notes:
  - Requires IAM role or AWS creds for S3 upload.
  - Prints the AUC table to stdout on the AWS server.
  - If --uids-val and --uids-test are supplied (local path or s3://...), the script
    will EXACT MATCH those users for 'val' and 'test' splits and will NOT do 80/10/10.
  - If --fold-id is provided, all uploaded outputs go under .../fold{ID}/ on S3.
"""

from __future__ import annotations
import argparse, json, gzip, os, sys, subprocess
from contextlib import nullcontext
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set
from dataset4_productgpt import load_json_dataset

import numpy as np
import pandas as pd
import torch
import re

import math
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer

# ── CHANGED: added matthews_corrcoef, log_loss, label_binarize ──────────────
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
    matthews_corrcoef, log_loss,
)
from sklearn.preprocessing import label_binarize
# ────────────────────────────────────────────────────────────────────────────

def parse_hp_from_ckpt_name(ckpt_path: Path) -> dict:
    """
    Parses:
      FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr..._w4_fold0.pt
    Returns dict with nb_features, d_model, d_ff, N, num_heads, weight.
    """
    name = ckpt_path.name
    m = re.search(
        r"performerfeatures(?P<nb>\d+)_dmodel(?P<dm>\d+)_ff(?P<ff>\d+)_N(?P<N>\d+)_heads(?P<h>\d+)_lr(?P<lr>[\deE\.\-]+)_w(?P<w>\d+)_fold(?P<fold>\d+)",
        name
    )
    if not m:
        raise ValueError(f"Cannot parse HPs from ckpt filename: {name}")
    d = m.groupdict()
    return {
        "nb_features": int(d["nb"]),
        "d_model": int(d["dm"]),
        "d_ff": int(d["ff"]),
        "N": int(d["N"]),
        "num_heads": int(d["h"]),
        "weight": int(d["w"]),
        "fold_id": int(d["fold"]),
    }
# --- Project imports (must exist in your repo) ---
from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer
from train1_decision_only_performer_aws import _ensure_jsonl, JsonLineDataset, _build_tok

# Optional: silence Intel/LLVM OpenMP clash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ══════════════════════════ VectorScaling (must match training) ═══════
class VectorScaling(torch.nn.Module):
    """Post-hoc calibration: per-class scale and shift."""
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(n_classes))
        self.b = torch.nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits_dec: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.a * logits_dec + self.b, dim=-1)

def load_calibrator(ckpt_path: Path, device: torch.device) -> VectorScaling | None:
    """
    Look for a calibrator file alongside the checkpoint.
    Expected name: calibrator_<same_uid>.pt in the same directory.
    """
    ckpt_dir = ckpt_path.parent
    uid_part = ckpt_path.stem.replace("FullProductGPT_", "")

    cal_path = ckpt_dir / f"calibrator_{uid_part}.pt"
    if not cal_path.exists():
        print(f"[INFO] No calibrator found at {cal_path}")
        return None

    cal = VectorScaling(n_classes=9).to(device)
    state = torch.load(cal_path, map_location=device)
    cal.a.data = state["a"].to(device)
    cal.b.data = state["b"].to(device)
    print(f"[INFO] Loaded calibrator from {cal_path}")
    print(f"[INFO]   a = {cal.a.data.cpu().numpy().round(4)}")
    print(f"[INFO]   b = {cal.b.data.cpu().numpy().round(4)}")
    return cal

def collate_fn(pad_id: int):
    def _inner(batch):
        uids = [b["uid"] for b in batch]
        lens = [len(b["x"]) for b in batch]
        Lmax = max(lens)
        X    = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        for i,(item,L) in enumerate(zip(batch,lens)):
            X[i,:L] = item["x"]
        return {"uid": uids, "x": X, "lens": lens}
    return _inner

# ===================== CLI ======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, help="ND-JSON events file (1 line per user)")
    p.add_argument("--ckpt",   required=True, help="*.pt checkpoint path")
    p.add_argument("--labels", required=True, help="JSON label file (clean_list_int_wide4_simple6.json)")
    p.add_argument("--s3",     required=True, help="S3 URI prefix (e.g., s3://bucket/folder/)")
    p.add_argument("--pred-out", default="",  help="Optional: local predictions path (.jsonl or .jsonl.gz)")
    p.add_argument("--feat-xlsx", default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx",
                   help="Feature Excel path for product embeddings")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--ai-rate", type=int, default=15, help="Stride for decision positions")
    p.add_argument("--thresh", type=float, default=0.5, help="Threshold for Hit/F1")
    p.add_argument("--seed",   type=int, default=33, help="Reproduce 80/10/10 split when no UID files are provided")

    p.add_argument("--uids-val",  default="", help="Text file (or s3://...) with validation UIDs, one per line")
    p.add_argument("--uids-test", default="", help="Text file (or s3://...) with test UIDs, one per line")

    p.add_argument("--fold-spec", default="s3://productgptbucket/folds/productgptfolds.json",
                   help="Fold assignment JSON (same as training SPEC_URI)")
    p.add_argument("--split-from", default="",
                   help="Dataset path to reproduce split (default: cfg['filepath'])")
    p.add_argument("--split-seed", type=int, default=33,
                   help="Seed used by training random_split (Phase-B default=33)")
    p.add_argument("--split-data-frac", type=float, default=1.0,
                   help="Match training data_frac (Phase-B=1.0; Phase-A might be 0.25/0.05)")
    p.add_argument("--split-subsample-seed", type=int, default=33,
                   help="Match training subsample_seed (default=33)")
    p.add_argument("--dump-uids", default="",
                   help="Optional: local folder to write uids_val.txt / uids_test.txt")
    p.add_argument("--calibration", choices=["calibrator", "analytic", "none"], default="none")

    p.add_argument("--fold-id", type=int, default=-1, help="If >=0, upload outputs under .../fold{ID}/")
    return p.parse_args()

# ================== Utilities ===================
def smart_open_w(path: str | Path):
    """stdout if '-', gzip if *.gz, else normal text file (write)."""
    if isinstance(path, Path):
        path = str(path)
    if path == "-":
        return nullcontext(sys.stdout)
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")

# --- S3 helpers ---
def parse_s3_uri(uri: str):
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
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + filename

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
    if path_or_s3.startswith("s3://"):
        text = s3_read_text(path_or_s3)
    else:
        text = Path(path_or_s3).read_text()
    uids = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        uids.add(line)
    return uids

def load_json_from_local_or_s3(path_or_s3: str) -> dict:
    if path_or_s3.startswith("s3://"):
        return json.loads(s3_read_text(path_or_s3))
    return json.loads(Path(path_or_s3).read_text())

def deterministic_subsample_indices(n: int, frac: float, seed: int) -> set[int]:
    if frac >= 1.0:
        return set(range(n))
    k = max(1, int(n * frac))
    rng = __import__("random").Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return set(idx[:k])

def phase_split_uids_exact(
    *,
    fold_id: int,
    fold_spec_uri: str,
    split_from_path: str,
    split_seed: int,
    data_frac: float,
    subsample_seed: int,
) -> tuple[set[str], set[str]]:
    spec = load_json_from_local_or_s3(fold_spec_uri)

    uids_test_fold = [u for u, f in spec["assignment"].items() if f == fold_id]
    uids_trainval  = [u for u in spec["assignment"] if u not in set(uids_test_fold)]
    if not uids_trainval:
        raise ValueError(f"No uids_trainval found for fold_id={fold_id}")

    raw = load_json_dataset(split_from_path, keep_uids=set(uids_trainval))

    keep = deterministic_subsample_indices(len(raw), data_frac, subsample_seed)
    if len(keep) != len(raw):
        raw = [raw[i] for i in range(len(raw)) if i in keep]

    n = len(raw)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    if n_val <= 0 or n_test <= 0:
        raise ValueError(f"Not enough data to split: n={n} -> {n_train}/{n_val}/{n_test}")

    g = torch.Generator().manual_seed(split_seed)
    _, va_i, te_i = random_split(range(n), [n_train, n_val, n_test], generator=g)

    val_uids  = {flat_uid(raw[i]["uid"]) for i in va_i.indices}
    test_uids = {flat_uid(raw[i]["uid"]) for i in te_i.indices}

    overlap = val_uids & test_uids
    if overlap:
        raise RuntimeError(f"Split overlap (should not happen): {list(overlap)[:5]}")

    return val_uids, test_uids

# ================= Data / Labels =================
def to_int_vec(x):
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(int(v) if isinstance(item, str) else item for v in str(item).split())
        return out
    raise TypeError(type(x))

def flat_uid(u):
    return str(u[0] if isinstance(u, list) else u)

def load_labels(label_path: Path) -> Dict[str, Dict[str, List[int]]]:
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

# ================= Feature tensor ================
FEATURE_COLS = [
    "Rarity","MaxLife","MaxOffense","MaxDefense",
    "WeaponTypeOneHandSword","WeaponTypeTwoHandSword","WeaponTypeArrow","WeaponTypeMagic","WeaponTypePolearm",
    "EthnicityIce","EthnicityRock","EthnicityWater","EthnicityFire","EthnicityThunder","EthnicityWind",
    "GenderFemale","GenderMale","CountryRuiYue","CountryDaoQi","CountryZhiDong","CountryMengDe",
    "type_figure","MinimumAttack","MaximumAttack","MinSpecialEffect","MaxSpecialEffect","SpecialEffectEfficiency",
    "SpecialEffectExpertise","SpecialEffectAttack","SpecialEffectSuper","SpecialEffectRatio","SpecialEffectPhysical",
    "SpecialEffectLife","LTO",
]
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
UNK_PROD_ID = 59
MAX_TOKEN_ID = UNK_PROD_ID

def load_feature_tensor(xls_path: Path) -> torch.Tensor:
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)

# ================= Dataset & Collate =============
class PredictDataset(JsonLineDataset):
    def __init__(self, path, pad_id: int):
        super().__init__(path)
        self.pad_id = pad_id
    def to_int_or_pad(self, tok: str) -> int:
        try:
            return int(tok)
        except ValueError:
            return self.pad_id
    def __getitem__(self, idx):
        row     = super().__getitem__(idx)
        seq_raw = row["AggregateInput"]
        if isinstance(seq_raw, list):
            if len(seq_raw) == 1 and isinstance(seq_raw[0], str):
                seq_str = seq_raw[0]
            else:
                seq_str = " ".join(map(str, seq_raw))
        else:
            seq_str = str(seq_raw)
        toks  = [self.to_int_or_pad(t) for t in seq_str.strip().split()]
        uid   = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": flat_uid(uid), "x": torch.tensor(toks, dtype=torch.long)}

# =================== Metrics Setup ===============
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
    # roc_auc_score handles multi-class natively with multi_class='ovr'.
    # Falls back to NaN if any class is entirely absent from y_arr.
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
            # Class entirely absent: skip to avoid ill-defined AP
            continue
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
    # Works natively for multi-class; returns scalar in [-1, 1].
    mcc = matthews_corrcoef(y_arr, y_hat)

    # ── 5. Log-Loss (Cross-Entropy) ───────────────────────────────────────
    # Validates calibration quality; lower is better.
    # Clip probabilities to avoid log(0).
    p_clipped = np.clip(p_arr, 1e-7, 1.0)
    ll = log_loss(y_arr, p_clipped, labels=NINE_CLASSES)

    # ── 6. Top-2 Accuracy ─────────────────────────────────────────────────
    # True if the ground-truth class appears among the top-2 predicted classes.
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

# ======================= Main ====================
def main():
    args = parse_args()
    data_path   = _ensure_jsonl(args.data)
    ckpt_path   = Path(args.ckpt)
    hp = parse_hp_from_ckpt_name(ckpt_path)
    label_path  = Path(args.labels)
    feat_path   = Path(args.feat_xlsx)
    s3_prefix   = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    pred_out    = args.pred_out

    if args.fold_id is not None and args.fold_id >= 0:
        s3_prefix_effective = s3_join_folder(s3_prefix, f"fold{args.fold_id}")
    else:
        s3_prefix_effective = s3_prefix
    print(f"[INFO] S3 upload prefix: {s3_prefix_effective}")

    # ---------- Config ----------
    cfg = get_config()
    cfg["ai_rate"]    = args.ai_rate
    cfg["batch_size"] = args.batch_size

    # ---------- Tokenizer / PAD ----------
    tok_path = Path(cfg["model_folder"]) / "tokenizer_tgt.json"
    tok_tgt  = (Tokenizer.from_file(str(tok_path)) if tok_path.exists()
                else _build_tok())
    pad_id   = tok_tgt.token_to_id("[PAD]")
    SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
    EOS_PROD_ID, SOS_PROD_ID          = 57, 58
    SPECIAL_IDS = [pad_id, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

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
        fold_for_split = args.fold_id if (args.fold_id is not None and args.fold_id >= 0) else hp["fold_id"]
        split_from_path = args.split_from.strip() or cfg["filepath"]

        uids_val_override, uids_test_override = phase_split_uids_exact(
            fold_id=fold_for_split,
            fold_spec_uri=args.fold_spec,
            split_from_path=split_from_path,
            split_seed=args.split_seed,
            data_frac=args.split_data_frac,
            subsample_seed=args.split_subsample_seed,
        )

        def which_split(u):
            return "val" if u in uids_val_override else "test" if u in uids_test_override else "train"

        print(
            f"[INFO] Using EXACT training split via fold-spec: "
            f"fold={fold_for_split}, val={len(uids_val_override)}, "
            f"test={len(uids_test_override)}, seed={args.split_seed}, "
            f"data_frac={args.split_data_frac}"
        )

        if args.dump_uids:
            out = Path(args.dump_uids)
            out.mkdir(parents=True, exist_ok=True)
            (out / "uids_val.txt").write_text("\n".join(sorted(uids_val_override)) + "\n")
            (out / "uids_test.txt").write_text("\n".join(sorted(uids_test_override)) + "\n")
            print(f"[INFO] Wrote UID lists to: {out}/uids_val.txt and {out}/uids_test.txt")

    # ---------- DataLoader ----------
    ds = PredictDataset(data_path, pad_id=pad_id)
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn(pad_id)
    )

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["seq_len_ai"] = cfg["seq_len_tgt"] * cfg["ai_rate"]

    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_ai"],
        d_model=hp["d_model"],
        n_layers=hp["N"],
        n_heads=hp["num_heads"],
        d_ff=hp["d_ff"],
        dropout=0.0,
        nb_features=hp["nb_features"],
        kernel_type=cfg["kernel_type"],
        feature_tensor=load_feature_tensor(feat_path),
        special_token_ids=SPECIAL_IDS,
    ).to(device).eval()

    weight_class9 = hp["weight"]

    # ---------- Load checkpoint ----------
    def clean_state_dict(raw):
        def strip_prefix(k): return k[7:] if k.startswith("module.") else k
        ignore = ("weight_fake_quant", "activation_post_process")
        return {strip_prefix(k): v for k, v in raw.items()
                if not any(tok in k for tok in ignore)}
    torch.cuda.empty_cache()
    state = torch.load(ckpt_path, map_location=device)
    raw_sd = state["model_state_dict"] if "model_state_dict" in state else \
             state["module"]           if "module" in state           else state
    model.load_state_dict(clean_state_dict(raw_sd), strict=True)

    calibrator = None
    logit_bias_9 = None

    if args.calibration == "calibrator":
        calibrator = load_calibrator(ckpt_path, device)
        if calibrator is None:
            print("[WARN] No calibrator found, falling back to analytic correction")
            args.calibration = "analytic"

    if args.calibration == "analytic":
        if "logit_bias_9" in state:
            logit_bias_9 = torch.tensor(state["logit_bias_9"], device=device, dtype=torch.float32)
            print(f"[INFO] Loaded per-class logit_bias_9 from checkpoint: {logit_bias_9.cpu().numpy().round(4)}")
        else:
            weight_class9 = hp["weight"]
            logit_bias_9 = torch.zeros(9, device=device, dtype=torch.float32)
            if weight_class9 != 1:
                logit_bias_9[8] = math.log(weight_class9)
            print(f"[INFO] Constructed logit_bias_9 from weight={weight_class9}: {logit_bias_9.cpu().numpy().round(4)}")

    if args.calibration == "none":
        print("[INFO] No calibration applied (raw model probabilities)")

    print(f"[INFO] Calibration method: {args.calibration}")

    # ---------- Metric accumulators ----------
    scores       = defaultdict(lambda: {"y": [], "p": []})
    multi_scores = defaultdict(lambda: {"y": [], "p": []})
    length_note  = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    pred_writer = None
    if pred_out:
        pred_writer = smart_open_w(pred_out)

    uid_to_pred_len = {}
    uid_to_lbl_len  = {}

    # ---------- Inference + streaming eval ----------
    focus_ids = torch.arange(1, 10, device=device)
    with torch.no_grad():
        for batch in loader:
            x    = batch["x"].to(device)
            uids = batch["uid"]
            lens = batch["lens"]
            logits_full = model(x)

            if x.size(1) < cfg["ai_rate"]:
                pos = torch.empty((0,), dtype=torch.long, device=device)
            else:
                pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)

            if logits_full.size(1) == x.size(1):
                logits = logits_full[:, pos, :]
            else:
                logits = logits_full

            V_out = logits.size(-1)

            if V_out == 9:
                prob_dec_9 = torch.softmax(logits, dim=-1)
            else:
                probs_all = torch.softmax(logits, dim=-1)
                prob_dec_9 = probs_all[..., 1:10]

            prob_dec_focus = prob_dec_9

            for i, uid in enumerate(uids):
                actual_len_i   = int((x[i] != pad_id).sum().item())
                n_valid_slots_i = actual_len_i // cfg["ai_rate"]
                probs_seq_np   = prob_dec_focus[i, :n_valid_slots_i].detach().cpu().numpy()

                if pred_writer:
                    pred_writer.write(
                        json.dumps({"uid": uid, "probs": np.round(probs_seq_np, 6).tolist()}) + "\n"
                    )

                lbl_info = label_dict.get(uid)
                if lbl_info is None:
                    reject += 1
                    continue

            for i, uid in enumerate(uids):

                actual_len_i    = lens[i]
                n_valid_slots_i = actual_len_i // cfg["ai_rate"]
                probs_seq_np    = prob_dec_focus[i, :n_valid_slots_i].detach().cpu().numpy()

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
                p_arr      = probs_seq_np[:L]

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
                    y_bin  = np.isin(y_arr, list(TASK_POSSETS[task])).astype(np.int8)
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

    print("\n[DEBUG] group distribution in scored samples:")
    for grp in GROUP_ORDER:
        for spl in ["val", "test"]:
            n = len(scores[("BuyOne", grp, spl)]["y"])
            print(f"  BuyOne | {grp} | {spl}: {n} samples")

    print(f"[INFO] parsed: {accept} users accepted, {reject} users missing labels.")
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

    print(f"[INFO] parsed: {accept} users accepted, {reject} users missing labels.")
    if length_note:
        print("[INFO] length mismatches:", dict(length_note))

    if args.uids_val and args.uids_test:
        print(f"[INFO] coverage: val={len(accept_users.get('val', set()))} / {len(load_uid_set(args.uids_val))}, "
              f"test={len(accept_users.get('test', set()))} / {len(load_uid_set(args.uids_test))}")

    # ── CHANGED: bucket stats (unchanged) ────────────────────────────────────
    bucket_stats = []
    for task in BIN_TASKS:
        for grp in ["Calibration","HoldoutA","HoldoutB"]:
            for spl in ["val","test"]:
                y = scores[(task, grp, spl)]["y"]
                p = scores[(task, grp, spl)]["p"]
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

    stats_df = pd.DataFrame(bucket_stats).sort_values(["Split","Group","Task"])
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
                    "Task": task,
                    "Group": grp,
                    "Split": spl,
                    "AUC": auc,
                    "AUPRC": auprc
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

    # ── CHANGED: pivot table and panel helper now cover all 6 metrics ────────
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
        tbl = (
            sub.pivot(index="Group", columns="Split", values=MULTI_METRICS)
            .reindex(columns=pd.MultiIndex.from_product([MULTI_METRICS, SPLIT_ORDER]))
            .round(4)
        )
        return tbl

    multi_calibration_tbl = make_multi_panel("Calibration")
    multi_holdoutA_tbl    = make_multi_panel("HoldoutA")
    multi_holdoutB_tbl    = make_multi_panel("HoldoutB")
    # ────────────────────────────────────────────────────────────────────────

    # Binary AUC panel helpers (unchanged)
    paper_auc = metrics[metrics["Task"].isin(PAPER_AUC_TASKS.keys())].copy()
    paper_auc["TaskPretty"] = paper_auc["Task"].map(PAPER_AUC_TASKS)

    def make_auc_panel(group_name: str) -> pd.DataFrame:
        sub = paper_auc[paper_auc["Group"] == group_name].copy()
        tbl = (
            sub.pivot(index="TaskPretty", columns="Split", values="AUC")
            .reindex(index=PAPER_AUC_ORDER, columns=SPLIT_ORDER)
            .round(4)
        )
        return tbl

    auc_calibration_tbl = make_auc_panel("Calibration")
    auc_holdoutA_tbl    = make_auc_panel("HoldoutA")
    auc_holdoutB_tbl    = make_auc_panel("HoldoutB")

    # ---------- Print tables ----------
    def _p(title: str, df: pd.DataFrame):
        print(f"\n=============  {title}  =======================")
        print(df.fillna(" NA"))
        print("============================================================")

    # ── CHANGED: updated print titles ────────────────────────────────────────
    _p("9-CLASS METRICS (ALL GROUPS)", paper_multi_tbl)
    _p("9-CLASS METRICS - CALIBRATION",  multi_calibration_tbl)
    _p("9-CLASS METRICS - HOLDOUT A",    multi_holdoutA_tbl)
    _p("9-CLASS METRICS - HOLDOUT B",    multi_holdoutB_tbl)
    _p("SELECTED BINARY AUC TABLE - CALIBRATION", auc_calibration_tbl)
    _p("SELECTED BINARY AUC TABLE - HOLDOUT A", auc_holdoutA_tbl)
    _p("SELECTED BINARY AUC TABLE - HOLDOUT B", auc_holdoutB_tbl)
    # ────────────────────────────────────────────────────────────────────────

    # ---------- Save locally & upload to S3 ----------
    out_dir = Path("/tmp/predict_eval_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_multi_csv        = out_dir / "paper_multiclass_table.csv"
    multi_calibration_csv  = out_dir / "paper_multi_calibration.csv"
    multi_holdoutA_csv     = out_dir / "paper_multi_holdoutA.csv"
    multi_holdoutB_csv     = out_dir / "paper_multi_holdoutB.csv"
    auc_calibration_csv    = out_dir / "paper_auc_calibration.csv"
    auc_holdoutA_csv       = out_dir / "paper_auc_holdoutA.csv"
    auc_holdoutB_csv       = out_dir / "paper_auc_holdoutB.csv"

    paper_multi_tbl.to_csv(paper_multi_csv)
    multi_calibration_tbl.to_csv(multi_calibration_csv)
    multi_holdoutA_tbl.to_csv(multi_holdoutA_csv)
    multi_holdoutB_tbl.to_csv(multi_holdoutB_csv)
    auc_calibration_tbl.to_csv(auc_calibration_csv)
    auc_holdoutA_tbl.to_csv(auc_holdoutA_csv)
    auc_holdoutB_tbl.to_csv(auc_holdoutB_csv)

    for pth in [multi_calibration_csv, multi_holdoutA_csv, multi_holdoutB_csv,
            paper_multi_csv, auc_calibration_csv, auc_holdoutA_csv, auc_holdoutB_csv]:
        dest = s3_join(s3_prefix_effective, pth.name)
        s3_upload_file(pth, dest)
        print(f"[S3] uploaded: {dest}")

    if pred_out:
        dest = s3_join(s3_prefix_effective, Path(pred_out).name)
        s3_upload_file(pred_out, dest)
        print(f"[S3] uploaded: {dest}")

if __name__ == "__main__":
    main()