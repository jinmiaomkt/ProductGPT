#!/usr/bin/env python3
"""
predict_mixture2_and_eval.py

End-to-end:
  - Run inference for 9-way decision probabilities (every ai_rate steps)
  - For seen training users: use exact user-specific gate
  - For unseen users: use mean gate from training users
  - Compute AUC / Hit / F1 / AUPRC by (Task, PeriodGroup, Split)
  - Tune threshold on validation (by Task x Group), then apply to both val/test
  - Print tables to console
  - Save CSV tables (and optional predictions) locally and upload to S3
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split

# --- Project imports (must exist in your repo) ---
from config4 import get_config
from dataset4_productgpt import load_json_dataset
from model4_mixture2_decoderonly_feature_performer import build_transformer
from train1_decision_only_performer_aws import JsonLineDataset, _build_tok, _ensure_jsonl

# Optional: silence Intel/LLVM OpenMP clash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ═══════════════════════════════════════════════════════════════
# Calibration helper
# ═══════════════════════════════════════════════════════════════
class VectorScaling(torch.nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(n_classes))
        self.b = torch.nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits_dec: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.a * logits_dec + self.b, dim=-1)


def load_calibrator(ckpt_path: Path, device: torch.device):
    uid_part = ckpt_path.stem.replace("FullProductGPT_", "")
    cal_path = ckpt_path.parent / f"calibrator_{uid_part}.pt"
    if not cal_path.exists():
        print(f"[INFO] No calibrator at {cal_path}")
        return None

    cal = VectorScaling(9).to(device)
    state = torch.load(cal_path, map_location=device)
    cal.a.data = state["a"].to(device)
    cal.b.data = state["b"].to(device)
    print(
        f"[INFO] Loaded calibrator: "
        f"a={cal.a.data.cpu().numpy().round(4)}, "
        f"b={cal.b.data.cpu().numpy().round(4)}"
    )
    return cal


# ═══════════════════════════════════════════════════════════════
# Checkpoint-name parser
# ═══════════════════════════════════════════════════════════════
def parse_hp_from_ckpt_name(ckpt_path: Path) -> dict:
    """
    Parses names like:
      FullProductGPT_mixture2_performerfeatures32_dmodel64_ff192_N3_heads4_lr0.0002_w2_fold0.pt
    """
    name = ckpt_path.name
    m = re.search(
        r"performerfeatures(?P<nb>\d+)_dmodel(?P<dm>\d+)_ff(?P<ff>\d+)_N(?P<N>\d+)_heads(?P<h>\d+)_lr(?P<lr>[\deE\.\-]+)_w(?P<w>\d+)_fold(?P<fold>\d+)",
        name,
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


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="ND-JSON events file (1 line per user)")
    p.add_argument("--ckpt", required=True, help="*.pt checkpoint path")
    p.add_argument("--labels", required=True, help="JSON label file")
    p.add_argument("--s3", required=True, help="S3 URI prefix, e.g. s3://bucket/folder/")
    p.add_argument("--pred-out", default="", help="Optional local predictions path (.jsonl or .jsonl.gz)")
    p.add_argument(
        "--feat-xlsx",
        default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx",
        help="Feature Excel path for product embeddings",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--ai-rate", type=int, default=15, help="Stride for decision positions")
    p.add_argument("--thresh", type=float, default=0.5, help="Fallback threshold if val-tuning fails")
    p.add_argument("--seed", type=int, default=33, help="Seed for 80/10/10 split if no UID files provided")

    p.add_argument(
        "--fold-spec",
        default="s3://productgptbucket/folds/productgptfolds.json",
        help="Same SPEC_URI used by training",
    )
    p.add_argument(
        "--train-file",
        default="",
        help="Training json used to reproduce training UID mapping (recommended)",
    )

    p.add_argument("--calibration", choices=["calibrator", "analytic", "none"], default="none")

    p.add_argument("--uids-val", default="", help="Text file (or s3://...) with validation UIDs")
    p.add_argument("--uids-test", default="", help="Text file (or s3://...) with test UIDs")

    p.add_argument("--fold-id", type=int, default=-1, help="If >=0, upload outputs under .../fold{ID}/")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Generic helpers
# ═══════════════════════════════════════════════════════════════
def smart_open_w(path: str | Path):
    if isinstance(path, Path):
        path = str(path)
    if path == "-":
        return nullcontext(sys.stdout)
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def normalize_uid(uid) -> str:
    if uid is None:
        return ""
    if isinstance(uid, (list, tuple, set)):
        vals = [str(x) for x in uid if x is not None]
        return "|".join(sorted(vals)) if vals else ""
    return str(uid)


def flat_uid(u):
    return str(u[0] if isinstance(u, list) else u)


def to_int_vec(x):
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(int(v) if isinstance(item, str) else item for v in str(item).split())
        return out
    raise TypeError(type(x))


# ═══════════════════════════════════════════════════════════════
# S3 helpers
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# Split helpers
# ═══════════════════════════════════════════════════════════════
def load_labels(label_path: Path) -> Tuple[Dict[str, Dict[str, List[int]]], list]:
    raw = json.loads(label_path.read_text())
    records = list(raw) if isinstance(raw, list) else [{k: raw[k][i] for k in raw} for i in range(len(raw["uid"]))]
    label_dict = {
        flat_uid(rec["uid"]): {
            "label": to_int_vec(rec["Decision"]),
            "idx_h": to_int_vec(rec["IndexBasedHoldout"]),
            "feat_h": to_int_vec(rec["FeatureBasedHoldout"]),
        }
        for rec in records
    }
    return label_dict, records


def build_splits(records, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(records)
    tr, va = int(0.8 * n), int(0.1 * n)
    tr_i, va_i, te_i = random_split(range(n), [tr, va, n - tr - va], generator=g)
    val_uid = {flat_uid(records[i]["uid"]) for i in va_i.indices}
    test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}

    def which_split(u):
        return "val" if u in val_uid else "test" if u in test_uid else "train"

    return which_split


# ═══════════════════════════════════════════════════════════════
# Feature tensor
# ═══════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "Rarity",
    "MaxLife",
    "MaxOffense",
    "MaxDefense",
    "WeaponTypeOneHandSword",
    "WeaponTypeTwoHandSword",
    "WeaponTypeArrow",
    "WeaponTypeMagic",
    "WeaponTypePolearm",
    "EthnicityIce",
    "EthnicityRock",
    "EthnicityWater",
    "EthnicityFire",
    "EthnicityThunder",
    "EthnicityWind",
    "GenderFemale",
    "GenderMale",
    "CountryRuiYue",
    "CountryDaoQi",
    "CountryZhiDong",
    "CountryMengDe",
    "type_figure",
    "MinimumAttack",
    "MaximumAttack",
    "MinSpecialEffect",
    "MaxSpecialEffect",
    "SpecialEffectEfficiency",
    "SpecialEffectExpertise",
    "SpecialEffectAttack",
    "SpecialEffectSuper",
    "SpecialEffectRatio",
    "SpecialEffectPhysical",
    "SpecialEffectLife",
    "LTO",
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


# ═══════════════════════════════════════════════════════════════
# Dataset / Collate
# ═══════════════════════════════════════════════════════════════
class PredictDatasetWithUserID(JsonLineDataset):
    """
    Returns:
      uid
      x
      user_id        : exact training user id if available, else 0
      seen_in_train  : 1 if uid is in training user map, else 0
    """

    def __init__(self, path, pad_id, uid_to_index):
        super().__init__(path)
        self.pad_id = pad_id
        self.uid_to_index = uid_to_index

    def __getitem__(self, idx):
        row = super().__getitem__(idx)

        seq_raw = row["AggregateInput"]
        if isinstance(seq_raw, list):
            seq_str = seq_raw[0] if len(seq_raw) == 1 and isinstance(seq_raw[0], str) else " ".join(map(str, seq_raw))
        else:
            seq_str = str(seq_raw)

        toks = []
        for t in seq_str.strip().split():
            try:
                toks.append(int(t))
            except ValueError:
                toks.append(self.pad_id)

        uid_raw = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        uid = flat_uid(uid_raw)
        uid_norm = normalize_uid(uid)

        user_id = self.uid_to_index.get(uid_norm, 0)  # 0 = unseen / unknown
        seen_in_train = int(uid_norm in self.uid_to_index and user_id > 0)

        return {
            "uid": uid,
            "x": torch.tensor(toks, dtype=torch.long),
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "seen_in_train": torch.tensor(seen_in_train, dtype=torch.bool),
        }


def collate_fn_with_uid(pad_id):
    def _inner(batch):
        uids = [b["uid"] for b in batch]
        user_ids = torch.stack([b["user_id"] for b in batch])
        seen_in_train = torch.stack([b["seen_in_train"] for b in batch])

        lens = [len(b["x"]) for b in batch]
        Lmax = max(lens)
        X = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        for i, (item, L) in enumerate(zip(batch, lens)):
            X[i, :L] = item["x"]

        return {
            "uid": uids,
            "x": X,
            "user_id": user_ids,
            "seen_in_train": seen_in_train,
        }

    return _inner


# ═══════════════════════════════════════════════════════════════
# User-ID mapping
# ═══════════════════════════════════════════════════════════════
def build_uid_to_index(cfg, fold_spec_uri, fold_id):
    """
    Rebuild the training UID map exactly as training did, as closely as possible.
    IMPORTANT: cfg["filepath"] should match the training file.
    """
    spec = load_json_from_local_or_s3(fold_spec_uri)
    uids_test_fold = [u for u, f in spec["assignment"].items() if f == fold_id]
    uids_trainval = [u for u in spec["assignment"] if u not in set(uids_test_fold)]

    raw = load_json_dataset(cfg["filepath"], keep_uids=set(uids_trainval))
    unique_uids = sorted({normalize_uid(rec.get("uid", "")) for rec in raw})
    uid_to_index = {u: i + 1 for i, u in enumerate(unique_uids)}  # reserve 0 for UNK/mean
    num_users = len(uid_to_index) + 1

    print(f"[INFO] Built uid_to_index: {num_users} users (including UNK=0)")
    return uid_to_index, num_users


def extract_ckpt_num_users(state_dict_like: dict) -> int | None:
    for k, v in state_dict_like.items():
        if "gate.logits.weight" in k:
            return int(v.shape[0])
    return None


# ═══════════════════════════════════════════════════════════════
# Metrics setup
# ═══════════════════════════════════════════════════════════════
BIN_TASKS = {
    "BuyNone": [9],
    "BuyOne": [1, 3, 5, 7],
    "BuyTen": [2, 4, 6, 8],
    "BuyRegular": [1, 2],
    "BuyFigure": [3, 4, 5, 6],
    "BuyWeapon": [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}
GROUPS = ["Calibration", "HoldoutA", "HoldoutB"]
SPLITS = ["val", "test"]


def period_group(idx_h, feat_h):
    if feat_h == 0:
        return "Calibration"
    if feat_h == 1 and idx_h == 0:
        return "HoldoutA"
    if idx_h == 1:
        return "HoldoutB"
    return "UNASSIGNED"


def best_f1_threshold(y, p, default_thresh=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float(default_thresh)

    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return float(default_thresh)

    f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    j = int(np.nanargmax(f1s))
    return float(thresholds[j])


def fit_thresholds(scores, default_thresh=0.5):
    """
    Tune threshold on validation only.
    Primary: (Task, Group) threshold from that group's val set.
    Fallback: Task-wide threshold pooled over all validation groups.
    Final fallback: default_thresh.
    """
    thresholds = {}
    threshold_rows = []

    task_fallback = {}
    for task in BIN_TASKS:
        y_all, p_all = [], []
        for grp in GROUPS:
            y_all.extend(scores[(task, grp, "val")]["y"])
            p_all.extend(scores[(task, grp, "val")]["p"])
        if len(y_all) > 0 and len(set(y_all)) >= 2:
            task_fallback[task] = best_f1_threshold(y_all, p_all, default_thresh=default_thresh)
            source = "task_val"
        else:
            task_fallback[task] = float(default_thresh)
            source = "default"
        threshold_rows.append(
            {"Task": task, "Group": "ALL", "Threshold": round(task_fallback[task], 6), "Source": source}
        )

    for task in BIN_TASKS:
        for grp in GROUPS:
            y = scores[(task, grp, "val")]["y"]
            p = scores[(task, grp, "val")]["p"]
            if len(y) > 0 and len(set(y)) >= 2:
                thr = best_f1_threshold(y, p, default_thresh=task_fallback[task])
                src = "task_group_val"
            else:
                thr = task_fallback[task]
                src = "task_val_fallback"
            thresholds[(task, grp)] = float(thr)
            threshold_rows.append({"Task": task, "Group": grp, "Threshold": round(thr, 6), "Source": src})

    threshold_df = pd.DataFrame(threshold_rows).sort_values(["Task", "Group"])
    return thresholds, threshold_df


# ═══════════════════════════════════════════════════════════════
# Model forward helpers
# ═══════════════════════════════════════════════════════════════
def forward_with_gate_mode(model, x, user_ids, gate_mode: str):
    """
    gate_mode='user' for exact seen-user gating
    gate_mode='mean' for average training-user gating

    This tries the explicit mode first, then falls back to toggling model.gate.use_mean_gate.
    """
    if gate_mode not in {"user", "mean"}:
        raise ValueError(f"Unsupported gate_mode={gate_mode}")

    if gate_mode == "mean":
        try:
            return model(x, user_ids, projection_gate_mode="mean")
        except (TypeError, ValueError):
            pass

        old = None
        has_flag = hasattr(model, "gate") and hasattr(model.gate, "use_mean_gate")
        if has_flag:
            old = model.gate.use_mean_gate
        try:
            if has_flag:
                model.gate.use_mean_gate = True
            return model(x, user_ids)
        finally:
            if has_flag and old is not None:
                model.gate.use_mean_gate = old

    # gate_mode == "user"
    try:
        return model(x, user_ids, projection_gate_mode="user")
    except (TypeError, ValueError):
        pass

    old = None
    has_flag = hasattr(model, "gate") and hasattr(model.gate, "use_mean_gate")
    if has_flag:
        old = model.gate.use_mean_gate
    try:
        if has_flag:
            model.gate.use_mean_gate = False
        return model(x, user_ids)
    finally:
        if has_flag and old is not None:
            model.gate.use_mean_gate = old


def select_decision_logits(logits_full: torch.Tensor, x: torch.Tensor, ai_rate: int):
    pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=x.device)
    if logits_full.size(1) == x.size(1):
        logits = logits_full[:, pos, :]
    else:
        logits = logits_full
    return logits


def logits_to_prob_dec_9(
    logits: torch.Tensor,
    calibration_mode: str,
    calibrator,
    logit_bias_9,
) -> torch.Tensor:
    """
    Converts logits to 9-way decision probabilities.

    IMPORTANT:
    If V_out > 9, we first slice decision logits and then softmax over those 9 decision logits.
    That avoids depressing the decision probabilities by unrelated extra classes.
    """
    V_out = logits.size(-1)

    if V_out == 9:
        logits_dec = logits
    elif V_out >= 10:
        logits_dec = logits[..., 1:10]
    else:
        raise ValueError(f"Unexpected output dimension V_out={V_out}; cannot form 9-way decision probs.")

    if calibration_mode == "calibrator" and calibrator is not None:
        return calibrator(logits_dec)

    if calibration_mode == "analytic" and logit_bias_9 is not None:
        logits_dec = logits_dec - logit_bias_9.view(1, 1, -1)

    return torch.softmax(logits_dec, dim=-1)


def infer_batch_prob_dec_9(
    model,
    x: torch.Tensor,
    user_ids: torch.Tensor,
    seen_mask: torch.Tensor,
    ai_rate: int,
    calibration_mode: str,
    calibrator,
    logit_bias_9,
):
    """
    Seen users: exact user-specific gate
    Unseen users: mean gate

    Returns (B, T_decision, 9)
    """
    B = x.size(0)
    out = None

    seen_idx = torch.nonzero(seen_mask, as_tuple=False).squeeze(-1)
    unseen_idx = torch.nonzero(~seen_mask, as_tuple=False).squeeze(-1)

    if len(seen_idx) > 0:
        logits_seen_full = forward_with_gate_mode(model, x[seen_idx], user_ids[seen_idx], gate_mode="user")
        logits_seen = select_decision_logits(logits_seen_full, x[seen_idx], ai_rate)
        prob_seen = logits_to_prob_dec_9(logits_seen, calibration_mode, calibrator, logit_bias_9)

        if out is None:
            out = torch.empty(
                (B, prob_seen.size(1), prob_seen.size(2)),
                dtype=prob_seen.dtype,
                device=prob_seen.device,
            )
        out[seen_idx] = prob_seen

    if len(unseen_idx) > 0:
        logits_mean_full = forward_with_gate_mode(model, x[unseen_idx], user_ids[unseen_idx], gate_mode="mean")
        logits_mean = select_decision_logits(logits_mean_full, x[unseen_idx], ai_rate)
        prob_mean = logits_to_prob_dec_9(logits_mean, calibration_mode, calibrator, logit_bias_9)

        if out is None:
            out = torch.empty(
                (B, prob_mean.size(1), prob_mean.size(2)),
                dtype=prob_mean.dtype,
                device=prob_mean.device,
            )
        out[unseen_idx] = prob_mean

    return out


def infer_prob_9_all_positions(
    model,
    x_1d: torch.Tensor,
    user_id: int,
    seen_in_train: bool,
    device: torch.device,
    calibration_mode: str,
    calibrator,
    logit_bias_9,
):
    """
    Return 9-way probabilities for ALL sequence positions, shape (T, 9).
    """
    x = x_1d.unsqueeze(0).to(device)  # (1, T)
    user_ids = torch.tensor([user_id], dtype=torch.long, device=device)

    gate_mode = "user" if seen_in_train else "mean"
    logits_full = forward_with_gate_mode(model, x, user_ids, gate_mode=gate_mode)

    prob_all = logits_to_prob_dec_9(
        logits_full,
        calibration_mode=calibration_mode,
        calibrator=calibrator,
        logit_bias_9=logit_bias_9,
    )[0].detach().cpu().numpy()  # (T, 9)

    return prob_all


def debug_prefix_compare(
    ds,
    label_dict,
    model,
    device,
    cfg,
    calibrator,
    logit_bias_9,
    calibration_mode: str,
    sample_users: int = 5,
    max_decisions_per_user: int = 3,
):
    print("\n=============  PREFIX DIAGNOSTIC  =======================")

    shown = 0
    for i in range(len(ds)):
        item = ds[i]
        uid = item["uid"]
        x = item["x"]
        user_id = int(item["user_id"].item())
        seen = bool(item["seen_in_train"].item())

        if len(x) < cfg["ai_rate"]:
            continue

        decision_positions = list(range(cfg["ai_rate"] - 1, len(x), cfg["ai_rate"]))
        if len(decision_positions) == 0:
            continue

        full_prob_all = infer_prob_9_all_positions(
            model=model,
            x_1d=x,
            user_id=user_id,
            seen_in_train=seen,
            device=device,
            calibration_mode=calibration_mode,
            calibrator=calibrator,
            logit_bias_9=logit_bias_9,
        )

        print(f"\n[UID={uid}] seen_in_train={seen} seq_len={len(x)} decision_blocks={len(decision_positions)}")

        n_checked = 0
        for t, pos_t in enumerate(decision_positions):
            if n_checked >= max_decisions_per_user:
                break

            # inclusive prefix: x[:pos_t+1]
            x_incl = x[: pos_t + 1]
            incl_prob_all = infer_prob_9_all_positions(
                model=model,
                x_1d=x_incl,
                user_id=user_id,
                seen_in_train=seen,
                device=device,
                calibration_mode=calibration_mode,
                calibrator=calibrator,
                logit_bias_9=logit_bias_9,
            )

            # full vs inclusive prefix at the SAME position
            p_full_same = full_prob_all[pos_t]
            p_incl_same = incl_prob_all[pos_t]

            delta_future_l1 = float(np.abs(p_full_same - p_incl_same).sum())
            delta_future_max = float(np.abs(p_full_same - p_incl_same).max())

            # optional heuristic: x[:pos_t] vs x[:pos_t+1]
            heuristic_msg = "heuristic_excl_vs_incl: NA"
            if pos_t >= 1:
                x_excl = x[:pos_t]
                excl_prob_all = infer_prob_9_all_positions(
                    model=model,
                    x_1d=x_excl,
                    user_id=user_id,
                    seen_in_train=seen,
                    device=device,
                    calibration_mode=calibration_mode,
                    calibrator=calibrator,
                    logit_bias_9=logit_bias_9,
                )
                p_excl_last = excl_prob_all[-1]
                delta_addedtoken_l1 = float(np.abs(p_excl_last - p_incl_same).sum())
                delta_addedtoken_max = float(np.abs(p_excl_last - p_incl_same).max())
                heuristic_msg = (
                    f"heuristic_excl_vs_incl: "
                    f"L1={delta_addedtoken_l1:.6g}, "
                    f"max={delta_addedtoken_max:.6g}"
                )

            true_y = None
            if uid in label_dict and t < len(label_dict[uid]["label"]):
                true_y = int(label_dict[uid]["label"][t])

            pred_full = int(np.argmax(p_full_same) + 1)
            pred_incl = int(np.argmax(p_incl_same) + 1)

            tok_here = int(x[pos_t].item())
            tok_prev = int(x[pos_t - 1].item()) if pos_t >= 1 else None

            print(
                f"  t={t:02d} pos={pos_t:04d} "
                f"true={true_y} tok_prev={tok_prev} tok_here={tok_here} "
                f"pred_full={pred_full} pred_incl={pred_incl} "
                f"future_delta: L1={delta_future_l1:.6g}, max={delta_future_max:.6g}; "
                f"{heuristic_msg}"
            )

            n_checked += 1

        shown += 1
        if shown >= sample_users:
            break

    print("============================================================\n")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    data_path = _ensure_jsonl(args.data)
    ckpt_path = Path(args.ckpt)
    label_path = Path(args.labels)
    feat_path = Path(args.feat_xlsx)
    pred_out = args.pred_out

    s3_prefix = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    hp = parse_hp_from_ckpt_name(ckpt_path)

    if args.fold_id is not None and args.fold_id >= 0:
        s3_prefix_effective = s3_join_folder(s3_prefix, f"fold{args.fold_id}")
    else:
        s3_prefix_effective = s3_prefix
    print(f"[INFO] S3 upload prefix: {s3_prefix_effective}")

    # ---------- Config ----------
    cfg = get_config()
    cfg["ai_rate"] = args.ai_rate
    cfg["batch_size"] = args.batch_size
    if args.train_file:
        cfg["filepath"] = args.train_file
        print(f"[INFO] Overriding cfg['filepath'] with --train-file: {cfg['filepath']}")

    # ---------- Tokenizer / PAD ----------
    tok_path = Path(cfg["model_folder"]) / "tokenizer_tgt.json"
    tok_tgt = Tokenizer.from_file(str(tok_path)) if tok_path.exists() else _build_tok()
    pad_id = tok_tgt.token_to_id("[PAD]")

    SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
    EOS_PROD_ID, SOS_PROD_ID = 57, 58
    SPECIAL_IDS = [pad_id, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

    # ---------- Labels ----------
    label_dict, records = load_labels(label_path)

    # ---------- Split ----------
    uids_val_override = load_uid_set(args.uids_val) if args.uids_val else set()
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
        print(f"[INFO] Using 80/10/10 split on ALL label users with seed={args.seed}")

    # ---------- Rebuild training UID map ----------
    uid_to_index, num_users = build_uid_to_index(cfg, args.fold_spec, hp["fold_id"])

    # ---------- Inspect checkpoint num_users ----------
    tmp_state = torch.load(ckpt_path, map_location="cpu")
    tmp_sd = tmp_state.get("model_state_dict", tmp_state.get("module", tmp_state))
    ckpt_num_users = extract_ckpt_num_users(tmp_sd)
    if ckpt_num_users is None:
        raise RuntimeError("Could not find gate.logits.weight in checkpoint; cannot infer num_users.")

    if ckpt_num_users != num_users:
        print(f"[WARN] num_users mismatch: build_uid_to_index={num_users}, checkpoint={ckpt_num_users}")
        print("[WARN] This usually means cfg['filepath'] / --train-file or fold-spec does not exactly match training.")
        print(f"[INFO] Truncating uid_to_index to checkpoint size ({ckpt_num_users} users including UNK=0)")
        uid_to_index = {u: idx for u, idx in uid_to_index.items() if idx < ckpt_num_users}
        num_users = ckpt_num_users

    del tmp_state

    # ---------- Dataset ----------
    ds = PredictDatasetWithUserID(data_path, pad_id=pad_id, uid_to_index=uid_to_index)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_with_uid(pad_id),
    )

    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure seq_len matches training
    cfg["seq_len_ai"] = cfg["seq_len_tgt"] * cfg["ai_rate"]

    # ---------- Build model ----------
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
        num_users=num_users,
    ).to(device).eval()

    # ---------- Load checkpoint ----------
    def clean_state_dict(raw):
        def strip_prefix(k):
            return k[7:] if k.startswith("module.") else k

        ignore = ("weight_fake_quant", "activation_post_process")
        return {strip_prefix(k): v for k, v in raw.items() if not any(tok in k for tok in ignore)}

    torch.cuda.empty_cache()
    state = torch.load(ckpt_path, map_location=device)
    raw_sd = (
        state["model_state_dict"]
        if "model_state_dict" in state
        else state["module"]
        if "module" in state
        else state
    )
    sd = clean_state_dict(raw_sd)
    sd.pop("projection.output_head.mean_alpha_buffer", None)
    model.load_state_dict(sd, strict=False)

    # ---------- Build mean gate from training users ----------
    train_user_ids = sorted(set(uid_to_index.values()))
    if len(train_user_ids) == 0:
        raise RuntimeError("No training user IDs available to build mean gate.")

    model.set_projection_mean_gate_from_train_users(train_user_ids)
    if hasattr(model.gate, "update_mean_gate"):
        model.gate.update_mean_gate()

    # ---------- Calibration ----------
    calibrator = None
    logit_bias_9 = None

    if args.calibration == "calibrator":
        calibrator = load_calibrator(ckpt_path, device)
        if calibrator is None:
            print("[WARN] No calibrator found; falling back to analytic.")
            args.calibration = "analytic"

    if args.calibration == "analytic":
        if "logit_bias_9" in state:
            logit_bias_9 = torch.tensor(state["logit_bias_9"], device=device, dtype=torch.float32)
        else:
            logit_bias_9 = torch.zeros(9, device=device, dtype=torch.float32)
            w = hp["weight"]
            if w != 1:
                logit_bias_9[8] = math.log(w)
        print(f"[INFO] logit_bias_9 = {logit_bias_9.cpu().numpy().round(4)}")

    print(f"[INFO] Calibration: {args.calibration}")

    debug_prefix_compare(
    ds=ds,
    label_dict=label_dict,
    model=model,
    device=device,
    cfg=cfg,
    calibrator=calibrator,
    logit_bias_9=logit_bias_9,
    calibration_mode=args.calibration,
    sample_users=5,
    max_decisions_per_user=3,
)

    # ---------- Pass 1: user-level predictions ----------
    all_user_probs = {}  # uid -> (T, 9) numpy array or None for too-short users
    seen_user_count = 0
    unseen_user_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            uids = batch["uid"]
            user_ids = batch["user_id"].to(device)
            seen_mask = batch["seen_in_train"].to(device)

            seen_user_count += int(seen_mask.sum().item())
            unseen_user_count += int((~seen_mask).sum().item())

            if x.size(1) < cfg["ai_rate"]:
                for uid in uids:
                    all_user_probs[uid] = None
                continue

            prob_dec_9 = infer_batch_prob_dec_9(
                model=model,
                x=x,
                user_ids=user_ids,
                seen_mask=seen_mask,
                ai_rate=cfg["ai_rate"],
                calibration_mode=args.calibration,
                calibrator=calibrator,
                logit_bias_9=logit_bias_9,
            )

            prob_dec_9_np = prob_dec_9.detach().cpu().numpy()
            for i, uid in enumerate(uids):
                all_user_probs[uid] = prob_dec_9_np[i]

    print(f"[INFO] Users seen in training map: {seen_user_count}")
    print(f"[INFO] Users not seen in training map (mean gate used): {unseen_user_count}")

    # ---------- Population average fallback for users with no valid input ----------
    valid_probs = [v for v in all_user_probs.values() if v is not None]
    if len(valid_probs) == 0:
        raise RuntimeError("No valid user probabilities were produced.")

    max_slots = max(v.shape[0] for v in valid_probs)
    slot_sums = np.zeros((max_slots, 9), dtype=np.float64)
    slot_counts = np.zeros(max_slots, dtype=np.float64)

    for v in valid_probs:
        n = v.shape[0]
        slot_sums[:n] += v
        slot_counts[:n] += 1

    global_avg = (slot_sums.sum(axis=0) / slot_counts.sum()).astype(np.float32)
    per_slot_avg = np.zeros((max_slots, 9), dtype=np.float32)
    for t in range(max_slots):
        if slot_counts[t] > 0:
            per_slot_avg[t] = slot_sums[t] / slot_counts[t]
        else:
            per_slot_avg[t] = global_avg

    print(f"[INFO] Population average prob (global): {np.round(global_avg, 4)}")
    print(
        f"[INFO] Users with valid input: {len(valid_probs)}, "
        f"users needing average: {sum(1 for v in all_user_probs.values() if v is None)}"
    )

    for uid, v in all_user_probs.items():
        if v is None:
            lbl_info = label_dict.get(uid)
            if lbl_info is not None:
                n_labels = len(lbl_info["label"])
                if n_labels <= max_slots:
                    all_user_probs[uid] = per_slot_avg[:n_labels].copy()
                else:
                    extra = np.tile(global_avg, (n_labels - max_slots, 1))
                    all_user_probs[uid] = np.vstack([per_slot_avg.copy(), extra])

    # ---------- Evaluate ----------
    scores = defaultdict(lambda: {"y": [], "p": []})
    length_note = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    pred_cm = smart_open_w(pred_out) if pred_out else None
    pred_writer = pred_cm.__enter__() if pred_cm else None

    try:
        for uid, probs_seq_np in all_user_probs.items():
            if pred_writer:
                pred_writer.write(json.dumps({"uid": uid, "probs": np.round(probs_seq_np, 6).tolist()}) + "\n")

            lbl_info = label_dict.get(uid)
            if lbl_info is None:
                reject += 1
                continue

            L_pred, L_lbl = len(probs_seq_np), len(lbl_info["label"])
            if L_pred != L_lbl:
                length_note["pred>lbl" if L_pred > L_lbl else "pred<label"] += 1
            L = min(L_pred, L_lbl)

            split_tag = which_split(uid)
            accept_users.setdefault(split_tag, set()).add(uid)

            for t in range(L):
                y = lbl_info["label"][t]
                idx_h = lbl_info["idx_h"][t]
                feat_h = lbl_info["feat_h"][t]
                probs = probs_seq_np[t]

                group = period_group(idx_h, feat_h)

                for task, pos_classes in BIN_TASKS.items():
                    y_bin = int(y in TASK_POSSETS[task])
                    p_bin = float(sum(probs[j - 1] for j in pos_classes))
                    key = (task, group, split_tag)
                    scores[key]["y"].append(y_bin)
                    scores[key]["p"].append(p_bin)

            accept += 1
    finally:
        if pred_cm:
            pred_cm.__exit__(None, None, None)

    print(f"[INFO] parsed: {accept} users accepted, {reject} users missing labels.")
    if length_note:
        print("[INFO] length mismatches:", dict(length_note))
    if args.uids_val and args.uids_test:
        print(
            f"[INFO] coverage: val={len(accept_users.get('val', set()))} / {len(load_uid_set(args.uids_val))}, "
            f"test={len(accept_users.get('test', set()))} / {len(load_uid_set(args.uids_test))}"
        )

    # ---------- Diagnostics ----------
    print("\n=============  PROBABILITY DIAGNOSTICS @ 0.5  =======================")
    for task in BIN_TASKS:
        for grp in GROUPS:
            for spl in SPLITS:
                p = np.array(scores[(task, grp, spl)]["p"], dtype=float)
                y = np.array(scores[(task, grp, spl)]["y"], dtype=int)
                if len(p) == 0:
                    continue
                print(
                    task,
                    grp,
                    spl,
                    "min=", round(float(p.min()), 4),
                    "q50=", round(float(np.quantile(p, 0.50)), 4),
                    "q90=", round(float(np.quantile(p, 0.90)), 4),
                    "q99=", round(float(np.quantile(p, 0.99)), 4),
                    "max=", round(float(p.max()), 4),
                    "pred_pos@0.5=", round(float((p >= 0.5).mean()), 6),
                    "prev=", round(float(y.mean()), 4),
                )
    print("============================================================")

    # ---------- Bucket stats ----------
    bucket_stats = []
    for task in BIN_TASKS:
        for grp in GROUPS:
            for spl in SPLITS:
                y = scores[(task, grp, spl)]["y"]
                if not y:
                    continue
                n = len(y)
                pos = sum(y)
                neg = n - pos
                prev = pos / n if n else float("nan")
                bucket_stats.append(
                    {
                        "Task": task,
                        "Group": grp,
                        "Split": spl,
                        "N": n,
                        "Pos": pos,
                        "Neg": neg,
                        "Prev": round(prev, 4),
                    }
                )

    stats_df = pd.DataFrame(bucket_stats).sort_values(["Split", "Group", "Task"])
    print("\n=============  BUCKET SIZES & PREVALENCE  =======================")
    print(stats_df.to_string(index=False))
    print("============================================================")

    # ---------- Thresholds ----------
    thresholds, threshold_df = fit_thresholds(scores, default_thresh=args.thresh)
    print("\n=============  VALIDATION-TUNED THRESHOLDS  =======================")
    print(threshold_df.to_string(index=False))
    print("============================================================")

    # ---------- Metrics ----------
    rows = []
    for task in BIN_TASKS:
        for grp in GROUPS:
            thr = thresholds[(task, grp)]
            for spl in SPLITS:
                y = scores[(task, grp, spl)]["y"]
                p = scores[(task, grp, spl)]["p"]
                if not y:
                    continue

                y_arr = np.asarray(y, dtype=int)
                p_arr = np.asarray(p, dtype=float)
                y_hat = (p_arr >= thr).astype(int)

                if len(set(y_arr)) < 2:
                    auc = np.nan
                    auprc = np.nan
                else:
                    auc = roc_auc_score(y_arr, p_arr)
                    auprc = average_precision_score(y_arr, p_arr)

                acc = accuracy_score(y_arr, y_hat)
                f1 = f1_score(y_arr, y_hat, zero_division=0)

                rows.append(
                    {
                        "Task": task,
                        "Group": grp,
                        "Split": spl,
                        "Threshold": thr,
                        "AUC": auc,
                        "Hit": acc,
                        "F1": f1,
                        "AUPRC": auprc,
                    }
                )

    metrics = pd.DataFrame(rows)

    def pivot(metric: str) -> pd.DataFrame:
        return (
            metrics.pivot(index=["Task", "Group"], columns="Split", values=metric)
            .reindex(columns=["val", "test"])
            .round(4)
            .sort_index()
        )

    auc_tbl = pivot("AUC")
    hit_tbl = pivot("Hit")
    f1_tbl = pivot("F1")
    auprc_tbl = pivot("AUPRC")
    thr_tbl = pivot("Threshold")

    macro_period_tbl = (
        metrics.groupby(["Group", "Split"])[["AUC", "Hit", "F1", "AUPRC"]]
        .mean()
        .unstack("Split")
        .round(4)
    )
    macro_period_tbl = macro_period_tbl.reorder_levels([1, 0], axis=1)
    macro_period_tbl = macro_period_tbl.sort_index(axis=1, level=0)
    macro_period_tbl = macro_period_tbl[["val", "test"]]

    def _p(title: str, df: pd.DataFrame):
        print(f"\n=============  {title}  =======================")
        print(df.fillna(" NA"))
        print("============================================================")

    _p("THRESHOLD TABLE", thr_tbl)
    _p("BINARY ROC-AUC TABLE", auc_tbl)
    _p("HIT-RATE (ACCURACY) TABLE", hit_tbl)
    _p("POSITIVE-CLASS F1 TABLE", f1_tbl)
    _p("AUPRC TABLE", auprc_tbl)
    _p("AGGREGATE MACRO METRICS", macro_period_tbl)

    # ---------- Save locally & upload ----------
    out_dir = Path("/tmp/predict_eval_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_csv = out_dir / "auc_table.csv"
    hit_csv = out_dir / "hit_table.csv"
    f1_csv = out_dir / "f1_table.csv"
    auprc_csv = out_dir / "auprc_table.csv"
    thr_csv = out_dir / "threshold_table.csv"
    thr_long_csv = out_dir / "threshold_table_long.csv"
    macro_csv = out_dir / "macro_period_table.csv"

    auc_tbl.to_csv(auc_csv)
    hit_tbl.to_csv(hit_csv)
    f1_tbl.to_csv(f1_csv)
    auprc_tbl.to_csv(auprc_csv)
    thr_tbl.to_csv(thr_csv)
    threshold_df.to_csv(thr_long_csv, index=False)
    macro_period_tbl.to_csv(macro_csv)

    for pth in [auc_csv, hit_csv, f1_csv, auprc_csv, thr_csv, thr_long_csv, macro_csv]:
        dest = s3_join(s3_prefix_effective, pth.name)
        s3_upload_file(pth, dest)
        print(f"[S3] uploaded: {dest}")

    if pred_out:
        dest = s3_join(s3_prefix_effective, Path(pred_out).name)
        s3_upload_file(pred_out, dest)
        print(f"[S3] uploaded: {dest}")


if __name__ == "__main__":
    main()