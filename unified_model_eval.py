#!/usr/bin/env python3
"""
unified_eval_and_compare.py

Runs end-to-end inference + evaluation for any combination of:
  - ProductGPT (featurebased performer)
  - ProductGPT Mixture-head (mixture2 performer)
  - GRU
  - LSTM

and produces cross-model comparison tables.

Config JSON example (models.json):
  {
    "models": [
      {
        "name": "productgpt_fold0",
        "model_family": "productgpt",
        "ckpt": "/tmp/FullProductGPT_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.0005_w1_fold0.pt",
        "feat_xlsx": "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx",
        "calibration": "calibrator",
        "ai_rate": 15,
        "batch_size": 2
      },
      {
        "name": "mixture2_fold0",
        "model_family": "mixture",
        "ckpt": "/tmp/FullProductGPT_mixture2_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.0005_w1_fold0.pt",
        "feat_xlsx": "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx",
        "calibration": "calibrator",
        "ai_rate": 15,
        "batch_size": 2,
        "fold_spec": "s3://productgptbucket/folds/productgptfolds.json",
        "train_file": "/home/ec2-user/data/clean_list_int_wide4_simple6.json"
      },
      {
        "name": "gru_fold0",
        "model_family": "gru",
        "ckpt": "/home/ec2-user/tmp_gru/gru_h128_lr0.001_bs4.pt",
        "hidden_size": 128,
        "input_dim": 15,
        "batch_size": 128
      },
      {
        "name": "lstm_fold0",
        "model_family": "lstm",
        "ckpt": "/home/ec2-user/tmp_lstm/lstm_h128_lr0.001_bs4.pt",
        "hidden_size": 128,
        "input_dim": 15,
        "batch_size": 128
      }
    ]
  }

Usage:
  python3 unified_eval_and_compare.py \\
    --config    models.json \\
    --data      /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --labels    /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --uids-val  s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \\
    --uids-test s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \\
    --fold-id   0 \\
    --output-dir /tmp/unified_eval_fold0 \\
    --s3        s3://productgptbucket/evals/unified_fold0/ \\
    --compare-on test
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Optional ProductGPT (featurebased) imports ────────────────────────────────
try:
    from dataset4_productgpt import load_json_dataset as productgpt_load_json_dataset
    from config4 import get_config as productgpt_get_config
    from model4_decoderonly_feature_performer import build_transformer as productgpt_build_transformer
    from train1_decision_only_performer_aws import _ensure_jsonl as productgpt_ensure_jsonl
    from train1_decision_only_performer_aws import JsonLineDataset as ProductGPTJsonLineDataset
    from train1_decision_only_performer_aws import _build_tok as productgpt_build_tok
    from tokenizers import Tokenizer
    PRODUCTGPT_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    productgpt_load_json_dataset = None
    productgpt_get_config = None
    productgpt_build_transformer = None
    productgpt_ensure_jsonl = None
    ProductGPTJsonLineDataset = object
    productgpt_build_tok = None
    Tokenizer = None
    PRODUCTGPT_IMPORT_ERROR = exc

# ── Optional Mixture model imports ────────────────────────────────────────────
try:
    from model4_mixture2_decoderonly_feature_performer import build_transformer as mixture_build_transformer
    from config4 import get_config as mixture_get_config
    from train1_decision_only_performer_aws import _ensure_jsonl as mixture_ensure_jsonl
    from train1_decision_only_performer_aws import JsonLineDataset as MixtureJsonLineDataset
    from train1_decision_only_performer_aws import _build_tok as mixture_build_tok
    from dataset4_productgpt import load_json_dataset as mixture_load_json_dataset
    MIXTURE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    mixture_build_transformer = None
    mixture_get_config = None
    mixture_ensure_jsonl = None
    MixtureJsonLineDataset = object
    mixture_build_tok = None
    mixture_load_json_dataset = None
    MIXTURE_IMPORT_ERROR = exc


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
GROUP_ORDER = ["Calibration", "HoldoutA", "HoldoutB"]
SPLIT_ORDER = ["val", "test"]
NINE_CLASSES = list(range(1, 10))
PERCLASS_METRICS = ["AUC_OvR", "AUPRC", "F1", "Support"]
MULTICLASS_METRICS = [
    "MacroOvR_AUC", "MacroAUPRC", "MacroF1", "MCC", "LogLoss", "Top1Acc", "Top2Acc",
]
BIN_TASKS = {
    "BuyNone":   [9],
    "BuyOne":    [1, 3, 5, 7],
    "BuyTen":    [2, 4, 6, 8],
    "BuyRegular":[1, 2],
    "BuyFigure": [3, 4, 5, 6],
    "BuyWeapon": [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}
PAPER_AUC_TASKS = {
    "BuyOne":    "Buy One",
    "BuyTen":    "Buy Ten",
    "BuyFigure": "Character Event Wish",
    "BuyWeapon": "Weapon Event Wish",
    "BuyRegular":"Regular Wish",
}
PAPER_AUC_ORDER = [
    "Buy One", "Buy Ten", "Character Event Wish", "Weapon Event Wish", "Regular Wish",
]
FEATURE_COLS = [
    "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
    "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow", "WeaponTypeMagic", "WeaponTypePolearm",
    "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire", "EthnicityThunder", "EthnicityWind",
    "GenderFemale", "GenderMale", "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
    "type_figure", "MinimumAttack", "MaximumAttack", "MinSpecialEffect", "MaxSpecialEffect",
    "SpecialEffectEfficiency", "SpecialEffectExpertise", "SpecialEffectAttack", "SpecialEffectSuper",
    "SpecialEffectRatio", "SpecialEffectPhysical", "SpecialEffectLife", "LTO",
]
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
UNK_PROD_ID = 59
MAX_TOKEN_ID = UNK_PROD_ID


@dataclass
class EvalResult:
    model_name: str
    binary_metrics: pd.DataFrame
    multiclass_metrics: pd.DataFrame
    perclass_metrics: pd.DataFrame
    selected_binary_auc: pd.DataFrame
    info: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Generic utilities
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified evaluator for ProductGPT, Mixture, GRU, and LSTM."
    )
    p.add_argument("--config",      required=True, help="JSON file describing all models to evaluate")
    p.add_argument("--data",        required=True, help="Common data path used by all model adapters")
    p.add_argument("--labels",      required=True, help="Label JSON used by all models")
    p.add_argument("--uids-val",    default="",    help="Validation UID file (local or s3://)")
    p.add_argument("--uids-test",   default="",    help="Test UID file (local or s3://)")
    p.add_argument("--fold-id",     type=int,      default=-1)
    p.add_argument("--seed",        type=int,      default=33)
    p.add_argument("--thresh",      type=float,    default=0.5)
    p.add_argument("--compare-on",  choices=["test", "val"], default="test")
    p.add_argument("--output-dir",  required=True, help="Local directory for all outputs")
    p.add_argument("--s3",          default="",    help="Optional S3 prefix for uploading all outputs")
    p.add_argument("--save-preds",  action="store_true")
    return p.parse_args()


def smart_open_w(path: str | Path):
    if isinstance(path, Path):
        path = str(path)
    if path == "-":
        return nullcontext(sys.stdout)
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 uri: {uri}")
    no_scheme = uri[5:]
    bucket, key = no_scheme.split("/", 1) if "/" in no_scheme else (no_scheme, "")
    return bucket, key


def s3_join(prefix: str, filename: str) -> str:
    if not prefix.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {prefix}")
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + filename


def s3_join_folder(prefix: str, folder: str) -> str:
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + folder.strip("/") + "/"


def s3_upload_file(local_path: str | Path, s3_uri_full: str):
    assert not s3_uri_full.endswith("/")
    try:
        import boto3
        bucket, key = parse_s3_uri(s3_uri_full)
        boto3.client("s3").upload_file(str(local_path), bucket, key)
    except Exception:
        rc = os.system(f"aws s3 cp '{local_path}' '{s3_uri_full}'")
        if rc != 0:
            raise RuntimeError(f"Failed to upload {local_path} to {s3_uri_full}")


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
    return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")}


def load_json_from_local_or_s3(path_or_s3: str) -> Any:
    if path_or_s3.startswith("s3://"):
        return json.loads(s3_read_text(path_or_s3))
    return json.loads(Path(path_or_s3).read_text())


def to_int_vec(x: Any) -> List[int]:
    if isinstance(x, str):
        return [int(v) for v in x.split()]
    if isinstance(x, list):
        out: List[int] = []
        for item in x:
            if isinstance(item, str):
                out.extend(int(v) for v in item.split())
            else:
                out.append(int(item))
        return out
    raise TypeError(type(x))


def flat_uid(u: Any) -> str:
    return str(u[0] if isinstance(u, list) else u)


def normalize_uid(uid: Any) -> str:
    """Stable string key for uid, used by mixture gate mapping."""
    if uid is None:
        return ""
    if isinstance(uid, (list, tuple, set)):
        vals = [str(x) for x in uid if x is not None]
        return "|".join(sorted(vals)) if vals else ""
    return str(uid)


def load_labels(label_path: Path) -> Tuple[Dict[str, Dict[str, List[int]]], List[dict]]:
    raw = json.loads(label_path.read_text())
    records = list(raw) if isinstance(raw, list) else [
        {k: raw[k][i] for k in raw} for i in range(len(raw["uid"]))
    ]
    label_dict = {
        flat_uid(rec["uid"]): {
            "label": to_int_vec(rec["Decision"]),
            "idx_h": to_int_vec(rec["IndexBasedHoldout"]),
            "feat_h": to_int_vec(rec["FeatureBasedHoldout"]),
        }
        for rec in records
    }
    return label_dict, records


def build_splits(records: Sequence[dict], seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(records)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    tr_i, va_i, te_i = random_split(range(n), [n_train, n_val, n - n_train - n_val], generator=g)
    val_uid  = {flat_uid(records[i]["uid"]) for i in va_i.indices}
    test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}
    def which(u: str) -> str:
        return "val" if u in val_uid else "test" if u in test_uid else "train"
    return which


def make_splitter(records, uids_val_path, uids_test_path, seed):
    uids_val  = load_uid_set(uids_val_path)  if uids_val_path  else set()
    uids_test = load_uid_set(uids_test_path) if uids_test_path else set()
    if uids_val or uids_test:
        if not (uids_val and uids_test):
            raise ValueError("Provide BOTH --uids-val and --uids-test (or neither).")
        overlap = uids_val & uids_test
        if overlap:
            raise ValueError(f"UID overlap: {sorted(list(overlap))[:5]}")
        def which(u: str) -> str:
            return "val" if u in uids_val else "test" if u in uids_test else "train"
        return which, {"mode": "exact_uids", "val": len(uids_val), "test": len(uids_test)}, uids_val, uids_test
    which = build_splits(records, seed=seed)
    return which, {"mode": "seeded_random_split", "seed": seed}, set(), set()


# ═══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ═══════════════════════════════════════════════════════════════════════════════
def compute_multiclass_metrics(y_arr: np.ndarray, p_arr: np.ndarray) -> Dict[str, float]:
    y_hat = p_arr.argmax(axis=1) + 1
    try:
        macro_ovr_auc = roc_auc_score(
            y_arr, p_arr, multi_class="ovr", average="macro", labels=NINE_CLASSES,
        )
    except ValueError:
        macro_ovr_auc = np.nan
    y_bin = label_binarize(y_arr, classes=NINE_CLASSES)
    per_class_ap = [
        average_precision_score(y_bin[:, c], p_arr[:, c])
        for c in range(9) if y_bin[:, c].sum() > 0
    ]
    macro_auprc = float(np.mean(per_class_ap)) if per_class_ap else np.nan
    macro_f1  = f1_score(y_arr, y_hat, labels=NINE_CLASSES, average="macro", zero_division=0)
    mcc       = matthews_corrcoef(y_arr, y_hat)
    ll        = log_loss(y_arr, np.clip(p_arr, 1e-7, 1.0), labels=NINE_CLASSES)
    top1      = float(np.mean(y_arr == y_hat))
    top2_idx  = np.argsort(p_arr, axis=1)[:, -2:] + 1
    top2_acc  = float(np.mean([y_arr[i] in top2_idx[i] for i in range(len(y_arr))]))
    return {
        "MacroOvR_AUC": round(float(macro_ovr_auc), 4),
        "MacroAUPRC":   round(float(macro_auprc),   4),
        "MacroF1":      round(float(macro_f1),       4),
        "MCC":          round(float(mcc),            4),
        "LogLoss":      round(float(ll),             4),
        "Top1Acc":      round(float(top1),           4),
        "Top2Acc":      round(float(top2_acc),       4),
    }


def compute_perclass_metrics(y_arr: np.ndarray, p_arr: np.ndarray) -> Dict[int, Dict[str, float]]:
    y_hat = p_arr.argmax(axis=1) + 1
    y_bin = label_binarize(y_arr, classes=NINE_CLASSES)
    results: Dict[int, Dict[str, float]] = {}
    for c_idx, c in enumerate(NINE_CLASSES):
        support     = int((y_arr == c).sum())
        y_bin_c     = y_bin[:, c_idx]
        p_c         = p_arr[:, c_idx]
        y_hat_bin_c = (y_hat == c).astype(int)
        if y_bin_c.sum() == 0 or y_bin_c.sum() == len(y_arr):
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_bin_c, p_c)
            except ValueError:
                auc = np.nan
        auprc = average_precision_score(y_bin_c, p_c) if y_bin_c.sum() > 0 else np.nan
        f1    = f1_score(y_bin_c, y_hat_bin_c, zero_division=0)
        results[c] = {
            "AUC_OvR": round(float(auc),   4),
            "AUPRC":   round(float(auprc), 4),
            "F1":      round(float(f1),    4),
            "Support": support,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Shared model helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_feature_tensor(xls_path: Path) -> torch.Tensor:
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


def parse_hp_from_ckpt_name(ckpt_path: Path) -> Dict[str, int]:
    import re
    m = re.search(
        r"performerfeatures(?P<nb>\d+)_dmodel(?P<dm>\d+)_ff(?P<ff>\d+)_N(?P<N>\d+)_heads(?P<h>\d+)_lr(?P<lr>[\deE\.\-]+)_w(?P<w>\d+)_fold(?P<fold>\d+)",
        ckpt_path.name,
    )
    if not m:
        raise ValueError(f"Cannot parse HPs from ckpt filename: {ckpt_path.name}")
    d = m.groupdict()
    return {
        "nb_features": int(d["nb"]), "d_model": int(d["dm"]), "d_ff": int(d["ff"]),
        "N": int(d["N"]), "num_heads": int(d["h"]), "weight": int(d["w"]), "fold_id": int(d["fold"]),
    }


class VectorScaling(torch.nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(n_classes))
        self.b = torch.nn.Parameter(torch.zeros(n_classes))
    def forward(self, logits_dec: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.a * logits_dec + self.b, dim=-1)


def load_calibrator_from_path(cal_path: Path, device: torch.device) -> Optional[VectorScaling]:
    if not cal_path.exists():
        return None
    cal = VectorScaling(n_classes=9).to(device)
    state = torch.load(cal_path, map_location=device)
    cal.a.data = state["a"].to(device)
    cal.b.data = state["b"].to(device)
    print(f"[INFO] Loaded calibrator from {cal_path}")
    print(f"[INFO]   a={cal.a.data.cpu().numpy().round(4)}  b={cal.b.data.cpu().numpy().round(4)}")
    return cal


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


# ═══════════════════════════════════════════════════════════════════════════════
# Adapter base
# ═══════════════════════════════════════════════════════════════════════════════
class BaseAdapter:
    def __init__(self, spec: Dict[str, Any], args: argparse.Namespace):
        self.spec  = spec
        self.args  = args
        self.name  = spec["name"]
        self.model_family = spec["model_family"]

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
# ProductGPT adapter
# ═══════════════════════════════════════════════════════════════════════════════
class ProductGPTPredictDataset(ProductGPTJsonLineDataset):
    def __init__(self, path: str, pad_id: int):
        super().__init__(path)
        self.pad_id = pad_id
    def to_int_or_pad(self, tok: str) -> int:
        try:   return int(tok)
        except ValueError: return self.pad_id
    def __getitem__(self, idx: int):
        row = super().__getitem__(idx)
        seq_raw = row["AggregateInput"]
        if isinstance(seq_raw, list):
            seq_str = seq_raw[0] if len(seq_raw) == 1 and isinstance(seq_raw[0], str) else " ".join(map(str, seq_raw))
        else:
            seq_str = str(seq_raw)
        toks = [self.to_int_or_pad(t) for t in seq_str.strip().split()]
        uid  = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": flat_uid(uid), "x": torch.tensor(toks, dtype=torch.long)}


def productgpt_collate_fn(pad_id: int):
    def _inner(batch):
        uids = [b["uid"] for b in batch]
        lens = [len(b["x"]) for b in batch]
        Lmax = max(lens)
        X = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        for i, (item, L) in enumerate(zip(batch, lens)):
            X[i, :L] = item["x"]
        return {"uid": uids, "x": X, "lens": lens}
    return _inner


class ProductGPTAdapter(BaseAdapter):
    def __init__(self, spec: Dict[str, Any], args: argparse.Namespace):
        super().__init__(spec, args)
        if PRODUCTGPT_IMPORT_ERROR is not None:
            raise RuntimeError("ProductGPT dependencies unavailable") from PRODUCTGPT_IMPORT_ERROR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt   = Path(spec["ckpt"])
        self.hp     = parse_hp_from_ckpt_name(self.ckpt)
        self.cfg    = productgpt_get_config()
        self.cfg["ai_rate"]    = int(spec.get("ai_rate", 15))
        self.cfg["batch_size"] = int(spec.get("batch_size", 2))
        self.cfg["seq_len_ai"] = self.cfg["seq_len_tgt"] * self.cfg["ai_rate"]
        tok_path = Path(self.cfg["model_folder"]) / "tokenizer_tgt.json"
        self.tok_tgt  = Tokenizer.from_file(str(tok_path)) if tok_path.exists() else productgpt_build_tok()
        self.pad_id   = self.tok_tgt.token_to_id("[PAD]")
        self.special_ids = [self.pad_id, 10, 11, 12, 57, 58]
        self.data_path    = productgpt_ensure_jsonl(args.data)
        self.feat_path    = Path(spec["feat_xlsx"])
        self.calibration  = spec.get("calibration", "none")
        self.calibrator   = None
        self.logit_bias_9 = None
        self.model = self._build_model()
        self._load_calibration()

    def _build_model(self):
        model = productgpt_build_transformer(
            vocab_size_tgt=self.cfg["vocab_size_tgt"],
            vocab_size_src=self.cfg["vocab_size_src"],
            max_seq_len=self.cfg["seq_len_ai"],
            d_model=self.hp["d_model"],
            n_layers=self.hp["N"],
            n_heads=self.hp["num_heads"],
            d_ff=self.hp["d_ff"],
            dropout=0.0,
            nb_features=self.hp["nb_features"],
            kernel_type=self.cfg["kernel_type"],
            feature_tensor=load_feature_tensor(self.feat_path),
            special_token_ids=self.special_ids,
        ).to(self.device).eval()
        state  = torch.load(self.ckpt, map_location=self.device)
        raw_sd = state.get("model_state_dict", state.get("module", state))
        clean_sd = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in raw_sd.items()
            if "weight_fake_quant" not in k and "activation_post_process" not in k
        }
        model.load_state_dict(clean_sd, strict=True)
        self._state = state
        return model

    def _load_calibration(self):
        if self.calibration == "calibrator":
            stem = self.ckpt.stem.replace("FullProductGPT_", "")
            cal_path = Path(self.spec.get("calibrator_ckpt") or (self.ckpt.parent / f"calibrator_{stem}.pt"))
            self.calibrator = load_calibrator_from_path(cal_path, self.device)
            if self.calibrator is None:
                print(f"[WARN] {self.name}: calibrator not found; falling back to none")
                self.calibration = "none"
        elif self.calibration == "analytic":
            if "logit_bias_9" in self._state:
                self.logit_bias_9 = torch.tensor(self._state["logit_bias_9"], device=self.device, dtype=torch.float32)
            else:
                w = self.hp["weight"]
                self.logit_bias_9 = torch.zeros(9, device=self.device, dtype=torch.float32)
                if w != 1:
                    self.logit_bias_9[8] = math.log(w)

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        ds     = ProductGPTPredictDataset(self.data_path, pad_id=self.pad_id)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=False,
                            collate_fn=productgpt_collate_fn(self.pad_id))
        with torch.no_grad():
            for batch in loader:
                x    = batch["x"].to(self.device)
                uids = batch["uid"]
                lens = batch["lens"]
                logits_full = self.model(x)
                if x.size(1) < self.cfg["ai_rate"]:
                    pos = torch.empty((0,), dtype=torch.long, device=self.device)
                else:
                    pos = torch.arange(self.cfg["ai_rate"] - 1, x.size(1), self.cfg["ai_rate"], device=self.device)
                logits     = logits_full[:, pos, :] if logits_full.size(1) == x.size(1) else logits_full
                logits_dec = logits if logits.size(-1) == 9 else logits[..., 1:10]
                if self.calibration == "calibrator" and self.calibrator is not None:
                    probs = self.calibrator(logits_dec)
                elif self.calibration == "analytic" and self.logit_bias_9 is not None:
                    probs = torch.softmax(logits_dec + self.logit_bias_9.view(1, 1, 9), dim=-1)
                else:
                    probs = torch.softmax(logits_dec, dim=-1)
                yield {"uid": uids, "lens": lens, "probs_dec_9": probs.detach().cpu().numpy(), "ai_rate": self.cfg["ai_rate"]}


# ═══════════════════════════════════════════════════════════════════════════════
# Mixture adapter
# ═══════════════════════════════════════════════════════════════════════════════
class MixturePredictDataset(MixtureJsonLineDataset):
    """Extends JsonLineDataset to return user_id and seen_in_train flag."""
    def __init__(self, path: str, pad_id: int, uid_to_index: Dict[str, int]):
        super().__init__(path)
        self.pad_id       = pad_id
        self.uid_to_index = uid_to_index

    def __getitem__(self, idx: int):
        row     = super().__getitem__(idx)
        seq_raw = row["AggregateInput"]
        if isinstance(seq_raw, list):
            seq_str = seq_raw[0] if len(seq_raw) == 1 and isinstance(seq_raw[0], str) else " ".join(map(str, seq_raw))
        else:
            seq_str = str(seq_raw)
        toks = []
        for t in seq_str.strip().split():
            try:   toks.append(int(t))
            except ValueError: toks.append(self.pad_id)
        uid_raw  = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        uid      = flat_uid(uid_raw)
        uid_norm = normalize_uid(uid)
        user_id  = self.uid_to_index.get(uid_norm, 0)
        return {
            "uid":           uid,
            "x":             torch.tensor(toks, dtype=torch.long),
            "user_id":       torch.tensor(user_id, dtype=torch.long),
            "seen_in_train": torch.tensor(uid_norm in self.uid_to_index and user_id > 0, dtype=torch.bool),
        }


def mixture_collate_fn(pad_id: int):
    def _inner(batch):
        uids          = [b["uid"]           for b in batch]
        user_ids      = torch.stack([b["user_id"]       for b in batch])
        seen_in_train = torch.stack([b["seen_in_train"] for b in batch])
        lens          = [len(b["x"]) for b in batch]
        lengths       = torch.tensor(lens, dtype=torch.long)
        Lmax = max(lens)
        X = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        for i, (item, L) in enumerate(zip(batch, lens)):
            X[i, :L] = item["x"]
        return {"uid": uids, "x": X, "user_id": user_ids,
                "seen_in_train": seen_in_train, "lengths": lengths}
    return _inner


def build_uid_to_index_for_mixture(
    fold_spec_uri: str,
    fold_id: int,
    train_file: str,
) -> Tuple[Dict[str, int], int]:
    """Rebuild the training UID→index map exactly as training did."""
    spec          = load_json_from_local_or_s3(fold_spec_uri)
    test_uids     = {u for u, f in spec["assignment"].items() if f == fold_id}
    trainval_uids = {u for u in spec["assignment"] if u not in test_uids}
    raw           = mixture_load_json_dataset(train_file, keep_uids=trainval_uids)
    unique_uids   = sorted({normalize_uid(rec.get("uid", "")) for rec in raw})
    uid_to_index  = {u: i + 1 for i, u in enumerate(unique_uids)}   # 0 = UNK
    num_users     = len(uid_to_index) + 1
    print(f"[INFO] Mixture uid_to_index: {num_users} users (UNK=0 + {len(uid_to_index)} training users)")
    return uid_to_index, num_users


def extract_ckpt_num_users(state_dict: dict) -> Optional[int]:
    for k, v in state_dict.items():
        if "gate.logits.weight" in k:
            return int(v.shape[0])
    return None


def forward_with_gate_mode(model, x, user_ids, gate_mode: str, return_hidden: bool = False):
    try:
        return model(x, user_ids, projection_gate_mode=gate_mode, return_hidden=return_hidden)
    except TypeError:
        pass
    mm  = _unwrap_model(model)
    old = None
    if hasattr(mm, "gate") and hasattr(mm.gate, "use_mean_gate"):
        old = mm.gate.use_mean_gate
    try:
        if old is not None:
            mm.gate.use_mean_gate = (gate_mode == "mean")
        return model(x, user_ids, return_hidden=return_hidden)
    finally:
        if old is not None:
            mm.gate.use_mean_gate = old


def mixture_infer_batch(
    model, x, user_ids, seen_mask, ai_rate,
    calibration_mode, calibrator, logit_bias_9, device,
):
    """Returns (B, n_decision_slots, 9) probabilities using per-user gate routing."""
    B  = x.size(0)
    mm = _unwrap_model(model)
    out: Optional[torch.Tensor] = None

    def _infer_subset(idx, gate_mode):
        nonlocal out
        if len(idx) == 0:
            return
        x_sub, u_sub = x[idx], user_ids[idx]
        _, hidden = forward_with_gate_mode(model, x_sub, u_sub, gate_mode, return_hidden=True)
        proj_result = mm.projection(
            hidden, user_idx=u_sub, gate_mode=gate_mode,
            return_alpha=True, return_head_logits=True,
        )
        _, alpha, head_logits = proj_result       # alpha (B,H), head_logits (B,T,H,V)
        pos = torch.arange(ai_rate - 1, x_sub.size(1), ai_rate, device=device)
        if head_logits.size(1) == x_sub.size(1):
            head_logits = head_logits[:, pos, :, :]
        head_logits_dec = head_logits[..., 1:10]  # (B, slots, H, 9)
        alpha_bt        = alpha[:, None, :, None]  # (B, 1, H, 1)
        if calibrator is not None and calibration_mode == "calibrator":
            mixed   = (alpha_bt * head_logits_dec).sum(dim=2)
            prob_dec = calibrator(mixed)
        elif logit_bias_9 is not None and calibration_mode == "analytic":
            bias     = logit_bias_9.to(device=device, dtype=head_logits_dec.dtype)
            corrected = F.softmax(head_logits_dec - bias, dim=-1)
            prob_dec  = (alpha_bt * corrected).sum(dim=2)
        else:
            prob_dec = (alpha_bt * F.softmax(head_logits_dec, dim=-1)).sum(dim=2)
        if out is None:
            out = torch.empty((B, prob_dec.size(1), 9), dtype=prob_dec.dtype, device=device)
        out[idx] = prob_dec

    _infer_subset(torch.nonzero(seen_mask,  as_tuple=False).squeeze(-1), "user")
    _infer_subset(torch.nonzero(~seen_mask, as_tuple=False).squeeze(-1), "mean")
    return out


class MixtureAdapter(BaseAdapter):
    def __init__(self, spec: Dict[str, Any], args: argparse.Namespace):
        super().__init__(spec, args)
        if MIXTURE_IMPORT_ERROR is not None:
            raise RuntimeError("Mixture model dependencies unavailable") from MIXTURE_IMPORT_ERROR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt   = Path(spec["ckpt"])
        self.hp     = parse_hp_from_ckpt_name(self.ckpt)
        self.cfg    = mixture_get_config()
        self.cfg["ai_rate"]    = int(spec.get("ai_rate", 15))
        self.cfg["batch_size"] = int(spec.get("batch_size", 2))
        self.cfg["seq_len_ai"] = self.cfg["seq_len_tgt"] * self.cfg["ai_rate"]
        if spec.get("train_file"):
            self.cfg["filepath"] = spec["train_file"]
        tok_path = Path(self.cfg["model_folder"]) / "tokenizer_tgt.json"
        tok_tgt  = (Tokenizer.from_file(str(tok_path)) if tok_path.exists()
                    else mixture_build_tok())
        self.pad_id      = tok_tgt.token_to_id("[PAD]")
        self.special_ids = [self.pad_id, 10, 11, 12, 57, 58]
        self.data_path   = mixture_ensure_jsonl(args.data)
        self.feat_path   = Path(spec["feat_xlsx"])
        self.calibration = spec.get("calibration", "none")
        self.calibrator   = None
        self.logit_bias_9 = None
        self.fold_spec    = spec.get("fold_spec", "s3://productgptbucket/folds/productgptfolds.json")
        self.uid_to_index, self.num_users = build_uid_to_index_for_mixture(
            self.fold_spec, self.hp["fold_id"], self.cfg["filepath"]
        )
        self.model = self._build_model()
        self._load_calibration()
        # build mean gate from training users
        train_user_ids = sorted(set(self.uid_to_index.values()))
        self.model.set_projection_mean_gate_from_train_users(train_user_ids)
        if hasattr(_unwrap_model(self.model).gate, "update_mean_gate"):
            _unwrap_model(self.model).gate.update_mean_gate()
        print(f"[INFO] {self.name}: mean gate built from {len(train_user_ids)} training users")

    def _build_model(self):
        model = mixture_build_transformer(
            vocab_size_tgt=self.cfg["vocab_size_tgt"],
            vocab_size_src=self.cfg["vocab_size_src"],
            max_seq_len=self.cfg["seq_len_ai"],
            d_model=self.hp["d_model"],
            n_layers=self.hp["N"],
            n_heads=self.hp["num_heads"],
            d_ff=self.hp["d_ff"],
            dropout=0.0,
            nb_features=self.hp["nb_features"],
            kernel_type=self.cfg["kernel_type"],
            feature_tensor=load_feature_tensor(self.feat_path),
            special_token_ids=self.special_ids,
            num_users=self.num_users,
        ).to(self.device).eval()
        state  = torch.load(self.ckpt, map_location=self.device)
        raw_sd = state.get("model_state_dict", state.get("module", state))
        clean_sd = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in raw_sd.items()
            if "weight_fake_quant" not in k and "activation_post_process" not in k
        }
        clean_sd.pop("projection.output_head.mean_alpha_buffer", None)
        model.load_state_dict(clean_sd, strict=False)
        # reconcile num_users if checkpoint differs
        tmp_sd = raw_sd
        ckpt_nu = extract_ckpt_num_users(tmp_sd)
        if ckpt_nu is not None and ckpt_nu != self.num_users:
            print(f"[WARN] {self.name}: num_users mismatch (built={self.num_users}, ckpt={ckpt_nu}). Truncating.")
            self.uid_to_index = {u: idx for u, idx in self.uid_to_index.items() if idx < ckpt_nu}
            self.num_users    = ckpt_nu
        self._state = state
        return model

    def _load_calibration(self):
        if self.calibration == "calibrator":
            stem     = self.ckpt.stem.replace("FullProductGPT_", "")
            cal_path = Path(self.spec.get("calibrator_ckpt") or (self.ckpt.parent / f"calibrator_{stem}.pt"))
            self.calibrator = load_calibrator_from_path(cal_path, self.device)
            if self.calibrator is None:
                print(f"[WARN] {self.name}: calibrator not found; falling back to none")
                self.calibration = "none"
        elif self.calibration == "analytic":
            if "logit_bias_9" in self._state:
                self.logit_bias_9 = torch.tensor(self._state["logit_bias_9"], device=self.device, dtype=torch.float32)
            else:
                w = self.hp["weight"]
                self.logit_bias_9 = torch.zeros(9, device=self.device, dtype=torch.float32)
                if w != 1:
                    self.logit_bias_9[8] = math.log(w)

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        ds     = MixturePredictDataset(self.data_path, pad_id=self.pad_id, uid_to_index=self.uid_to_index)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=False,
                            collate_fn=mixture_collate_fn(self.pad_id))
        seen_count = unseen_count = 0
        with torch.no_grad():
            for batch in loader:
                x             = batch["x"].to(self.device)
                user_ids      = batch["user_id"].to(self.device)
                seen_mask     = batch["seen_in_train"].to(self.device)
                lengths       = batch["lengths"].tolist()
                uids          = batch["uid"]
                seen_count   += int(seen_mask.sum().item())
                unseen_count += int((~seen_mask).sum().item())
                probs = mixture_infer_batch(
                    model=self.model, x=x, user_ids=user_ids, seen_mask=seen_mask,
                    ai_rate=self.cfg["ai_rate"],
                    calibration_mode=self.calibration,
                    calibrator=self.calibrator,
                    logit_bias_9=self.logit_bias_9,
                    device=self.device,
                )
                yield {
                    "uid":       uids,
                    "lens":      lengths,
                    "probs_dec_9": probs.detach().cpu().numpy(),
                    "ai_rate":   self.cfg["ai_rate"],
                }
        print(f"[INFO] {self.name}: seen={seen_count} unseen(mean-gate)={unseen_count}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRU / LSTM adapters
# ═══════════════════════════════════════════════════════════════════════════════
class SequenceFeatureDataset(Dataset):
    def __init__(self, json_path: Path, input_dim: int):
        raw = json.loads(Path(json_path).read_text())
        if not isinstance(raw, list):
            raise ValueError("Input JSON must be a list of objects")
        self.rows      = raw
        self.input_dim = input_dim
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, idx: int):
        rec      = self.rows[idx]
        uid      = flat_uid(rec["uid"])
        feat_str = rec["AggregateInput"][0] if isinstance(rec["AggregateInput"], list) else rec["AggregateInput"]
        flat     = [0.0 if tok == "NA" else float(tok) for tok in str(feat_str).strip().split()]
        T        = len(flat) // self.input_dim
        x        = torch.zeros((1, self.input_dim), dtype=torch.float32) if T == 0 else \
                   torch.tensor(flat[:T * self.input_dim], dtype=torch.float32).view(T, self.input_dim)
        T        = max(T, 1)
        return {"uid": uid, "x": x, "T": T}


def seq_collate(batch) -> Dict[str, Any]:
    uids  = [b["uid"] for b in batch]
    xs    = [b["x"]   for b in batch]
    Ts    = [b["T"]   for b in batch]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return {"uid": uids, "x": x_pad, "T": Ts}


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)


class LSTMDecisionModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def _load_rnn_state(ckpt_path: Path, device: torch.device) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    sd = state if isinstance(state, dict) and all(isinstance(k, str) for k in state) \
         else state.get("model_state_dict", state)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


class GRUAdapter(BaseAdapter):
    def __init__(self, spec, args):
        super().__init__(spec, args)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim   = int(spec.get("input_dim", 15))
        self.hidden_size = int(spec["hidden_size"])
        self.batch_size  = int(spec.get("batch_size", 128))
        self.data_path   = Path(args.data)
        self.model = GRUClassifier(self.input_dim, self.hidden_size).to(self.device).eval()
        self.model.load_state_dict(_load_rnn_state(Path(spec["ckpt"]), self.device), strict=True)

    def predict_batches(self):
        loader = DataLoader(SequenceFeatureDataset(self.data_path, self.input_dim),
                            batch_size=self.batch_size, shuffle=False, collate_fn=seq_collate)
        with torch.no_grad():
            for batch in loader:
                probs9 = F.softmax(self.model(batch["x"].to(self.device))[..., 1:], dim=-1)
                yield {"uid": batch["uid"], "lens": batch["T"],
                       "probs_dec_9": probs9.detach().cpu().numpy(), "ai_rate": 1}


class LSTMAdapter(BaseAdapter):
    def __init__(self, spec, args):
        super().__init__(spec, args)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim   = int(spec.get("input_dim", 15))
        self.hidden_size = int(spec["hidden_size"])
        self.batch_size  = int(spec.get("batch_size", 128))
        self.data_path   = Path(args.data)
        self.model = LSTMDecisionModel(self.input_dim, self.hidden_size).to(self.device).eval()
        self.model.load_state_dict(_load_rnn_state(Path(spec["ckpt"]), self.device), strict=True)

    def predict_batches(self):
        loader = DataLoader(SequenceFeatureDataset(self.data_path, self.input_dim),
                            batch_size=self.batch_size, shuffle=False, collate_fn=seq_collate)
        with torch.no_grad():
            for batch in loader:
                probs9 = F.softmax(self.model(batch["x"].to(self.device))[..., 1:], dim=-1)
                yield {"uid": batch["uid"], "lens": batch["T"],
                       "probs_dec_9": probs9.detach().cpu().numpy(), "ai_rate": 1}


def make_adapter(spec: Dict[str, Any], args: argparse.Namespace) -> BaseAdapter:
    family = spec["model_family"].lower()
    if family == "productgpt": return ProductGPTAdapter(spec, args)
    if family == "mixture":    return MixtureAdapter(spec, args)
    if family == "gru":        return GRUAdapter(spec, args)
    if family == "lstm":       return LSTMAdapter(spec, args)
    raise ValueError(f"Unsupported model_family: {family}")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    adapter: BaseAdapter,
    label_dict: Dict[str, Dict[str, List[int]]],
    which_split,
    args: argparse.Namespace,
    output_dir: Path,
) -> EvalResult:
    model_dir = output_dir / adapter.name
    model_dir.mkdir(parents=True, exist_ok=True)

    scores       = defaultdict(lambda: {"y": [], "p": []})
    multi_scores = defaultdict(lambda: {"y": [], "p": []})
    length_note  = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    pred_path   = model_dir / f"preds_{adapter.name}.jsonl.gz"
    pred_writer = smart_open_w(pred_path) if args.save_preds else None

    with (pred_writer or nullcontext(None)) as writer:
        for batch in adapter.predict_batches():
            uids         = batch["uid"]
            lens         = batch["lens"]
            probs_dec_9  = batch["probs_dec_9"]
            ai_rate      = int(batch["ai_rate"])

            for i, uid in enumerate(uids):
                actual_len      = int(lens[i])
                n_valid_slots   = actual_len // ai_rate if ai_rate > 1 else actual_len
                probs_seq_np    = probs_dec_9[i, :n_valid_slots]

                if writer is not None:
                    writer.write(json.dumps({"uid": uid, "probs": np.round(probs_seq_np, 6).tolist()}) + "\n")

                lbl_info = label_dict.get(uid)
                if lbl_info is None:
                    reject += 1
                    continue

                L_pred = len(probs_seq_np)
                L_lbl  = len(lbl_info["label"])
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
                    y_bin   = np.isin(y_arr, list(TASK_POSSETS[task])).astype(np.int8)
                    col_idx = [j - 1 for j in pos_classes]
                    p_bin   = p_arr[:, col_idx].sum(axis=1)
                    for g_idx, g_name in enumerate(GROUP_ORDER):
                        mask = group_idx == g_idx
                        if not mask.any():
                            continue
                        key = (task, g_name, split_tag)
                        scores[key]["y"].extend(y_bin[mask].tolist())
                        scores[key]["p"].extend(p_bin[mask].tolist())
                accept += 1

    # ── Binary metrics ─────────────────────────────────────────────────────────
    bin_rows = []
    for task in BIN_TASKS:
        for grp in GROUP_ORDER:
            for spl in SPLIT_ORDER:
                y = scores[(task, grp, spl)]["y"]
                p = scores[(task, grp, spl)]["p"]
                if not y:
                    continue
                if len(set(y)) < 2:
                    auc = hit = f1 = auprc = np.nan
                else:
                    y_arr = np.asarray(y, dtype=int)
                    p_arr = np.asarray(p, dtype=float)
                    y_hat = (p_arr >= args.thresh).astype(int)
                    auc   = roc_auc_score(y_arr, p_arr)
                    hit   = accuracy_score(y_arr, y_hat)
                    f1    = f1_score(y_arr, y_hat, zero_division=0)
                    auprc = average_precision_score(y_arr, p_arr)
                bin_rows.append({
                    "Task":       task,
                    "TaskPretty": PAPER_AUC_TASKS.get(task, task),
                    "Group":      grp, "Split": spl,
                    "AUC":   round(float(auc),   4) if not np.isnan(auc)  else np.nan,
                    "Hit":   round(float(hit),   4) if not np.isnan(hit)  else np.nan,
                    "F1":    round(float(f1),    4) if not np.isnan(f1)   else np.nan,
                    "AUPRC": round(float(auprc), 4) if not np.isnan(auprc) else np.nan,
                })
    binary_metrics = pd.DataFrame(bin_rows)

    # ── Multiclass metrics ─────────────────────────────────────────────────────
    multi_rows = []
    for grp in GROUP_ORDER:
        for spl in SPLIT_ORDER:
            y = multi_scores[(grp, spl)]["y"]
            p = multi_scores[(grp, spl)]["p"]
            if not y:
                continue
            y_arr = np.asarray(y, dtype=np.int64)
            p_arr = np.vstack(p)
            multi_rows.append({"Group": grp, "Split": spl,
                                **compute_multiclass_metrics(y_arr, p_arr)})
    multiclass_metrics = pd.DataFrame(multi_rows)

    # ── Per-class metrics ──────────────────────────────────────────────────────
    perclass_rows = []
    for grp in GROUP_ORDER:
        for spl in SPLIT_ORDER:
            y = multi_scores[(grp, spl)]["y"]
            p = multi_scores[(grp, spl)]["p"]
            if not y:
                continue
            y_arr = np.asarray(y, dtype=np.int64)
            p_arr = np.vstack(p)
            for c, cm in compute_perclass_metrics(y_arr, p_arr).items():
                perclass_rows.append({"Group": grp, "Split": spl, "Class": c, **cm})
    perclass_metrics = pd.DataFrame(perclass_rows)

    selected_binary_auc = (
        binary_metrics[binary_metrics["Task"].isin(PAPER_AUC_TASKS)]
        .pivot_table(index=["Group", "TaskPretty"], columns="Split", values="AUC")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, PAPER_AUC_ORDER],
                                                  names=["Group", "TaskPretty"]))
        .reindex(columns=SPLIT_ORDER).round(4)
    )

    # ── Save per-model CSVs ────────────────────────────────────────────────────
    binary_metrics.to_csv(model_dir / "binary_metrics_long.csv", index=False)
    multiclass_metrics.to_csv(model_dir / "multiclass_metrics_long.csv", index=False)
    perclass_metrics.to_csv(model_dir / "perclass_metrics_long.csv", index=False)
    selected_binary_auc.to_csv(model_dir / "selected_binary_auc.csv")

    # ── Per-model combined report (same layout as individual eval scripts) ─────
    BIN_METRIC_COLS  = ["BinaryAUC", "BinaryAUPRC"]
    COMBINED_METRICS = MULTICLASS_METRICS + PERCLASS_METRICS + BIN_METRIC_COLS
    COMBINED_COLS    = pd.MultiIndex.from_product([COMBINED_METRICS, SPLIT_ORDER],
                                                  names=["Metric", "Split"])
    all_binary = binary_metrics.copy()
    all_binary["Label"] = all_binary["Task"].map(lambda t: PAPER_AUC_TASKS.get(t, t))

    for grp in GROUP_ORDER:
        rows_dict: Dict[str, Dict[Tuple, float]] = {}
        # aggregate
        agg_sub = multiclass_metrics[multiclass_metrics["Group"] == grp]
        for _, row in agg_sub.iterrows():
            spl = row["Split"]
            for m in MULTICLASS_METRICS:
                rows_dict.setdefault("Aggregate", {})[m, spl] = row[m]
        # per-class
        pc_sub = perclass_metrics[perclass_metrics["Group"] == grp]
        for _, row in pc_sub.iterrows():
            lbl = f"Class {int(row['Class'])}"
            for m in PERCLASS_METRICS:
                rows_dict.setdefault(lbl, {})[m, row["Split"]] = row[m]
        # binary
        bin_sub = all_binary[all_binary["Group"] == grp]
        for _, row in bin_sub.iterrows():
            lbl = row["Label"]
            rows_dict.setdefault(lbl, {})["BinaryAUC",   row["Split"]] = row["AUC"]
            rows_dict.setdefault(lbl, {})["BinaryAUPRC", row["Split"]] = row["AUPRC"]
        # assemble
        class_labels  = [f"Class {c}" for c in NINE_CLASSES]
        binary_labels = bin_sub[["Label"]].drop_duplicates()["Label"].tolist()
        seen, ordered = set(), []
        for lbl in ["Aggregate"] + class_labels + binary_labels:
            if lbl not in seen and lbl in rows_dict:
                ordered.append(lbl); seen.add(lbl)
        tbl = pd.DataFrame(rows_dict, dtype=float).T.reindex(ordered)
        tbl.columns = pd.MultiIndex.from_tuples(tbl.columns, names=["Metric", "Split"])
        tbl = tbl.reindex(columns=COMBINED_COLS).round(4)
        tbl.index.name = "Row"
        sep = pd.DataFrame([[np.nan] * len(COMBINED_COLS)], columns=COMBINED_COLS,
                           index=["─────────────────"])
        sep.index.name = "Row"
        agg_rows   = ["Aggregate"]
        class_rows = [l for l in ordered if l.startswith("Class")]
        bin_rows_l = [l for l in ordered if l not in agg_rows and l not in class_rows]
        parts = []
        if agg_rows:   parts += [tbl.loc[agg_rows],  sep]
        if class_rows: parts += [tbl.loc[class_rows], sep]
        if bin_rows_l: parts += [tbl.loc[bin_rows_l]]
        combined_tbl = pd.concat(parts)
        combined_tbl.to_csv(model_dir / f"combined_report_{grp.lower()}.csv")

    info = {
        "accepted_users": accept,
        "missing_labels": reject,
        "length_note": dict(length_note),
        "coverage": {k: len(v) for k, v in accept_users.items()},
        "model_family": adapter.model_family,
    }
    with open(model_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[INFO] {adapter.name}: accepted={accept}, rejected={reject}, "
          f"val={len(accept_users.get('val', set()))}, test={len(accept_users.get('test', set()))}")
    return EvalResult(
        model_name=adapter.name,
        binary_metrics=binary_metrics,
        multiclass_metrics=multiclass_metrics,
        perclass_metrics=perclass_metrics,
        selected_binary_auc=selected_binary_auc,
        info=info,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-model comparison
# ═══════════════════════════════════════════════════════════════════════════════
def build_comparison_tables(
    results: List[EvalResult], compare_on: str, out_dir: Path
) -> Dict[str, pd.DataFrame]:

    # 1. Multiclass metrics by group × metric
    multi_rows = []
    for r in results:
        df = r.multiclass_metrics[r.multiclass_metrics["Split"] == compare_on]
        for _, row in df.iterrows():
            for metric in MULTICLASS_METRICS:
                multi_rows.append({"Model": r.model_name, "Group": row["Group"],
                                   "Metric": metric, "Value": row[metric]})
    multi_long = pd.DataFrame(multi_rows)
    multiclass_comparison = (
        multi_long.pivot_table(index=["Group", "Metric"], columns="Model", values="Value")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, MULTICLASS_METRICS],
                                                  names=["Group", "Metric"]))
        .round(4)
    )

    # 2. Per-class AUC_OvR by group × class
    pc_rows = []
    for r in results:
        df = r.perclass_metrics[r.perclass_metrics["Split"] == compare_on]
        for _, row in df.iterrows():
            pc_rows.append({"Model": r.model_name, "Group": row["Group"],
                            "Class": int(row["Class"]), "AUC_OvR": row["AUC_OvR"],
                            "AUPRC": row["AUPRC"], "F1": row["F1"], "Support": row["Support"]})
    pc_long = pd.DataFrame(pc_rows)
    perclass_auc_comparison = (
        pc_long.pivot_table(index=["Group", "Class"], columns="Model", values="AUC_OvR")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, NINE_CLASSES],
                                                  names=["Group", "Class"]))
        .round(4)
    )
    perclass_f1_comparison = (
        pc_long.pivot_table(index=["Group", "Class"], columns="Model", values="F1")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, NINE_CLASSES],
                                                  names=["Group", "Class"]))
        .round(4)
    )

    # 3. Selected binary AUC by group × task
    bin_rows = []
    for r in results:
        df = r.binary_metrics
        df = df[(df["Split"] == compare_on) & (df["Task"].isin(PAPER_AUC_TASKS))]
        for _, row in df.iterrows():
            bin_rows.append({"Model": r.model_name, "Group": row["Group"],
                             "TaskPretty": row["TaskPretty"], "AUC": row["AUC"]})
    bin_long = pd.DataFrame(bin_rows)
    binary_auc_comparison = (
        bin_long.pivot_table(index=["Group", "TaskPretty"], columns="Model", values="AUC")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, PAPER_AUC_ORDER],
                                                  names=["Group", "TaskPretty"]))
        .round(4)
    )

    # 4. Summary tables
    summary_rows = []
    for r in results:
        bm  = r.binary_metrics
        mm  = r.multiclass_metrics
        sel = bm[(bm["Split"] == compare_on) & (bm["Task"].isin(PAPER_AUC_TASKS))]
        for grp in GROUP_ORDER:
            gsel = sel[sel["Group"] == grp]
            gm   = mm[(mm["Split"] == compare_on) & (mm["Group"] == grp)]
            row_base = {"Model": r.model_name, "Group": grp,
                        "AvgBinaryAUC": round(float(gsel["AUC"].mean()), 4) if not gsel.empty else np.nan}
            if not gm.empty:
                for m in MULTICLASS_METRICS:
                    row_base[m] = gm.iloc[0][m]
            summary_rows.append(row_base)
        summary_rows.append({
            "Model": r.model_name, "Group": "Overall",
            "AvgBinaryAUC": round(float(sel["AUC"].mean()), 4) if not sel.empty else np.nan,
        })
    summary_long = pd.DataFrame(summary_rows)

    def _pivot_summary(metric):
        sub = summary_long[summary_long[metric].notna()] if metric in summary_long else pd.DataFrame()
        if sub.empty:
            return pd.DataFrame()
        return (
            sub.pivot_table(index="Group", columns="Model", values=metric)
            .reindex(index=GROUP_ORDER + ["Overall"])
            .round(4)
        )

    tables = {
        "multiclass_comparison":    multiclass_comparison,
        "binary_auc_comparison":    binary_auc_comparison,
        "perclass_auc_comparison":  perclass_auc_comparison,
        "perclass_f1_comparison":   perclass_f1_comparison,
        "summary_avg_binary_auc":   _pivot_summary("AvgBinaryAUC"),
        "summary_macrof1":          _pivot_summary("MacroF1"),
        "summary_top1acc":          _pivot_summary("Top1Acc"),
        "summary_mcc":              _pivot_summary("MCC"),
        "summary_logloss":          _pivot_summary("LogLoss"),
    }
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(out_dir / f"{name}_{compare_on}.csv")
    return tables


def save_markdown_summary(tables: Dict[str, pd.DataFrame], out_dir: Path, compare_on: str):
    md_path = out_dir / f"comparison_summary_{compare_on}.md"
    with open(md_path, "w") as f:
        f.write(f"# Unified model comparison ({compare_on})\n\n")
        for title, df in tables.items():
            if df.empty:
                continue
            f.write(f"## {title}\n\n")
            f.write(df.fillna("NA").to_markdown())
            f.write("\n\n")


def maybe_upload_outputs(out_dir: Path, s3_prefix: str, fold_id: int):
    if not s3_prefix:
        return
    effective = s3_join_folder(s3_prefix, f"fold{fold_id}") if fold_id >= 0 else s3_prefix
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel  = path.relative_to(out_dir)
            dest = s3_join(effective, str(rel).replace(os.sep, "/"))
            s3_upload_file(path, dest)
            print(f"[S3] uploaded: {dest}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)
    models = config["models"] if isinstance(config, dict) else config

    label_dict, records = load_labels(Path(args.labels))
    which_split, split_info, _, _ = make_splitter(records, args.uids_val, args.uids_test, args.seed)
    print(f"[INFO] split mode: {split_info}")

    results: List[EvalResult] = []
    for spec in models:
        print(f"\n{'='*60}")
        print(f"[INFO] Evaluating: {spec['name']}  ({spec['model_family']})")
        print(f"{'='*60}")
        adapter = make_adapter(spec, args)
        res     = evaluate_model(adapter, label_dict, which_split, args, output_dir)
        results.append(res)

    print(f"\n{'='*60}")
    print("[INFO] Building cross-model comparison tables …")
    tables = build_comparison_tables(results, args.compare_on, output_dir)
    save_markdown_summary(tables, output_dir, args.compare_on)
    maybe_upload_outputs(output_dir, args.s3, args.fold_id)

    # ── Console summary ────────────────────────────────────────────────────────
    def _p(title, df):
        print(f"\n=============  {title}  =======================")
        print(df.fillna(" NA") if not df.empty else "(empty)")
        print("============================================================")

    _p(f"MACRO-F1 COMPARISON ({args.compare_on})",          tables["summary_macrof1"])
    _p(f"TOP-1 ACCURACY COMPARISON ({args.compare_on})",    tables["summary_top1acc"])
    _p(f"AVG SELECTED BINARY AUC ({args.compare_on})",      tables["summary_avg_binary_auc"])
    _p(f"BINARY AUC BY GROUP × TASK ({args.compare_on})",   tables["binary_auc_comparison"])
    _p(f"PER-CLASS AUC_OvR ({args.compare_on})",            tables["perclass_auc_comparison"])
    _p(f"MULTICLASS FULL METRICS ({args.compare_on})",      tables["multiclass_comparison"])


if __name__ == "__main__":
    main()