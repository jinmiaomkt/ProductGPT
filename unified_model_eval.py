#!/usr/bin/env python3
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

# Optional ProductGPT imports. Only required when model_family=productgpt.
try:
    from dataset4_productgpt import load_json_dataset as productgpt_load_json_dataset
    from config4 import get_config as productgpt_get_config
    from model4_decoderonly_feature_performer import build_transformer as productgpt_build_transformer
    from train1_decision_only_performer_aws import _ensure_jsonl as productgpt_ensure_jsonl
    from train1_decision_only_performer_aws import JsonLineDataset as ProductGPTJsonLineDataset
    from train1_decision_only_performer_aws import _build_tok as productgpt_build_tok
    from tokenizers import Tokenizer
    PRODUCTGPT_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment-specific
    productgpt_load_json_dataset = None
    productgpt_get_config = None
    productgpt_build_transformer = None
    productgpt_ensure_jsonl = None
    ProductGPTJsonLineDataset = object
    productgpt_build_tok = None
    Tokenizer = None
    PRODUCTGPT_IMPORT_ERROR = exc


# =============================
# Constants and metric helpers
# =============================
GROUP_ORDER = ["Calibration", "HoldoutA", "HoldoutB"]
SPLIT_ORDER = ["val", "test"]
NINE_CLASSES = list(range(1, 10))
PERCLASS_METRICS = ["AUC_OvR", "AUPRC", "F1", "Support"]
MULTICLASS_METRICS = [
    "MacroOvR_AUC",
    "MacroAUPRC",
    "MacroF1",
    "MCC",
    "LogLoss",
    "Top1Acc",
    "Top2Acc",
]
BIN_TASKS = {
    "BuyNone": [9],
    "BuyOne": [1, 3, 5, 7],
    "BuyTen": [2, 4, 6, 8],
    "BuyRegular": [1, 2],
    "BuyFigure": [3, 4, 5, 6],
    "BuyWeapon": [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}
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
FEATURE_COLS = [
    "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
    "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow", "WeaponTypeMagic", "WeaponTypePolearm",
    "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire", "EthnicityThunder", "EthnicityWind",
    "GenderFemale", "GenderMale", "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
    "type_figure", "MinimumAttack", "MaximumAttack", "MinSpecialEffect", "MaxSpecialEffect", "SpecialEffectEfficiency",
    "SpecialEffectExpertise", "SpecialEffectAttack", "SpecialEffectSuper", "SpecialEffectRatio", "SpecialEffectPhysical",
    "SpecialEffectLife", "LTO",
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


# =============================
# Generic utilities
# =============================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified evaluator for ProductGPT, LSTM, and GRU with cross-model comparison."
    )
    p.add_argument("--config", required=True, help="JSON file describing all models to evaluate")
    p.add_argument("--data", required=True, help="Common data path used by all model adapters")
    p.add_argument("--labels", required=True, help="Label JSON used by all models")
    p.add_argument("--uids-val", default="", help="Validation UID file (local or s3://)")
    p.add_argument("--uids-test", default="", help="Test UID file (local or s3://)")
    p.add_argument("--fold-id", type=int, default=-1, help="Optional fold id used in output paths")
    p.add_argument("--seed", type=int, default=33, help="Fallback split seed when UID lists are omitted")
    p.add_argument("--thresh", type=float, default=0.5, help="Threshold for binary Hit/F1")
    p.add_argument("--compare-on", choices=["test", "val"], default="test")
    p.add_argument("--output-dir", required=True, help="Local directory for all outputs")
    p.add_argument("--s3", default="", help="Optional S3 prefix for uploading all outputs")
    p.add_argument("--save-preds", action="store_true", help="Save per-model prediction jsonl.gz files")
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
    if "/" in no_scheme:
        bucket, key = no_scheme.split("/", 1)
    else:
        bucket, key = no_scheme, ""
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
    assert not s3_uri_full.endswith("/"), f"S3 object key must not end with '/': {s3_uri_full}"
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
    n_val = int(0.1 * n)
    tr_i, va_i, te_i = random_split(range(n), [n_train, n_val, n - n_train - n_val], generator=g)
    val_uid = {flat_uid(records[i]["uid"]) for i in va_i.indices}
    test_uid = {flat_uid(records[i]["uid"]) for i in te_i.indices}

    def which(u: str) -> str:
        if u in val_uid:
            return "val"
        if u in test_uid:
            return "test"
        return "train"

    return which


def make_splitter(records: Sequence[dict], uids_val_path: str, uids_test_path: str, seed: int):
    uids_val = load_uid_set(uids_val_path) if uids_val_path else set()
    uids_test = load_uid_set(uids_test_path) if uids_test_path else set()
    if uids_val or uids_test:
        if not (uids_val and uids_test):
            raise ValueError("Provide BOTH --uids-val and --uids-test (or neither).")
        overlap = uids_val & uids_test
        if overlap:
            raise ValueError(f"UID overlap between val and test: {sorted(list(overlap))[:5]}")

        def which(u: str) -> str:
            if u in uids_val:
                return "val"
            if u in uids_test:
                return "test"
            return "train"

        split_info = {"mode": "exact_uids", "val_count": len(uids_val), "test_count": len(uids_test)}
        return which, split_info, uids_val, uids_test

    which = build_splits(records, seed=seed)
    split_info = {"mode": "seeded_random_split", "seed": seed}
    return which, split_info, set(), set()


def period_group(idx_h: int, feat_h: int) -> str:
    if feat_h == 0:
        return "Calibration"
    if feat_h == 1 and idx_h == 0:
        return "HoldoutA"
    if idx_h == 1:
        return "HoldoutB"
    return "UNASSIGNED"


def compute_multiclass_metrics(y_arr: np.ndarray, p_arr: np.ndarray) -> Dict[str, float]:
    y_hat = p_arr.argmax(axis=1) + 1
    try:
        macro_ovr_auc = roc_auc_score(
            y_arr,
            p_arr,
            multi_class="ovr",
            average="macro",
            labels=NINE_CLASSES,
        )
    except ValueError:
        macro_ovr_auc = np.nan

    y_bin = label_binarize(y_arr, classes=NINE_CLASSES)
    per_class_ap: List[float] = []
    for c in range(9):
        if y_bin[:, c].sum() == 0:
            continue
        per_class_ap.append(average_precision_score(y_bin[:, c], p_arr[:, c]))
    macro_auprc = float(np.mean(per_class_ap)) if per_class_ap else np.nan

    macro_f1 = f1_score(y_arr, y_hat, labels=NINE_CLASSES, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_arr, y_hat)
    p_clipped = np.clip(p_arr, 1e-7, 1.0)
    ll = log_loss(y_arr, p_clipped, labels=NINE_CLASSES)
    top1 = float(np.mean(y_arr == y_hat))
    top2_indices = np.argsort(p_arr, axis=1)[:, -2:]
    top2_labels = top2_indices + 1
    top2_acc = float(np.mean([y_arr[i] in top2_labels[i] for i in range(len(y_arr))]))
    return {
        "MacroOvR_AUC": round(float(macro_ovr_auc), 4),
        "MacroAUPRC": round(float(macro_auprc), 4),
        "MacroF1": round(float(macro_f1), 4),
        "MCC": round(float(mcc), 4),
        "LogLoss": round(float(ll), 4),
        "Top1Acc": round(float(top1), 4),
        "Top2Acc": round(float(top2_acc), 4),
    }


def compute_perclass_metrics(y_arr: np.ndarray, p_arr: np.ndarray) -> Dict[int, Dict[str, float]]:
    y_hat = p_arr.argmax(axis=1) + 1
    y_bin = label_binarize(y_arr, classes=NINE_CLASSES)
    results: Dict[int, Dict[str, float]] = {}
    for c_idx, c in enumerate(NINE_CLASSES):
        support = int((y_arr == c).sum())
        y_bin_c = y_bin[:, c_idx]
        p_c = p_arr[:, c_idx]
        y_hat_bin_c = (y_hat == c).astype(int)
        if y_bin_c.sum() == 0 or y_bin_c.sum() == len(y_arr):
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_bin_c, p_c)
            except ValueError:
                auc = np.nan
        if y_bin_c.sum() == 0:
            auprc = np.nan
        else:
            auprc = average_precision_score(y_bin_c, p_c)
        f1 = f1_score(y_bin_c, y_hat_bin_c, zero_division=0)
        results[c] = {
            "AUC_OvR": round(float(auc), 4),
            "AUPRC": round(float(auprc), 4),
            "F1": round(float(f1), 4),
            "Support": support,
        }
    return results


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


# =============================
# Adapters
# =============================
class BaseAdapter:
    def __init__(self, spec: Dict[str, Any], args: argparse.Namespace):
        self.spec = spec
        self.args = args
        self.name = spec["name"]
        self.family = spec["family"]

    def build_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class VectorScaling(torch.nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(n_classes))
        self.b = torch.nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits_dec: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.a * logits_dec + self.b, dim=-1)


def load_calibrator(cal_path: Path, device: torch.device) -> Optional[VectorScaling]:
    if not cal_path.exists():
        return None
    cal = VectorScaling(n_classes=9).to(device)
    state = torch.load(cal_path, map_location=device)
    cal.a.data = state["a"].to(device)
    cal.b.data = state["b"].to(device)
    return cal


class ProductGPTPredictDataset(ProductGPTJsonLineDataset):
    def __init__(self, path: str, pad_id: int):
        super().__init__(path)
        self.pad_id = pad_id

    def to_int_or_pad(self, tok: str) -> int:
        try:
            return int(tok)
        except ValueError:
            return self.pad_id

    def __getitem__(self, idx: int):
        row = super().__getitem__(idx)
        seq_raw = row["AggregateInput"]
        if isinstance(seq_raw, list):
            if len(seq_raw) == 1 and isinstance(seq_raw[0], str):
                seq_str = seq_raw[0]
            else:
                seq_str = " ".join(map(str, seq_raw))
        else:
            seq_str = str(seq_raw)
        toks = [self.to_int_or_pad(t) for t in seq_str.strip().split()]
        uid = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": flat_uid(uid), "x": torch.tensor(toks, dtype=torch.long)}


def productgpt_collate_fn(pad_id: int):
    def _inner(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
            raise RuntimeError(
                "ProductGPT dependencies could not be imported in this environment"
            ) from PRODUCTGPT_IMPORT_ERROR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = Path(spec["ckpt"])
        self.hp = parse_hp_from_ckpt_name(self.ckpt)
        self.cfg = productgpt_get_config()
        self.cfg["ai_rate"] = int(spec.get("ai_rate", 15))
        self.cfg["batch_size"] = int(spec.get("batch_size", 2))
        self.cfg["seq_len_ai"] = self.cfg["seq_len_tgt"] * self.cfg["ai_rate"]
        tok_path = Path(self.cfg["model_folder"]) / "tokenizer_tgt.json"
        self.tok_tgt = Tokenizer.from_file(str(tok_path)) if tok_path.exists() else productgpt_build_tok()
        self.pad_id = self.tok_tgt.token_to_id("[PAD]")
        self.special_ids = [self.pad_id, 10, 11, 12, 57, 58]
        self.data_path = productgpt_ensure_jsonl(args.data)
        self.feat_path = Path(spec["feat_xlsx"])
        self.model = self._build_model()
        self.calibration = spec.get("calibration", "none")
        self.calibrator = None
        self.logit_bias_9 = None
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
        state = torch.load(self.ckpt, map_location=self.device)
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
            cal_path = Path(self.spec.get("calibrator_ckpt") or (self.ckpt.parent / f"calibrator_{self.ckpt.stem.replace('FullProductGPT_', '')}.pt"))
            self.calibrator = load_calibrator(cal_path, self.device)
            if self.calibrator is None:
                print(f"[WARN] {self.name}: calibrator not found at {cal_path}; falling back to none", file=sys.stderr)
                self.calibration = "none"
        elif self.calibration == "analytic":
            if "logit_bias_9" in self._state:
                self.logit_bias_9 = torch.tensor(self._state["logit_bias_9"], device=self.device, dtype=torch.float32)
            else:
                weight_class9 = self.hp["weight"]
                self.logit_bias_9 = torch.zeros(9, device=self.device, dtype=torch.float32)
                if weight_class9 != 1:
                    self.logit_bias_9[8] = math.log(weight_class9)

    def build_dataloader(self) -> DataLoader:
        ds = ProductGPTPredictDataset(self.data_path, pad_id=self.pad_id)
        return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=False, collate_fn=productgpt_collate_fn(self.pad_id))

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        loader = self.build_dataloader()
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                uids = batch["uid"]
                lens = batch["lens"]
                logits_full = self.model(x)
                if x.size(1) < self.cfg["ai_rate"]:
                    pos = torch.empty((0,), dtype=torch.long, device=self.device)
                else:
                    pos = torch.arange(self.cfg["ai_rate"] - 1, x.size(1), self.cfg["ai_rate"], device=self.device)
                logits = logits_full[:, pos, :] if logits_full.size(1) == x.size(1) else logits_full
                if logits.size(-1) == 9:
                    logits_dec_9 = logits
                else:
                    logits_dec_9 = logits[..., 1:10]
                if self.calibration == "calibrator" and self.calibrator is not None:
                    probs_dec_9 = self.calibrator(logits_dec_9)
                elif self.calibration == "analytic" and self.logit_bias_9 is not None:
                    probs_dec_9 = torch.softmax(logits_dec_9 + self.logit_bias_9.view(1, 1, 9), dim=-1)
                else:
                    probs_dec_9 = torch.softmax(logits_dec_9, dim=-1)
                yield {"uid": uids, "lens": lens, "probs_dec_9": probs_dec_9.detach().cpu().numpy(), "ai_rate": self.cfg["ai_rate"]}


class SequenceFeatureDataset(Dataset):
    def __init__(self, json_path: Path, input_dim: int):
        raw = json.loads(Path(json_path).read_text())
        if not isinstance(raw, list):
            raise ValueError("Input JSON must be a list of objects")
        self.rows = raw
        self.input_dim = input_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rec = self.rows[idx]
        uid = flat_uid(rec["uid"])
        feat_str = rec["AggregateInput"][0] if isinstance(rec["AggregateInput"], list) else rec["AggregateInput"]
        flat = [0.0 if tok == "NA" else float(tok) for tok in str(feat_str).strip().split()]
        T = len(flat) // self.input_dim
        if T == 0:
            x = torch.zeros((1, self.input_dim), dtype=torch.float32)
        else:
            x = torch.tensor(flat[: T * self.input_dim], dtype=torch.float32).view(T, self.input_dim)
        return {"uid": uid, "x": x, "T": T}


def seq_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    uids = [b["uid"] for b in batch]
    xs = [b["x"] for b in batch]
    Ts = [b["T"] for b in batch]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return {"uid": uids, "x": x_pad, "T": Ts}


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out)


class LSTMDecisionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)


class GRUAdapter(BaseAdapter):
    def __init__(self, spec: Dict[str, Any], args: argparse.Namespace):
        super().__init__(spec, args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = int(spec.get("input_dim", 15))
        self.hidden_size = int(spec["hidden_size"])
        self.batch_size = int(spec.get("batch_size", 128))
        self.data_path = Path(args.data)
        self.ckpt = Path(spec["ckpt"])
        self.model = GRUClassifier(self.input_dim, self.hidden_size, num_classes=10).to(self.device).eval()
        state = torch.load(self.ckpt, map_location=self.device)
        sd = state if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()) else state.get("model_state_dict", state)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)

    def build_dataloader(self) -> DataLoader:
        ds = SequenceFeatureDataset(self.data_path, input_dim=self.input_dim)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=seq_collate)

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        loader = self.build_dataloader()
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                logits10 = self.model(x)
                probs9 = F.softmax(logits10[..., 1:], dim=-1)
                yield {"uid": batch["uid"], "lens": batch["T"], "probs_dec_9": probs9.detach().cpu().numpy(), "ai_rate": 1}


class LSTMAdapter(BaseAdapter):
    def __init__(self, spec: Dict[str, Any], args: argparse.Namespace):
        super().__init__(spec, args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = int(spec.get("input_dim", 15))
        self.hidden_size = int(spec["hidden_size"])
        self.batch_size = int(spec.get("batch_size", 128))
        self.data_path = Path(args.data)
        self.ckpt = Path(spec["ckpt"])
        self.model = LSTMDecisionModel(self.input_dim, self.hidden_size, num_classes=10).to(self.device).eval()
        state = torch.load(self.ckpt, map_location=self.device)
        sd = state.get("model_state_dict", state)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)

    def build_dataloader(self) -> DataLoader:
        ds = SequenceFeatureDataset(self.data_path, input_dim=self.input_dim)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=seq_collate)

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        loader = self.build_dataloader()
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                logits10 = self.model(x)
                probs9 = F.softmax(logits10[..., 1:], dim=-1)
                yield {"uid": batch["uid"], "lens": batch["T"], "probs_dec_9": probs9.detach().cpu().numpy(), "ai_rate": 1}


def make_adapter(spec: Dict[str, Any], args: argparse.Namespace) -> BaseAdapter:
    family = spec["family"].lower()
    if family == "productgpt":
        return ProductGPTAdapter(spec, args)
    if family == "gru":
        return GRUAdapter(spec, args)
    if family == "lstm":
        return LSTMAdapter(spec, args)
    raise ValueError(f"Unsupported family: {family}")


# =============================
# Evaluation engine
# =============================
def evaluate_model(
    adapter: BaseAdapter,
    label_dict: Dict[str, Dict[str, List[int]]],
    which_split,
    args: argparse.Namespace,
    output_dir: Path,
) -> EvalResult:
    model_dir = output_dir / adapter.name
    model_dir.mkdir(parents=True, exist_ok=True)

    scores = defaultdict(lambda: {"y": [], "p": []})
    multi_scores = defaultdict(lambda: {"y": [], "p": []})
    length_note = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}
    pred_path = model_dir / f"preds_{adapter.name}.jsonl.gz"
    pred_writer = smart_open_w(pred_path) if args.save_preds else None

    with (pred_writer or nullcontext(None)) as writer:
        for batch in adapter.predict_batches():
            uids = batch["uid"]
            lens = batch["lens"]
            probs_dec_9 = batch["probs_dec_9"]
            ai_rate = int(batch["ai_rate"])
            for i, uid in enumerate(uids):
                actual_len_i = int(lens[i])
                n_valid_slots_i = actual_len_i // ai_rate if ai_rate > 1 else actual_len_i
                probs_seq_np = probs_dec_9[i, :n_valid_slots_i]
                if writer is not None:
                    writer.write(json.dumps({"uid": uid, "probs": np.round(probs_seq_np, 6).tolist()}) + "\n")
                lbl_info = label_dict.get(uid)
                if lbl_info is None:
                    reject += 1
                    continue
                L_pred = len(probs_seq_np)
                L_lbl = len(lbl_info["label"])
                if L_pred != L_lbl:
                    length_note["pred>lbl" if L_pred > L_lbl else "pred<label"] += 1
                L = min(L_pred, L_lbl)
                lbl_offset = L_lbl - L
                y_arr = np.asarray(lbl_info["label"][lbl_offset : lbl_offset + L], dtype=np.int64)
                idx_h_arr = np.asarray(lbl_info["idx_h"][lbl_offset : lbl_offset + L], dtype=np.int64)
                feat_h_arr = np.asarray(lbl_info["feat_h"][lbl_offset : lbl_offset + L], dtype=np.int64)
                p_arr = probs_seq_np[:L]
                group_idx = np.where(feat_h_arr == 0, 0, np.where((feat_h_arr == 1) & (idx_h_arr == 0), 1, 2))
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
                    y_bin = np.isin(y_arr, list(TASK_POSSETS[task])).astype(np.int8)
                    col_idx = [j - 1 for j in pos_classes]
                    p_bin = p_arr[:, col_idx].sum(axis=1)
                    for g_idx, g_name in enumerate(GROUP_ORDER):
                        mask = group_idx == g_idx
                        if not mask.any():
                            continue
                        key = (task, g_name, split_tag)
                        scores[key]["y"].extend(y_bin[mask].tolist())
                        scores[key]["p"].extend(p_bin[mask].tolist())
                accept += 1

    rows: List[Dict[str, Any]] = []
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
                    auc = roc_auc_score(y, p)
                    y_hat = [int(prob >= args.thresh) for prob in p]
                    hit = accuracy_score(y, y_hat)
                    f1 = f1_score(y, y_hat)
                    auprc = average_precision_score(y, p)
                rows.append({
                    "Task": task,
                    "TaskPretty": PAPER_AUC_TASKS.get(task, task),
                    "Group": grp,
                    "Split": spl,
                    "AUC": round(float(auc), 4) if not np.isnan(auc) else np.nan,
                    "Hit": round(float(hit), 4) if not np.isnan(hit) else np.nan,
                    "F1": round(float(f1), 4) if not np.isnan(f1) else np.nan,
                    "AUPRC": round(float(auprc), 4) if not np.isnan(auprc) else np.nan,
                })
    binary_metrics = pd.DataFrame(rows)

    multi_rows: List[Dict[str, Any]] = []
    for grp in GROUP_ORDER:
        for spl in SPLIT_ORDER:
            y = multi_scores[(grp, spl)]["y"]
            p = multi_scores[(grp, spl)]["p"]
            if not y:
                continue
            y_arr = np.asarray(y, dtype=np.int64)
            p_arr = np.vstack(p)
            multi_rows.append({"Group": grp, "Split": spl, **compute_multiclass_metrics(y_arr, p_arr)})
    multiclass_metrics = pd.DataFrame(multi_rows)

    perclass_rows: List[Dict[str, Any]] = []
    for grp in GROUP_ORDER:
        for spl in SPLIT_ORDER:
            y = multi_scores[(grp, spl)]["y"]
            p = multi_scores[(grp, spl)]["p"]
            if not y:
                continue
            y_arr = np.asarray(y, dtype=np.int64)
            p_arr = np.vstack(p)
            pc = compute_perclass_metrics(y_arr, p_arr)
            for c, cm in pc.items():
                perclass_rows.append({"Group": grp, "Split": spl, "Class": c, **cm})
    perclass_metrics = pd.DataFrame(perclass_rows)

    selected_binary_auc = (
        binary_metrics[binary_metrics["Task"].isin(PAPER_AUC_TASKS)]
        .pivot_table(index=["Group", "TaskPretty"], columns="Split", values="AUC")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, PAPER_AUC_ORDER], names=["Group", "TaskPretty"]))
        .reindex(columns=SPLIT_ORDER)
        .round(4)
    )

    # Save model-level outputs
    binary_metrics.to_csv(model_dir / "binary_metrics_long.csv", index=False)
    multiclass_metrics.to_csv(model_dir / "multiclass_metrics_long.csv", index=False)
    perclass_metrics.to_csv(model_dir / "perclass_metrics_long.csv", index=False)
    selected_binary_auc.to_csv(model_dir / "selected_binary_auc.csv")

    info = {
        "accepted_users": accept,
        "missing_labels": reject,
        "length_note": dict(length_note),
        "coverage": {k: len(v) for k, v in accept_users.items()},
        "family": adapter.family,
        "spec": adapter.spec,
    }
    with open(model_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return EvalResult(
        model_name=adapter.name,
        binary_metrics=binary_metrics,
        multiclass_metrics=multiclass_metrics,
        perclass_metrics=perclass_metrics,
        selected_binary_auc=selected_binary_auc,
        info=info,
    )


# =============================
# Cross-model comparison reports
# =============================
def build_comparison_tables(results: List[EvalResult], compare_on: str, out_dir: Path) -> Dict[str, pd.DataFrame]:
    # 1) Multiclass summary by group and metric
    multi_rows = []
    for r in results:
        df = r.multiclass_metrics.copy()
        df = df[df["Split"] == compare_on]
        for _, row in df.iterrows():
            for metric in MULTICLASS_METRICS:
                multi_rows.append({
                    "Model": r.model_name,
                    "Group": row["Group"],
                    "Metric": metric,
                    "Value": row[metric],
                })
    multi_long = pd.DataFrame(multi_rows)
    multiclass_comparison = (
        multi_long.pivot_table(index=["Group", "Metric"], columns="Model", values="Value")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, MULTICLASS_METRICS], names=["Group", "Metric"]))
        .round(4)
    )

    # 2) Selected binary AUC comparison by task/group
    bin_rows = []
    for r in results:
        df = r.binary_metrics.copy()
        df = df[(df["Split"] == compare_on) & (df["Task"].isin(PAPER_AUC_TASKS))]
        for _, row in df.iterrows():
            bin_rows.append({
                "Model": r.model_name,
                "Group": row["Group"],
                "TaskPretty": row["TaskPretty"],
                "AUC": row["AUC"],
            })
    bin_long = pd.DataFrame(bin_rows)
    binary_auc_comparison = (
        bin_long.pivot_table(index=["Group", "TaskPretty"], columns="Model", values="AUC")
        .reindex(index=pd.MultiIndex.from_product([GROUP_ORDER, PAPER_AUC_ORDER], names=["Group", "TaskPretty"]))
        .round(4)
    )

    # 3) Overall summary by model
    summary_rows = []
    for r in results:
        bm = r.binary_metrics
        mm = r.multiclass_metrics
        sel = bm[(bm["Split"] == compare_on) & (bm["Task"].isin(PAPER_AUC_TASKS))]
        for group in GROUP_ORDER:
            gsel = sel[sel["Group"] == group]
            summary_rows.append({
                "Model": r.model_name,
                "Group": group,
                "AvgSelectedBinaryAUC": round(float(gsel["AUC"].mean()), 4) if not gsel.empty else np.nan,
            })
        overall_auc = sel["AUC"].mean() if not sel.empty else np.nan
        mtest = mm[mm["Split"] == compare_on]
        summary_rows.append({
            "Model": r.model_name,
            "Group": "Overall",
            "AvgSelectedBinaryAUC": round(float(overall_auc), 4) if not np.isnan(overall_auc) else np.nan,
        })
        for _, row in mtest.iterrows():
            summary_rows.append({
                "Model": r.model_name,
                "Group": row["Group"],
                "MacroF1": row.get("MacroF1", np.nan),
                "Top1Acc": row.get("Top1Acc", np.nan),
                "MCC": row.get("MCC", np.nan),
                "LogLoss": row.get("LogLoss", np.nan),
            })
    summary_long = pd.DataFrame(summary_rows)
    binary_group_summary = (
        summary_long[summary_long["Group"].isin(GROUP_ORDER + ["Overall"]) & summary_long["AvgSelectedBinaryAUC"].notna()]
        .pivot_table(index="Group", columns="Model", values="AvgSelectedBinaryAUC")
        .reindex(index=GROUP_ORDER + ["Overall"])
        .round(4)
    )
    multiclass_macrof1_summary = (
        summary_long[summary_long["Group"].isin(GROUP_ORDER) & summary_long["MacroF1"].notna()]
        .pivot_table(index="Group", columns="Model", values="MacroF1")
        .reindex(index=GROUP_ORDER)
        .round(4)
    )
    top1acc_summary = (
        summary_long[summary_long["Group"].isin(GROUP_ORDER) & summary_long["Top1Acc"].notna()]
        .pivot_table(index="Group", columns="Model", values="Top1Acc")
        .reindex(index=GROUP_ORDER)
        .round(4)
    )

    tables = {
        "multiclass_comparison": multiclass_comparison,
        "binary_auc_comparison": binary_auc_comparison,
        "binary_group_summary": binary_group_summary,
        "multiclass_macrof1_summary": multiclass_macrof1_summary,
        "top1acc_summary": top1acc_summary,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}_{compare_on}.csv")
    return tables


def save_markdown_summary(tables: Dict[str, pd.DataFrame], out_dir: Path, compare_on: str):
    md_path = out_dir / f"comparison_summary_{compare_on}.md"
    with open(md_path, "w") as f:
        f.write(f"# Unified comparison summary ({compare_on})\n\n")
        for title, df in tables.items():
            f.write(f"## {title}\n\n")
            f.write(df.to_markdown())
            f.write("\n\n")


def maybe_upload_outputs(out_dir: Path, s3_prefix: str, fold_id: int):
    if not s3_prefix:
        return
    effective_prefix = s3_join_folder(s3_prefix, f"fold{fold_id}") if fold_id >= 0 else s3_prefix
    for path in out_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(out_dir)
            dest = s3_join(effective_prefix, str(rel).replace(os.sep, "/"))
            s3_upload_file(path, dest)
            print(f"[S3] uploaded: {dest}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r") as f:
        config = json.load(f)
    models = config["models"] if isinstance(config, dict) else config
    label_dict, records = load_labels(Path(args.labels))
    which_split, split_info, _, _ = make_splitter(records, args.uids_val, args.uids_test, args.seed)
    print(f"[INFO] split mode: {split_info}")

    results: List[EvalResult] = []
    for spec in models:
        print(f"\n[INFO] Evaluating {spec['name']} ({spec['family']})")
        adapter = make_adapter(spec, args)
        res = evaluate_model(adapter, label_dict, which_split, args, output_dir)
        results.append(res)
        print(f"[INFO] Completed {spec['name']} | accepted={res.info['accepted_users']} missing_labels={res.info['missing_labels']}")

    tables = build_comparison_tables(results, args.compare_on, output_dir)
    save_markdown_summary(tables, output_dir, args.compare_on)
    maybe_upload_outputs(output_dir, args.s3, args.fold_id)

    # concise console output
    print("\n=============  MULTICLASS MACRO-F1 SUMMARY  =======================")
    print(tables["multiclass_macrof1_summary"].fillna(" NA"))
    print("============================================================")
    print("\n=============  TOP1 ACCURACY SUMMARY  =======================")
    print(tables["top1acc_summary"].fillna(" NA"))
    print("============================================================")
    print("\n=============  AVG SELECTED BINARY AUC SUMMARY  =======================")
    print(tables["binary_group_summary"].fillna(" NA"))
    print("============================================================")


if __name__ == "__main__":
    main()
