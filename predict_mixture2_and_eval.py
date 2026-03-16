#!/usr/bin/env python3
"""
predict_mixture2_and_eval.py

End-to-end:
  - Run inference for 9-way decision probabilities (every ai_rate steps)
  - Compute AUC / Hit / F1 / AUPRC by (Task, PeriodGroup, Split)
  - Print AUC table to console
  - Save CSV tables (and optional predictions) locally and upload to S3

Usage (example):

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

import numpy as np
import pandas as pd
import torch
import re
import math

from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
)

# ══════════ VectorScaling (must match training) ══════════
class VectorScaling(torch.nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(n_classes))
        self.b = torch.nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits_dec):
        return F.softmax(self.a * logits_dec + self.b, dim=-1)


def load_calibrator(ckpt_path, device):
    uid_part = ckpt_path.stem.replace("FullProductGPT_", "")
    cal_path = ckpt_path.parent / f"calibrator_{uid_part}.pt"
    if not cal_path.exists():
        print(f"[INFO] No calibrator at {cal_path}")
        return None
    cal = VectorScaling(9).to(device)
    state = torch.load(cal_path, map_location=device)
    cal.a.data = state["a"].to(device)
    cal.b.data = state["b"].to(device)
    print(f"[INFO] Loaded calibrator: a={cal.a.data.cpu().numpy().round(4)}, b={cal.b.data.cpu().numpy().round(4)}")
    return cal

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
from model4_mixture2_decoderonly_feature_performer import build_transformer
from train1_decision_only_performer_aws import _ensure_jsonl, JsonLineDataset, _build_tok
from dataset4_productgpt import load_json_dataset

# Optional: silence Intel/LLVM OpenMP clash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# FullProductGPT_featurebased_performerfeatures16_dmodel128_ff128_N6_heads4_lr0.0001_w2_fold0.pt .
# FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0.pt
# LP_ProductGPT_featurebased_performerfeatures32_dmodel128_ff128_N8_heads4_lr0.001_w2.pt 
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

    p.add_argument("--fold-spec", default="s3://productgptbucket/folds/productgptfolds.json",
                help="Same SPEC_URI used by Phase-B training")
    p.add_argument("--train-file", default="",
                help="Training json used to reproduce Phase-B split (default: cfg['filepath'])")
    
    p.add_argument("--calibration", choices=["calibrator", "analytic", "none"], default="none")

    # EXACT-MATCH UID overrides (local path or s3://bucket/key, one UID per line)
    p.add_argument("--uids-val",  default="", help="Text file (or s3://...) with validation UIDs, one per line")
    p.add_argument("--uids-test", default="", help="Text file (or s3://...) with test UIDs, one per line")

    # Optional fold index to route outputs under .../fold{ID}/
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
    return bucket, key  # key may be '' (prefix), but NEVER add a '/'

def s3_join(prefix: str, filename: str) -> str:
    # prefix = 's3://bucket/path' or 's3://bucket/path/'
    if not prefix.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {prefix}")
    if not filename:
        raise ValueError("filename is empty")
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + filename  # NO trailing slash after filename

def s3_join_folder(prefix: str, folder: str) -> str:
    """Join a folder to an S3 prefix, ensuring exactly one slash."""
    if not prefix.endswith("/"):
        prefix += "/"
    folder = folder.strip("/")
    return prefix + folder + "/"

def s3_upload_file(local_path: str | Path, s3_uri_full: str):
    # Assert we didn't accidentally create a "folder" key
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
    """Read small text file from S3 to memory (try boto3, fallback to AWS CLI)."""
    bucket, key = parse_s3_uri(s3_uri)
    try:
        import boto3
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        data = subprocess.check_output(["aws", "s3", "cp", s3_uri, "-"])
        return data.decode("utf-8")

def load_uid_set(path_or_s3: str) -> Set[str]:
    """Load a UID set from a local path or s3://bucket/key (one UID per line)."""
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

def phaseb_val_test_uids(cfg: dict, fold_id: int, fold_spec_uri: str) -> tuple[Set[str], Set[str]]:
    """
    Reproduce Phase-B split EXACTLY:
      - exclude fold holdout uids (uids_test)
      - load raw with keep_uids=uids_trainval
      - random_split(raw, [0.8,0.1,0.1], generator seed=33)
    """
    spec = load_json_from_local_or_s3(fold_spec_uri)

    # same as ray_tune4_feature_git.py
    uids_test_fold = [u for u, f in spec["assignment"].items() if f == fold_id]
    uids_trainval  = [u for u in spec["assignment"] if u not in set(uids_test_fold)]
    if not uids_trainval:
        raise ValueError(f"No uids_trainval found for fold_id={fold_id}")

    # IMPORTANT: load the same raw order as training
    raw = load_json_dataset(cfg["filepath"], keep_uids=set(uids_trainval))

    # Phase-B uses data_frac=1.0 (no subsample). If you ever want to match Phase-A,
    # you must also reproduce _deterministic_subsample(raw, data_frac, subsample_seed=33).

    n = len(raw)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    if n_val == 0 or n_test == 0:
        raise ValueError(f"Not enough data to split: n={n} -> train/val/test={n_train}/{n_val}/{n_test}")

    g = torch.Generator().manual_seed(33)  # EXACT match to training build_dataloaders
    tr_i, va_i, te_i = random_split(range(n), [n_train, n_val, n_test], generator=g)

    val_uids  = {flat_uid(raw[i]["uid"]) for i in va_i.indices}
    test_uids = {flat_uid(raw[i]["uid"]) for i in te_i.indices}

    overlap = val_uids & test_uids
    if overlap:
        raise RuntimeError(f"val/test overlap (should never happen): {list(overlap)[:5]}")

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

def flat_uid(u):  # u can be scalar or [scalar]
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
class PredictDatasetWithUserID(JsonLineDataset):
    """Extends PredictDataset to also return user_id integer."""
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

        # Normalize uid for lookup (same as MixtureTransformerDataset._normalize_uid)
        user_id = self.uid_to_index.get(uid, 0)  # 0 = UNK

        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.long),
                "user_id": torch.tensor(user_id, dtype=torch.long)}

def collate_fn_with_uid(pad_id):
    def _inner(batch):
        uids = [b["uid"] for b in batch]
        user_ids = torch.stack([b["user_id"] for b in batch])
        lens = [len(b["x"]) for b in batch]
        Lmax = max(lens)
        X = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        for i, (item, L) in enumerate(zip(batch, lens)):
            X[i, :L] = item["x"]
        return {"uid": uids, "x": X, "user_id": user_ids}
    return _inner

def build_uid_to_index(cfg, fold_spec_uri, fold_id):
    """
    Build the same uid_to_index mapping that training used.
    This ensures user_id integers match the trained embeddings.
    """
    spec = load_json_from_local_or_s3(fold_spec_uri)
    uids_test_fold = [u for u, f in spec["assignment"].items() if f == fold_id]
    uids_trainval = [u for u in spec["assignment"] if u not in set(uids_test_fold)]

    raw = load_json_dataset(cfg["filepath"], keep_uids=set(uids_trainval))

    def normalize_uid(uid):
        if uid is None: return ""
        if isinstance(uid, (list, tuple, set)):
            vals = [str(x) for x in uid if x is not None]
            return "|".join(sorted(vals)) if vals else ""
        return str(uid)

    unique_uids = sorted({normalize_uid(rec.get("uid", "")) for rec in raw})
    uid_to_index = {u: i + 1 for i, u in enumerate(unique_uids)}
    num_users = len(uid_to_index) + 1

    print(f"[INFO] Built uid_to_index: {num_users} users (including UNK=0)")
    return uid_to_index, num_users

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

# ======================= Main ====================
def main():
    args = parse_args()
    data_path   = _ensure_jsonl(args.data)
    ckpt_path   = Path(args.ckpt)
    label_path  = Path(args.labels)
    feat_path   = Path(args.feat_xlsx)
    s3_prefix   = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    pred_out    = args.pred_out

    hp = parse_hp_from_ckpt_name(ckpt_path)

    # If fold-id provided, nest under that folder
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
    # special ids (align PAD with tokenizer)
    SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
    EOS_PROD_ID, SOS_PROD_ID          = 57, 58
    SPECIAL_IDS = [pad_id, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

    # ---------- Labels ----------
    label_dict, records = load_labels(label_path)

    # ---------- EXACT MATCH OVERRIDE or fallback 80/10/10 ----------
    uids_val_override  = load_uid_set(args.uids_val)  if args.uids_val  else set()
    uids_test_override = load_uid_set(args.uids_test) if args.uids_test else set()

    if uids_val_override or uids_test_override:
        # Must provide BOTH
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

    # else:
    #     # ===== EXACT Phase-B split (matches training) =====
    #     # fold id used by training is encoded in checkpoint name
    #     fold_for_split = hp["fold_id"]

    #     # ensure cfg["filepath"] matches Phase-B training file
    #     if args.train_file:
    #         cfg["filepath"] = args.train_file  # override if provided

    #     uids_val, uids_test = phaseb_val_test_uids(cfg, fold_for_split, args.fold_spec)

    #     tmp_val = Path("/tmp") / f"phaseb_fold{fold_for_split}_val_uids.txt"
    #     tmp_te  = Path("/tmp") / f"phaseb_fold{fold_for_split}_test_uids.txt"
    #     tmp_val.write_text("\n".join(sorted(uids_val)))
    #     tmp_te.write_text("\n".join(sorted(uids_test)))
    #     print(f"[INFO] Wrote UID lists: {tmp_val} and {tmp_te}")

    #     def which_split(u):
    #         return "val" if u in uids_val else "test" if u in uids_test else "train"

    #     print(f"[INFO] Using Phase-B EXACT split: fold={fold_for_split}, "
    #         f"val={len(uids_val)}, test={len(uids_test)} (seed=33)")
    
    # ──────────────────────────────────────────────────────
    # FIX: Build user-id mapping (same as training used)
    # ──────────────────────────────────────────────────────
    uid_to_index, num_users = build_uid_to_index(cfg, args.fold_spec, hp["fold_id"])

    # uid_to_index, num_users = build_uid_to_index(cfg, args.fold_spec, hp["fold_id"])

    # Override num_users from checkpoint to match trained embedding size
    _tmp_state = torch.load(ckpt_path, map_location="cpu")
    _tmp_sd = _tmp_state.get("model_state_dict", _tmp_state.get("module", _tmp_state))
    for k, v in _tmp_sd.items():
        if "gate.logits.weight" in k:
            ckpt_num_users = v.shape[0]
            if ckpt_num_users != num_users:
                print(f"[WARN] num_users mismatch: build_uid_to_index={num_users}, checkpoint={ckpt_num_users}")
                print(f"[INFO] Truncating uid_to_index to match checkpoint ({ckpt_num_users} users)")
                # Keep only users with index < ckpt_num_users
                uid_to_index = {u: idx for u, idx in uid_to_index.items() if idx < ckpt_num_users}
                num_users = ckpt_num_users
            break
    del _tmp_state

    # ──────────────────────────────────────────────────────
    # FIX: Dataset now returns user_id
    # ──────────────────────────────────────────────────────
    ds = PredictDatasetWithUserID(data_path, pad_id=pad_id, uid_to_index=uid_to_index)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        collate_fn=collate_fn_with_uid(pad_id))

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure seq_len matches training
    cfg["seq_len_ai"] = cfg["seq_len_tgt"] * cfg["ai_rate"]

    # ──────────────────────────────────────────────────────
    # FIX: pass num_users to build_transformer (was missing)
    # ──────────────────────────────────────────────────────
    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"], vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_ai"], d_model=hp["d_model"], n_layers=hp["N"],
        n_heads=hp["num_heads"], d_ff=hp["d_ff"], dropout=0.0,
        nb_features=hp["nb_features"], kernel_type=cfg["kernel_type"],
        feature_tensor=load_feature_tensor(feat_path), special_token_ids=SPECIAL_IDS,
        num_users=num_users,            # FIX: was missing entirely
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
    
    sd = clean_state_dict(raw_sd)
    sd.pop("projection.output_head.mean_alpha_buffer", None)

    model.load_state_dict(sd, strict=False)

    # # Optional predictions writer
    # pred_writer = None
    # if pred_out:
    #     pred_writer = smart_open_w(pred_out)

    # Set mean projection gate from training users
    train_user_ids = sorted(uid_to_index.values())
    model.set_projection_mean_gate_from_train_users(train_user_ids)
    if hasattr(model.gate, "update_mean_gate"):
        model.gate.update_mean_gate()
        model.gate.use_mean_gate = True

    # Load calibration
    calibrator = None
    logit_bias_9 = None

    if args.calibration == "calibrator":
        calibrator = load_calibrator(ckpt_path, device)
        if calibrator is None:
            print("[WARN] No calibrator found, falling back to analytic")
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

    # ---------- Inference + streaming eval ----------

    # ---------- Metric accumulators ----------
    scores       = defaultdict(lambda: {"y": [], "p": []})
    length_note  = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}
    pred_writer = smart_open_w(args.pred_out) if args.pred_out else None
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            user_ids = batch["user_id"].to(device)
            uids = batch["uid"]
            pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)
            if pos.numel() == 0:
                continue

            # ──────────────────────────────────────────────────
            # FIX: Get hidden states + per-head logits + alpha
            # ──────────────────────────────────────────────────
            _, hidden = model(x, user_ids, projection_gate_mode="mean", return_hidden=True)

            proj_result = model.projection(
                hidden, user_idx=user_ids, gate_mode="mean",
                return_alpha=True, return_head_logits=True,
            )
            _, alpha, head_logits = proj_result  # alpha:(B,H), head_logits:(B,T,H,V)

            if head_logits.size(1) == x.size(1):
                head_logits = head_logits[:, pos, :, :]

            # ──────────────────────────────────────────────────
            # FIX: Per-head correction in 9-class subspace
            # ──────────────────────────────────────────────────
            head_logits_dec = head_logits[..., 1:10]  # (B, Nslots, H, 9)
            alpha_bt = alpha[:, None, :, None]         # (B, 1, H, 1)

            if calibrator is not None:
                mixed = (alpha_bt * head_logits_dec).sum(dim=2)
                prob_dec_9 = calibrator(mixed)
            elif logit_bias_9 is not None:
                bias = logit_bias_9.to(device=device, dtype=head_logits_dec.dtype)
                corrected = F.softmax(head_logits_dec - bias, dim=-1)
                prob_dec_9 = (alpha_bt * corrected).sum(dim=2)
            else:
                hp_probs = F.softmax(head_logits_dec, dim=-1)
                prob_dec_9 = (alpha_bt * hp_probs).sum(dim=2)

            # prob_dec_9: (B, Nslots, 9), sums to 1

            for i, uid in enumerate(uids):
                probs_seq = prob_dec_9[i].cpu().numpy()
                probs_seq = np.round(probs_seq, 6).tolist()
                if pred_writer:
                    pred_writer.write(json.dumps({"uid": uid, "probs": probs_seq}) + "\n")

                lbl_info = label_dict.get(uid)
                if lbl_info is None:
                    reject += 1
                    continue

                L = min(len(probs_seq), len(lbl_info["label"]))
                split_tag = which_split(uid)
                for t in range(L):
                    y = lbl_info["label"][t]
                    group = period_group(lbl_info["idx_h"][t], lbl_info["feat_h"][t])
                    probs = probs_seq[t]
                    for task, pos_classes in BIN_TASKS.items():
                        y_bin = int(y in TASK_POSSETS[task])
                        p_bin = sum(probs[j-1] for j in pos_classes)
                        scores[(task, group, split_tag)]["y"].append(y_bin)
                        scores[(task, group, split_tag)]["p"].append(p_bin)
                accept += 1

    if pred_writer:
        pred_writer.__exit__(None, None, None)  # close the context

    print(f"[INFO] parsed: {accept} users accepted, {reject} users missing labels.")

    if length_note:
        print("[INFO] length mismatches:", dict(length_note))
    # Report exact-match coverage if overrides were provided
    if args.uids_val and args.uids_test:
        print(f"[INFO] coverage: val={len(accept_users.get('val', set()))} / {len(load_uid_set(args.uids_val))}, "
              f"test={len(accept_users.get('test', set()))} / {len(load_uid_set(args.uids_test))}")

    # After computing `scores` and before rows = []
    bucket_stats = []  # list of dicts for counts and prevalence

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
                    # "AUC": roc_auc_score(y, p),
                    # "Hit": accuracy_score(y, [int(v >= args.thresh) for v in p]),
                    # "F1": f1_score(y, [int(v >= args.thresh) for v in p]),
                    # "AUPRC": average_precision_score(y, p),
                    "N": n, "Pos": pos, "Neg": neg, "Prev": round(prev, 4)})

    stats_df = pd.DataFrame(bucket_stats).sort_values(["Split","Group","Task"])
    print("\n=============  BUCKET SIZES & PREVALENCE  =======================")
    print(stats_df.to_string(index=False))
    print("============================================================")

    # ---------- Compute tables ----------
    rows = []
    for task in BIN_TASKS:
        for grp in ["Calibration","HoldoutA","HoldoutB"]:
            for spl in ["val","test"]:
                y, p = scores[(task, grp, spl)]["y"], scores[(task, grp, spl)]["p"]
                if not y:
                    continue
                # guard: need both classes present
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
    for metric in ["AUC", "Hit", "F1", "AUPRC"]:
        tbl = (metrics.pivot(index=["Task","Group"], columns="Split", values=metric)
               .reindex(columns=["val","test"]).round(4).sort_index())
        print(f"\n=============  {metric}  =======================")
        print(tbl.fillna(" NA"))

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
          .unstack("Split")   # columns become metric × split
          .round(4)
    )
    # reorder to outer split then metric, and val before test
    macro_period_tbl = macro_period_tbl.reorder_levels([1, 0], axis=1)
    macro_period_tbl = macro_period_tbl.sort_index(axis=1, level=0)
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
    out_dir = Path("/tmp/predict_eval_outputs")
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

    # Choose effective S3 prefix (with fold subfolder if provided)
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
