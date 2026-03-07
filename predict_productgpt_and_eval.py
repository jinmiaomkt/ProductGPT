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
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /path/to/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr..._w4_fold0.pt \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/YourEvalRuns/PhaseB/ \
  --pred-out /tmp/preds_phaseB.jsonl.gz \
  --uids-val /tmp/uids_val.txt \
  --uids-test /tmp/uids_test.txt \
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

from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
)

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

    # EXACT-MATCH UID overrides (local path or s3://bucket/key, one UID per line)
    p.add_argument("--uids-val",  default="", help="Text file (or s3://...) with validation UIDs, one per line")
    p.add_argument("--uids-test", default="", help="Text file (or s3://...) with test UIDs, one per line")

    # --- Phase-B exact split reproduction (when no --uids-val/test are provided) ---
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

def deterministic_subsample_indices(n: int, frac: float, seed: int) -> set[int]:
    """Exactly matches your training _deterministic_subsample logic."""
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
    """
    Reproduce training split exactly:
      - get uids_trainval from fold-spec (exclude fold holdout)
      - raw = load_json_dataset(split_from_path, keep_uids=uids_trainval)
      - optional deterministic subsample (if data_frac < 1)
      - random_split with torch.Generator().manual_seed(split_seed)
    """
    spec = load_json_from_local_or_s3(fold_spec_uri)

    uids_test_fold = [u for u, f in spec["assignment"].items() if f == fold_id]
    uids_trainval  = [u for u in spec["assignment"] if u not in set(uids_test_fold)]
    if not uids_trainval:
        raise ValueError(f"No uids_trainval found for fold_id={fold_id}")

    raw = load_json_dataset(split_from_path, keep_uids=set(uids_trainval))

    # Match training deterministic subsample behavior
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

def collate_fn(pad_id: int):
    def _inner(batch):
        uids = [b["uid"] for b in batch]
        lens = [len(b["x"]) for b in batch]
        Lmax = max(lens)
        X    = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
        for i,(item,L) in enumerate(zip(batch,lens)):
            X[i,:L] = item["x"]
        return {"uid": uids, "x": X}
    return _inner

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
    hp = parse_hp_from_ckpt_name(ckpt_path)  # <-- MUST be defined before split logic
    label_path  = Path(args.labels)
    feat_path   = Path(args.feat_xlsx)
    s3_prefix   = args.s3 if args.s3.endswith("/") else (args.s3 + "/")
    pred_out    = args.pred_out

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
    # else:
    #     which_split = build_splits(records, seed=args.seed)
    #     print(f"[INFO] Using fallback 80/10/10 split with seed={args.seed}")
    else:
        # ===== reproduce training split EXACTLY (Phase-B compatible) =====
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

        print(f"[INFO] Using EXACT training split via fold-spec:"
              f" fold={fold_for_split}, val={len(uids_val_override)}, test={len(uids_test_override)}, "
              f"seed={args.split_seed}, data_frac={args.split_data_frac}")

        # optional: dump the uid lists locally for reuse
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

    # Ensure seq_len matches training
    cfg["seq_len_ai"] = cfg["seq_len_tgt"] * cfg["ai_rate"]

    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_ai"],
        d_model=hp["d_model"],
        n_layers=hp["N"],
        n_heads=hp["num_heads"],
        d_ff=hp["d_ff"],
        dropout=0.0,                      # inference dropout off
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

    # ---------- Metric accumulators ----------
    scores       = defaultdict(lambda: {"y": [], "p": []})
    length_note  = Counter()
    accept = reject = 0
    accept_users = {"val": set(), "test": set(), "train": set()}

    # Optional predictions writer
    pred_writer = None
    if pred_out:
        pred_writer = smart_open_w(pred_out)

    # ---------- Inference + streaming eval ----------
    focus_ids = torch.arange(1, 10, device=device)  # decision classes 1..9
    with torch.no_grad():
        for batch in loader:
            x    = batch["x"].to(device)
            uids = batch["uid"]
            logits_full = model(x)  # (B, T, V) or (B, Nslots, V)

            # pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)

            if x.size(1) < cfg["ai_rate"]:
                # No decision slot exists in this batch (sequence too short)
                pos = torch.empty((0,), dtype=torch.long, device=device)
            else:
                pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)

            # Align to decision-slot axis
            if logits_full.size(1) == x.size(1):
                logits = logits_full[:, pos, :]     # (B, Nslots, V)
            else:
                logits = logits_full                # (B, Nslots, V)

            V_out = logits.size(-1)

            # Analytic correction for class-9 upweighting (same idea as train_model)
            # - If full vocab: class 9 is token id 9
            # - If decision-only: class 9 is index 8 (0-based)
            logits = logits.clone()
            if weight_class9 is not None and weight_class9 != 1:
                import math
                if V_out == 9:
                    logits[..., 8] -= math.log(weight_class9)
                elif V_out >= 10:
                    logits[..., 9] -= math.log(weight_class9)

            probs = torch.softmax(logits, dim=-1)

            # Extract 9-way decision probs for classes 1..9
            if V_out == 9:
                prob_dec_9 = probs                      # already (B, Nslots, 9)
            else:
                prob_dec_9 = probs[..., 1:10]           # token IDs 1..9 -> 9 columns

            prob_dec_focus = prob_dec_9                 # (B, N, 9)


            for i, uid in enumerate(uids):
                probs_seq = prob_dec_focus[i].detach().cpu().numpy()  # (N, 9)
                probs_seq = np.round(probs_seq, 6).tolist()           # list[list[float]]
                # write predictions line if requested
                if pred_writer:
                    pred_writer.write(json.dumps({"uid": uid, "probs": probs_seq}) + "\n")

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
                    y      = lbl_info["label"][t]
                    idx_h  = lbl_info["idx_h"][t]
                    feat_h = lbl_info["feat_h"][t]
                    probs  = probs_seq[t]  # length 9, corresponds to classes 1..9

                    group = period_group(idx_h, feat_h)
                    for task, pos_classes in BIN_TASKS.items():
                        y_bin = int(y in TASK_POSSETS[task])
                        # decisions are 1-indexed → probs[j-1]
                        p_bin = sum(probs[j-1] for j in pos_classes)
                        key   = (task, group, split_tag)
                        scores[key]["y"].append(y_bin)
                        scores[key]["p"].append(p_bin)

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
                    "N": n, "Pos": pos, "Neg": neg, "Prev": round(prev, 4)
                })

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
