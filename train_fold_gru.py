# train_fold_gru.py
"""
Train ONE GRU fold and return its metrics dict.
- Locks best HP: h=128, lr=0.001, bs=4
- Skips training if per-fold metrics already exist on S3 (resume)
- Uses boto3 instead of shelling out to aws cli
"""
from __future__ import annotations
import json, boto3, subprocess, pathlib, os, tempfile
from typing import Dict, Any, Tuple

# ---------- config ----------
TRAIN_SCRIPT = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py"
BUCKET       = "productgptbucket"
PREFIX       = "CV_GRU"   # s3://productgptbucket/CV_GRU/…
DATA_TRAIN   = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
INPUT_DIM    = "15"
BATCH_SIZE   = "4"        # best model
HIDDEN_SIZE  = "128"      # best model
LR_STR       = "0.001"    # best model
# ---------------------------

_s3 = boto3.client("s3")

def _metrics_key(fold_id: int) -> str:
    # Keep naming consistent with your trainer’s S3 upload
    return f"{PREFIX}/metrics/gru_h{HIDDEN_SIZE}_lr{LR_STR}_bs{BATCH_SIZE}_fold{fold_id}.json"

def _s3_exists(bucket: str, key: str) -> bool:
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except _s3.exceptions.NoSuchKey:
        return False
    except Exception:
        return False

def _load_fold_spec(spec_uri: str) -> dict:
    if spec_uri.startswith("s3://"):
        b, k = spec_uri[5:].split("/", 1)
        body = _s3.get_object(Bucket=b, Key=k)["Body"].read()
        return json.loads(body)
    with open(spec_uri, "r") as f:
        return json.load(f)

def _fetch_metrics_from_s3(fold_id: int) -> dict:
    key = _metrics_key(fold_id)
    obj = _s3.get_object(Bucket=BUCKET, Key=key)
    m = json.loads(obj["Body"].read())
    m["fold"] = fold_id
    return m

def run_single_fold(fold_id: int,
                    spec_uri: str = "s3://productgptbucket/CV/folds.json") -> Dict[str, Any]:
    # Resume: if metrics json already exists on S3, skip training
    key = _metrics_key(fold_id)
    if _s3_exists(BUCKET, key):
        print(f"[resume] found s3://{BUCKET}/{key}")
        return _fetch_metrics_from_s3(fold_id)

    # 1) fold split
    spec = _load_fold_spec(spec_uri)
    test_uids   = [u for u, f in spec["assignment"].items() if f == fold_id]
    train_uids  = [u for u in spec["assignment"] if u not in test_uids]

    # 2) launch trainer (subprocess keeps DS/torch state isolated)
    ckpt_tmp = tempfile.mkstemp(suffix=".pt")[1]
    out_tmp  = tempfile.mkstemp(suffix=".txt")[1]

    cmd = [
        "python3", TRAIN_SCRIPT,
        "--model",       "gru",
        "--fold",        str(fold_id),
        "--bucket",      BUCKET,
        "--prefix",      PREFIX,          # ensure trainer uploads to CV_GRU/...
        "--data",        DATA_TRAIN,
        "--ckpt",        ckpt_tmp,
        "--hidden_size", HIDDEN_SIZE,
        "--input_dim",   INPUT_DIM,
        "--batch_size",  BATCH_SIZE,
        "--lr",          LR_STR,
        "--out",         out_tmp,
        "--uids_trainval", json.dumps(train_uids),
        "--uids_test",     json.dumps(test_uids),
    ]
    subprocess.check_call(cmd)

    # 3) read the per-fold metrics (uploaded by the trainer)
    return _fetch_metrics_from_s3(fold_id)
