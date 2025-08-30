# train_fold_lstm.py
from __future__ import annotations
import json, boto3, subprocess, pathlib, os, tempfile
from typing import Dict, Any

TRAIN_SCRIPT = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py"
BUCKET       = "productgptbucket"
PREFIX       = "CV_GRU"
DATA_TRAIN   = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"

INPUT_DIM    = "15"
BATCH_SIZE   = "4"
HIDDEN_SIZE  = "128"
LR_STR       = "0.001"   # <-- your best GRU

MODELS_PREFIX = f"{PREFIX}/models"  # s3://productgptbucket/CV_GRU/models/...

def _model_key(fold_id: int) -> str:
    return f"{MODELS_PREFIX}/gru_h{HIDDEN_SIZE}_lr{LR_STR}_bs{BATCH_SIZE}_fold{fold_id}.pt"

_s3 = boto3.client("s3")

def _metrics_key(fold_id: int) -> str:
    return f"{PREFIX}/metrics/gru_h{HIDDEN_SIZE}_lr{LR_STR}_bs{BATCH_SIZE}_fold{fold_id}.json"

def _s3_exists(bucket: str, key: str) -> bool:
    try:
        _s3.head_object(Bucket=bucket, Key=key); return True
    except Exception:
        return False

def _load_spec(spec_uri: str) -> dict:
    if spec_uri.startswith("s3://"):
        b, k = spec_uri[5:].split("/", 1)
        body = _s3.get_object(Bucket=b, Key=k)["Body"].read()
        return json.loads(body)
    with open(spec_uri, "r") as f:
        return json.load(f)

def _fetch_metrics_from_s3(fold_id: int) -> dict:
    obj = _s3.get_object(Bucket=BUCKET, Key=_metrics_key(fold_id))
    m = json.loads(obj["Body"].read()); m["fold"] = fold_id
    return m

def run_single_fold(fold_id: int,
                    spec_uri: str = "s3://productgptbucket/CV/folds.json") -> Dict[str, Any]:

    # Resume if metrics already exist
    if _s3_exists(BUCKET, _metrics_key(fold_id)):
        print(f"[resume] s3://{BUCKET}/{_metrics_key(fold_id)}")
        return _fetch_metrics_from_s3(fold_id)

    # 1) Build splits
    spec = _load_spec(spec_uri)
    test_uids   = [u for u, f in spec["assignment"].items() if f == fold_id]
    train_uids  = [u for u in spec["assignment"] if u not in test_uids]

    # 2) Write UID lists to temp files instead of CLI args
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f_tr, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f_te, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".pt")   as f_ck, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")  as f_out:
        json.dump(train_uids, f_tr); tr_path = f_tr.name
        json.dump(test_uids,  f_te); te_path = f_te.name
        ckpt_tmp, out_tmp = f_ck.name, f_out.name

    # 3) Launch trainer (subprocess = clean state; works with/without deepspeed)
    cmd = [
        "python3", TRAIN_SCRIPT,
        "--model",       "gru",
        "--fold",        str(fold_id),
        "--bucket",      BUCKET,
        "--prefix",      PREFIX,
        "--data",        DATA_TRAIN,
        "--ckpt",        ckpt_tmp,
        "--hidden_size", HIDDEN_SIZE,
        "--input_dim",   INPUT_DIM,
        "--batch_size",  BATCH_SIZE,
        "--lr",          LR_STR,
        "--uids_trainval_file", tr_path,   # <-- pass paths
        "--uids_test_file",     te_path,   
    ]
    subprocess.check_call(cmd)

    # 4) Read metrics produced by trainer (uploaded to S3)
    m = _fetch_metrics_from_s3(fold_id)

    # 5) Upload checkpoint
    ckpt_key = _model_key(fold_id)
    try:
        if not _s3_exists(BUCKET, ckpt_key):
            _s3.upload_file(ckpt_tmp, BUCKET, ckpt_key)
        m["ckpt_s3"] = f"s3://{BUCKET}/{ckpt_key}"
        try: os.remove(ckpt_tmp)
        except Exception: pass
    except Exception as e:
        print(f"[warn] failed to upload checkpoint: {e}")

    return m
