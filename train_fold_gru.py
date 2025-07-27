# train_fold_gru.py  ────────────────────────────────────────────────
"""
Train ONE GRU fold and return a metrics dict.
This mirrors ProductGPT’s train_fold.py.
"""

import json, boto3, subprocess, pathlib, os, tempfile
from typing import Dict

# ---------- config constants ----------
TRAIN_SCRIPT = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py"
BUCKET       = "productgptbucket"
PREFIX       = "CV_GRU"                      # s3://productgptbucket/CV_GRU/…
DATA_TRAIN   = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
INPUT_DIM    = "15"
BATCH_SIZE   = "4"
HIDDEN_SIZE  = "128"
LR_STR       = "0.0001"                      # must match TRAIN_SCRIPT
# --------------------------------------

def _s3(obj_key: str) -> Dict:
    return boto3.client("s3").get_object(Bucket=BUCKET, Key=obj_key)

def run_single_fold(fold_id: int,
                    spec_uri: str = "s3://productgptbucket/CV/folds.json") -> Dict:
    # -------- 1) pull fold spec ----------
    bucket, key = spec_uri.replace("s3://", "").split("/", 1)
    spec = json.loads(boto3.client("s3")
                      .get_object(Bucket=bucket, Key=key)["Body"].read())

    test_uids   = [u for u, f in spec["assignment"].items() if f == fold_id]
    train_uids  = [u for u in spec["assignment"] if u not in test_uids]

    # -------- 2) launch the GRU trainer ---
    ckpt_tmp = tempfile.mkstemp(suffix=".pt")[1]
    out_tmp  = tempfile.mkstemp(suffix=".txt")[1]

    cmd = [
        "python3", TRAIN_SCRIPT,
        "--model",       "gru",
        "--fold",        str(fold_id),
        "--bucket",      BUCKET,
        "--data",        DATA_TRAIN,
        "--ckpt",        ckpt_tmp,
        "--hidden_size", HIDDEN_SIZE,
        "--out",         out_tmp,
        "--input_dim",   INPUT_DIM,
        "--batch_size",  BATCH_SIZE,
        "--uids_trainval", json.dumps(train_uids),
        "--uids_test",     json.dumps(test_uids),
    ]
    subprocess.check_call(cmd)

    # -------- 3) download per‑fold metrics JSON -----------
    json_key  = (f"gru_h{HIDDEN_SIZE}_lr{LR_STR}_bs{BATCH_SIZE}"
                 f"_fold{fold_id}.json")
    local     = pathlib.Path(json_key)
    subprocess.check_call(
        ["aws", "s3", "cp",
         f"s3://{BUCKET}/{PREFIX}/metrics/{json_key}", str(local)])
    metrics = json.loads(local.read_text())
    metrics["fold"] = fold_id
    return metrics
