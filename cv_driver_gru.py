"""
Run 10 GRU folds (sequentially or in parallel) on ONE–GPU machine.
Save per‑fold metrics → S3 and aggregate into csv/json.

Assumptions
-----------
* train_gru_lstm_new.py **does** accept the legacy --data --ckpt --hidden_size --out flags.
* Training script uploads   s3://productgptbucket/CV_GRU/metrics/gru_h128_lr0.0001_bs4_fold{k}.json
* One Ray head is already running on :6379   (see the sanity checklist below).
"""

import subprocess, json, pathlib, os, ray, boto3, pandas as pd

TRAIN  = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py"
BUCKET = "productgptbucket"
PREFIX = "CV_GRU"

# constants that control filenames
MODEL_NAME  = "gru"
HIDDEN      = "128"
LR_STR      = "0.0001"      # make sure it matches what the training script uses
BATCH_SIZE  = "4"
INPUT_DIM   = "15"

DATA_TRAIN  = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
CKPT_DUMMY  = "/tmp/dummy.pt"
OUT_DUMMY   = "/tmp/dummy.txt"

# ─────────────────────────  Ray  ──────────────────────────
ray.init(address="auto")              # must see exactly ONE head

@ray.remote(num_gpus=1)
def run_fold(k: int):
    """Train *one* fold, then download its metric JSON and return it as dict."""
    cmd = [
        "python3", TRAIN,
        "--model", MODEL_NAME,
        "--fold",  str(k),
        "--bucket", BUCKET,

        # legacy flags (values are ignored by your training script)
        "--data",         DATA_TRAIN,
        "--ckpt",         CKPT_DUMMY,
        "--hidden_size",  HIDDEN,
        "--out",          OUT_DUMMY,

        # added just to be explicit
        "--input_dim",    INPUT_DIM,
        "--batch_size",   BATCH_SIZE,
    ]
    subprocess.check_call(cmd)

    json_key = f"{MODEL_NAME}_h{HIDDEN}_lr{LR_STR}_bs{BATCH_SIZE}_fold{k}.json"
    local    = pathlib.Path(json_key)
    subprocess.check_call(
        ["aws", "s3", "cp",
         f"s3://{BUCKET}/{PREFIX}/metrics/{json_key}", str(local)]
    )
    return json.loads(local.read_text())

# ─── choose sequential or parallel ────────────────────────
SEQUENTIAL = True         # set False to let Ray queue tasks in parallel

metrics = []
if SEQUENTIAL:
    for k in range(10):
        metrics.append(ray.get(run_fold.remote(k)))   # blocks until done
        print(f"✓ fold{k} finished")
else:
    metrics = ray.get([run_fold.remote(k) for k in range(10)])

# ─── aggregate & upload tables ────────────────────────────
df = pd.DataFrame(metrics).set_index("fold")
df.to_csv("cv_gru_metrics.csv")

summary = df.select_dtypes(float).agg(["mean", "std"]).round(4)
summary.to_csv("cv_gru_summary.csv")
summary.to_json("cv_gru_summary.json", indent=2)

s3 = boto3.client("s3")
s3.upload_file("cv_gru_metrics.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_metrics.csv")
s3.upload_file("cv_gru_summary.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_summary.csv")
s3.upload_file("cv_gru_summary.json", BUCKET, f"{PREFIX}/tables/cv_gru_summary.json")
print("✓ GRU 10‑fold CV complete; summaries uploaded.")
