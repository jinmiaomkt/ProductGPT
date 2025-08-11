# run_cv_productgpt.py
from __future__ import annotations
import os, io, json, time, math, botocore
from pathlib import Path

import boto3
import pandas as pd
import torch
import random, os

from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model

S3_BUCKET = "productgptbucket"
SPEC_URI  = "s3://productgptbucket/CV/folds.json"

# Per-fold metrics (JSON) will be saved here
S3_PERFOLD_PREFIX = "CV/metrics/productgpt_full/"        # e.g., productgpt_full/fold_0.json
# Merged CSV of all folds
S3_MERGED_CSV_KEY = "CV/tables/productgpt_full_cv_metrics.csv"

BEST_HP = dict(
    nb_features=16, 
    d_model=128, 
    d_ff=128, 
    N=6, 
    num_heads=4,
    lr=1e-4, 
    weight=2, 
    gamma=1.0,   # keep gamma if your loss uses it
)

def _s3():
    return boto3.client("s3")

def _s3_exists(bucket, key) -> bool:
    s3 = _s3()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False

def _load_spec(spec_uri: str):
    s3 = _s3()
    if spec_uri.startswith("s3://"):
        bucket, key = spec_uri[5:].split("/", 1)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    else:
        with open(spec_uri, "r") as f:
            return json.load(f)

def _upload_json(obj: dict, bucket: str, key: str):
    s3 = _s3()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj).encode("utf-8"))

def _download_csv(bucket: str, key: str) -> pd.DataFrame | None:
    s3 = _s3()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise

def _upload_csv(df: pd.DataFrame, bucket: str, key: str):
    s3 = _s3()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)

def _build_cfg(fold_id: int, spec: dict) -> dict:
    # split by UID (no leakage)
    test_uids = [u for u, f in spec["assignment"].items() if f == fold_id]
    train_uids= [u for u in spec["assignment"] if u not in test_uids]
    assert test_uids and train_uids

    cfg = get_config()
    cfg.update({
        "mode": "train",
        "fold_id": fold_id,
        "uids_test": test_uids,
        "uids_trainval": train_uids,
        # speed + stability knobs
        "num_epochs": 60,
        "patience": 5,
        "fp16": True,                 # if your trainer reads this
        "ai_rate": 15,                # keep seq_len_ai in check
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    # lock best hyperparams
    cfg.update(BEST_HP)

    # deterministic single-GPU run
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # make filenames deterministic by including fold
    uid = (f"FullProductGPT_featurebased_performerfeatures{cfg['nb_features']}"
           f"_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}"
           f"_heads{cfg['num_heads']}_lr{cfg['lr']}_w{cfg['weight']}_fold{fold_id}")
    cfg["model_basename"] = uid
    return cfg

def _train_one_fold(fold_id: int, spec: dict) -> dict:
    cfg = _build_cfg(fold_id, spec)

    # Pick a free port in a safe range to avoid collisions
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(random.randint(20000, 29999))

    results = train_model(cfg)   # your trainer returns dict with best_checkpoint_path, etc.

    # Metrics JSON lives next to the checkpoint (same basename, .json)
    metrics_path = Path(results["best_checkpoint_path"]).with_suffix(".json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON missing: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # decorate and return (keep minimal to merge later)
    metrics["fold_id"] = fold_id
    metrics["model"]   = "ProductGPT_Full"
    metrics["basename"]= cfg["model_basename"]

    # upload per-fold metrics json (compact)
    per_fold_key = f"{S3_PERFOLD_PREFIX}fold_{fold_id}.json"
    _upload_json(metrics, S3_BUCKET, per_fold_key)

    # optional: upload ckpt & delete local to save disk (your trainer may already do this)
    # NOTE: you already handle checkpoint upload elsewhere; skipping here.

    # tidy local metrics json if you want
    try: metrics_path.unlink()
    except Exception: pass

    return metrics

def main():
    spec = _load_spec(SPEC_URI)
    folds = list(range(10))  # 0..9

    done_rows = []
    # resume: skip folds that already have per-fold JSON on S3
    for k in folds:
        per_fold_key = f"{S3_PERFOLD_PREFIX}fold_{k}.json"
        if _s3_exists(S3_BUCKET, per_fold_key):
            print(f"[resume] found {per_fold_key} → skipping train")
            body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=per_fold_key)["Body"].read()
            done_rows.append(json.loads(body))
            continue

        print(f"\n[Fold {k}] starting…")
        row = _train_one_fold(k, spec)
        done_rows.append(row)
        print(f"[Fold {k}] done; uploaded {per_fold_key}")

    # merge to CSV
    df = pd.DataFrame(done_rows)

    # select the columns you care about; add more if your JSON has them:
    cols = [
        "fold_id",
        "val_loss", 
        "val_ppl",
        "val_all_hit_rate", 
        "val_all_f1_score", 
        "val_all_auprc",
        "val_all_rev_mae",
        "test_all_hit_rate", 
        "test_all_f1_score", 
        "test_all_auprc",
        "model", 
        "basename"
    ]
    cols = [c for c in cols if c in df.columns]  # keep only present
    df = df[cols].sort_values("fold_id")

    # append to existing CSV or create fresh
    existing = _download_csv(S3_BUCKET, S3_MERGED_CSV_KEY)
    if existing is not None:
        out = pd.concat([existing, df], ignore_index=True)
        # drop duplicate folds (keep latest)
        if "fold_id" in out.columns:
            out = out.sort_values(["fold_id"]).drop_duplicates(["fold_id"], keep="last")
    else:
        out = df

    _upload_csv(out, S3_BUCKET, S3_MERGED_CSV_KEY)
    print(f"[CV] merged CSV → s3://{S3_BUCKET}/{S3_MERGED_CSV_KEY}")

if __name__ == "__main__":
    main()
