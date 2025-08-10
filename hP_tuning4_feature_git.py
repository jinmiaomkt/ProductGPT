# hyperparam_sweep_parallel_feature_git.py
from __future__ import annotations

import multiprocessing as mp
# 1) force 'spawn' before any CUDA or ProcessPoolExecutor is imported
mp.set_start_method('spawn', force=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import itertools
import os
import json
import uuid

import itertools, json, os, socket 
import boto3, botocore, torch
import torch

from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model

FOLD_ID  = 0
SPEC_URI = "s3://productgptbucket/CV/folds.json"

def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    else:
        with open(uri, "r") as f:
            return json.load(f)

spec = load_fold_spec(SPEC_URI)

# spec["assignment"] is assumed to be {uid: fold_idx, ...}
uids_test = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
uids_trainval = [u for u in spec["assignment"] if u not in uids_test]

# Sanity check
assert len(uids_test) > 0, "No test UIDs for the requested fold."
assert len(uids_trainval) > 0, "No train/val UIDs for the requested fold."

# Hyper‑parameter grids
nb_features_values = [32]
d_model_values     = [128]
d_ff_values        = [128]
N_values           = [6, 8]
num_heads_values   = [4]
gamma_values       = [1]
lr_values          = [1e-3, 5e-4]
weight_values      = [2]

# Precompute every combo
HP_GRID = list(itertools.product(
    nb_features_values,
    d_model_values,
    d_ff_values,
    N_values,
    num_heads_values,
    gamma_values,
    lr_values,
    weight_values
))

s3 = boto3.client("s3")

def upload_to_s3(local_path: str, bucket: str, key: str):
    """Upload a local file to S3."""
    s3.upload_file(local_path, bucket, key)

def get_s3():                # single client per process
    try:    return boto3.client("s3")
    except botocore.exceptions.BotoCoreError: return None

def s3_put(local, bucket, key, s3):
    if s3 is None: return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name} → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-WARN] {e}"); return False
    
def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("",0)); return s.getsockname()[1]

def run_one_experiment(params):
    """
    Unpack one hyperparam tuple, run train_model, save/upload results.
    Returns unique_id.
    """
    nbf, d_model, d_ff, N, num_heads, gamma, lr, weight = params

    # 1) Build config
    config = get_config()

    config.update({
        "mode": "train",
        "fold_id": FOLD_ID,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval
    })

    config["num_epochs"] = 200
    config["ai_rate"] = 15                          # <<<< FIX
    config["seq_len_ai"] = config["ai_rate"] * config["seq_len_tgt"]

    config.update({
        'nb_features': nbf,
        'd_model':    d_model,
        'd_ff':       d_ff,
        'N':          N,
        'num_heads':  num_heads,
        'gamma':      gamma,
        'lr':         lr,
        'weight':     weight,
    })

    # 2) Unique identifier
    unique_id = f"featurebased_performerfeatures{config['nb_features']}_dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_gamma{gamma}_lr{lr}_weight{weight}"
    config['model_basename'] = f"MyProductGPT_FeatureBased_{unique_id}"

    if torch.cuda.device_count():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hash(unique_id)%torch.cuda.device_count())
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())

    # 3) GPU pinning (one process per GPU)
    ngpu = torch.cuda.device_count()
    if ngpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hash(unique_id) % ngpu)

    # 4) Train
    results = train_model(config)

    # 5) Write metrics JSON
    json_path = Path(results["best_checkpoint_path"]).with_suffix(".json")

    if not json_path.exists():
        print(f"[WARN] Expected JSON not found: {json_path}")
    
    s3 = get_s3()
    bucket = config['s3_bucket']

    ckpt = Path(results["best_checkpoint_path"])
    if ckpt.exists() and s3_put(ckpt, bucket, f"FullProductGPT/performer/Feature/checkpoints/{ckpt.name}", s3):
         ckpt.unlink()

    if s3_put(json_path, bucket, f"FullProductGPT/performer/Feature/metrics/{json_path.name}", s3):
        json_path.unlink()
        
    return unique_id

def hyperparam_sweep_parallel(max_workers=None):
    # default: one worker per GPU, or cpu_count-1 if no GPU
    if max_workers is None:
        if torch.cuda.is_available():
            max_workers = torch.cuda.device_count()
        else:
            max_workers = max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_one_experiment, params): params
                   for params in HP_GRID}

        for fut in as_completed(futures):
            params = futures[fut]
            try:
                uid = fut.result()
                print(f"[Done] {uid}")
            except Exception as e:
                print(f"[Error] params={params} -> {e}")

if __name__ == "__main__":
    hyperparam_sweep_parallel()
