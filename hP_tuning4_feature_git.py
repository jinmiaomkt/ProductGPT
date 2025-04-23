# hyperparam_sweep_parallel_feature_git.py

import multiprocessing as mp
# 1) force 'spawn' before any CUDA or ProcessPoolExecutor is imported
mp.set_start_method('spawn', force=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import itertools
import os
import json
import uuid

import boto3
import torch

from config4git import get_config
from train4_decoderonly_feature_git import train_model

# Hyper‑parameter grids
d_model_values    = [64]
d_ff_values       = [64]
N_values          = [4, 6, 8]
num_heads_values  = [4, 6, 8]
gamma_values      = [0]
lr_values         = [1e-3,1e-4,1e-5]
weight_values     = [2, 4, 8, 10]

# Precompute every combo
HP_GRID = list(itertools.product(
    d_model_values,
    d_ff_values,
    N_values,
    num_heads_values,
    gamma_values,
    lr_values,
    weight_values
))

# S3 client
s3 = boto3.client("s3")

def upload_to_s3(local_path: str, bucket: str, key: str):
    """Upload a local file to S3."""
    s3.upload_file(local_path, bucket, key)

def run_one_experiment(params):
    """
    Unpack one hyperparam tuple, run train_model, save/upload results.
    Returns unique_id.
    """
    d_model, d_ff, N, num_heads, gamma, lr, weight = params

    # 1) Build config
    config = get_config()
    config.update({
        'd_model':    d_model,
        'd_ff':       d_ff,
        'N':          N,
        'num_heads':  num_heads,
        'gamma':      gamma,
        'lr':         lr,
        'weight':     weight,
    })

    # 2) Unique identifier
    unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_" \
                f"gamma{gamma}_lr{lr}_weight{weight}"
    config['model_basename'] = f"MyProductGPT_{unique_id}"

    # 3) GPU pinning (one process per GPU)
    ngpu = torch.cuda.device_count()
    if ngpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hash(unique_id) % ngpu)

    # 4) Train
    results = train_model(config)

    # 5) Write metrics JSON
    metrics_file = f"FeatureBased_FullProductGPT_{unique_id}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "d_model":   d_model,
            "d_ff":      d_ff,
            "N":         N,
            "num_heads": num_heads,
            "gamma":     gamma,
            "lr":        lr,
            "weight":    weight,
            "val_loss":  results['val_loss'],
            "val_ppl":   results['val_ppl'],
            "val_confusion_matrix": results['val_confusion_matrix'],
            "val_hit_rate":          results['val_hit_rate'],
            "val_f1_score":          results['val_f1_score'],
            "val_auprc":             results['val_auprc'],
            "best_checkpoint_path":  results['best_checkpoint_path']
        }, f, indent=2)

    # 6) Upload checkpoint + metrics, then clean up
    bucket = config['s3_bucket']
    ckpt   = results['best_checkpoint_path']
    if ckpt and Path(ckpt).exists():
        upload_to_s3(ckpt, bucket, f"checkpoints/{Path(ckpt).name}")
        os.remove(ckpt)

    upload_to_s3(metrics_file, bucket, f"metrics/{metrics_file}")
    os.remove(metrics_file)

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
