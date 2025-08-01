from __future__ import annotations

import multiprocessing as mp

# force every new process to be launched with 'spawn'
mp.set_start_method('spawn', force=True)

import multiprocessing as mp
import torch

if torch.cuda.is_available():
    max_workers = torch.cuda.device_count()
else:
    max_workers = max(1, mp.cpu_count() - 1)  # leave one core free

import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
import torch

from config4 import get_config
from train1_decision_only_performer_aws import train_model

# hyper‐parameter grids
nb_features_values = [8, 16]
d_model_values = [32, 64]
d_ff_values = [32, 64]
N_values = [6, 8]
num_heads_values = [4, 8]
lr_values = [1e-4]
weight_values = [2, 4, 8]

# S3 client
s3 = boto3.client("s3")

def upload_to_s3(local_path: str, bucket: str, key: str):
    s3.upload_file(local_path, bucket, key)

def run_one_experiment(params):
    """
    Worker function: unpacks one hyperparam tuple, runs training,
    writes out metrics JSON and uploads checkpoint+metrics to S3.
    Returns unique_id for logging.
    """
    nbf, d_model, d_ff, N, num_heads, lr, weight = params

    # 1) Build config
    config = get_config()
    config.update({
        "nb_features": nbf,
        "d_model":    d_model,
        "d_ff":       d_ff,
        "N":          N,
        "num_heads":  num_heads,
        "lr":         lr,
        "weight":     weight,
    })

    # 2) Unique basename / id
    unique_id = f"decisiononly_performerfeatures{config['nb_features']}_ai_rate{config['ai_rate']}_dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_lr{lr}_weight{weight}"
    config["model_basename"] = f"DecisionOnly_{unique_id}"

    # 3) (Optional) pin each process to a different GPU if you have >1 GPU
    #    e.g. round‑robin assignment:
    ngpu = torch.cuda.device_count()
    if ngpu > 0:
        this_rank = hash(unique_id) % ngpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(this_rank)

    # 4) Train!
    final_metrics = train_model(config)

    # 5) Dump metrics locally
    metrics_file = f"DecisionOnly_{unique_id}.json"
    with open(metrics_file, "w") as f:
        json.dump({
            **{k: v for k, v in final_metrics.items() if not isinstance(v, list)},
            # if you need to store confusion_matrix lists, include them explicitly…
            "val_confusion_matrix": final_metrics["val_confusion_matrix"],
        }, f, indent=2)

    # 6) Upload checkpoint + metrics to S3
    bucket = config["s3_bucket"]

    ckpt = final_metrics["best_checkpoint_path"]
    if ckpt and Path(ckpt).exists():
        upload_to_s3(ckpt, bucket, f"DecisionOnly/checkpoints/{Path(ckpt).name}")
        # upload_to_gcs(ckpt, bucket, f"checkpoints/{Path(ckpt).name}")
        os.remove(ckpt)

    if Path(metrics_file).exists():
        upload_to_s3(metrics_file, bucket, f"DecisionOnly/metrics/{metrics_file}")
        # upload_to_gcs(metrics_file, bucket, f"metrics/{metrics_file}")
        os.remove(metrics_file)

    return unique_id

def hyperparam_sweep_parallel(max_workers=max_workers):
    combos = itertools.product(
        nb_features_values,
        d_model_values,
        d_ff_values,
        N_values,
        num_heads_values,
        lr_values,
        weight_values
    )

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        # submit all jobs
        futures = {exe.submit(run_one_experiment, combo): combo for combo in combos}

        # as each finishes, log it
        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                uid = fut.result()
                print(f"[Done] {uid}")
            except Exception as e:
                print(f"[Error] combo={combo} -> {e}")

# if __name__ == "__main__":
#     # tune max_workers to however many GPUs (or CPU cores) you have
#     hyperparam_sweep_parallel(max_workers=4)

if __name__ == "__main__":
    # launch
    hyperparam_sweep_parallel(max_workers=max_workers)
