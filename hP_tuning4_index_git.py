import multiprocessing as mp
# 1) Force 'spawn' start method before CUDA or ProcessPoolExecutor is initialized
mp.set_start_method('spawn', force=True)

import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# import boto3
import torch

from config4_index_git import get_config
from train4_decoderonly_git import train_model

from google.cloud import storage

def upload_to_gcs(local_path: str, bucket_name: str, destination_blob_name: str):
    """Uploads a file to GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

# hyperparameter ranges
ctx_window_values  = [480, 960, 1920, 3840]
d_model_values     = [32, 64, 128]
d_ff_values        = [32, 64, 128]
N_values           = [4, 6, 8]
num_heads_values   = [4, 8]
weight_values      = [2, 4, 8]

# precompute the grid
HP_GRID = list(itertools.product(
    ctx_window_values,
    d_model_values,
    d_ff_values,
    N_values,
    num_heads_values,
#     lr_values,
    weight_values
))

# Initialize S3 client
# s3 = boto3.client("s3")

# def upload_to_s3(local_path: str, bucket: str, key: str):
#    s3.upload_file(local_path, bucket, key)

def run_one_experiment(params):
    ctx_window, d_model, d_ff, N, num_heads, lr, weight = params

    # 1) Build config
    config = get_config()
    config.update({
        'ctx_window': ctx_window_values,
        'd_model':    d_model,
        'd_ff':       d_ff,
        'N':          N,
        'num_heads':  num_heads,
        'lr':         lr,
        'weight':     weight,
    })

    ctx_window = ctx_window / 15
    # 2) Unique identifier
    unique_id = (
        f"ctx_window{ctx_window}_dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}"
        f"_lr{lr}_weight{weight}"
    )
    config['model_basename'] = f"MyProductGPT_{unique_id}"

    # 3) Pin to a GPU if available
    ngpu = torch.cuda.device_count()
    if ngpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hash(unique_id) % ngpu)

    # 4) Run training
    results = train_model(config)

    # 5) Write metrics JSON
    metrics_file = f"IndexBased_FullProductGPT_{unique_id}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "ctx_window": ctx_window,
            "d_model":   d_model,
            "d_ff":      d_ff,
            "N":         N,
            "num_heads": num_heads,
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

    # 6) Upload checkpoint + metrics to S3 and clean up
    # bucket = config["s3_bucket"]
    gcs_bucket = config["gcp_bucket"]  # <- your bucket name
    
    ckpt = results["best_checkpoint_path"]
    if ckpt and Path(ckpt).exists():
        # upload_to_s3(ckpt, bucket, f"checkpoints/{Path(ckpt).name}")
        upload_to_gcs(ckpt, gcs_bucket, f"checkpoints/{Path(ckpt).name}")
        os.remove(ckpt)

    # upload_to_s3(metrics_file, bucket, f"metrics/{metrics_file}")
    upload_to_gcs(metrics_file, gcs_bucket, f"metrics/{metrics_file}")
    os.remove(metrics_file)

    return unique_id

def hyperparam_sweep_parallel(max_workers=None):
    # default to one worker per GPU, or CPU cores - 1 if no GPU
    if max_workers is None:
        if torch.cuda.is_available():
            max_workers = torch.cuda.device_count()
        else:
            max_workers = max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one_experiment, params): params
            for params in HP_GRID
        }
        for fut in as_completed(futures):
            params = futures[fut]
            try:
                uid = fut.result()
                print(f"[Done] {uid}")
            except Exception as e:
                print(f"[Error] params={params} -> {e}")

if __name__ == "__main__":
    hyperparam_sweep_parallel()
