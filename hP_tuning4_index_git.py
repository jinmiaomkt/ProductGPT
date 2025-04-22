# # hyperparam_sweep.py

# import itertools
# import json
# import torch

# # Local modules
# from config4git import get_config
# from train4_decoderonly_git import train_model

# import os
# import boto3

# # Define the hyperparameter ranges
# d_model_values = [32, 64, 128]
# d_ff_values    = [32, 64, 128]
# N_values       = [4, 6, 8]
# num_heads_values = [4, 6, 8]
# lr_values      = [0.0001, 0.00001, 0.000001]
# weight_values  = [4, 8]

# # Initialize S3 client
# s3 = boto3.client("s3")

# def upload_to_s3_boto(local_path, bucket_name, s3_key):
#     print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key} ...")
#     s3.upload_file(local_path, bucket_name, s3_key)

# def hyperparam_sweep():
#     all_combinations = itertools.product(d_model_values, d_ff_values, N_values, num_heads_values, lr_values,weight_values)

#     for (d_model, d_ff, N, num_heads, lr, weight) in all_combinations:
#         # 1) Get default config
#         config = get_config()

#         # 2) Override hyperparams
#         config['d_model']   = d_model
#         config['d_ff']      = d_ff
#         config['N']         = N
#         config['num_heads'] = num_heads
#         config['lr']        = lr
#         config['weight']    = weight

#         # 3) Unique name
#         unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_lr{lr}_weight{weight}"
#         config['model_basename'] = f"MyProductGPT_{unique_id}"

#         # 4) Train model
#         final_metrics = train_model(config)

#         # 5) Save final metrics for this combo
#         metrics_out = {
#             "d_model": d_model,
#             "d_ff": d_ff,
#             "N": N,
#             "num_heads": num_heads,
#             "lr": lr,
#             "weight": weight,
#             "val_loss": final_metrics['val_loss'],
#             "val_ppl": final_metrics['val_ppl'],
#             "val_confusion_matrix": final_metrics['val_confusion_matrix'],
#             "val_hit_rate": final_metrics['val_hit_rate'],
#             "val_f1_score": final_metrics['val_f1_score'],
#             "val_auprc": final_metrics['val_auprc'],
#             "best_checkpoint_path": final_metrics['best_checkpoint_path']
#         }
#         metrics_file = f"IndexBased_FullProductGPT_{unique_id}.json"
#         with open(metrics_file, 'w') as f:
#             json.dump(metrics_out, f, indent=2)

#         print(f"[Done] {unique_id} -> {metrics_file}")

#         # 6) Upload checkpoint + metrics to S3
#         bucket_name = config["s3_bucket"]

#         best_ckpt_path = final_metrics["best_checkpoint_path"]
#         if best_ckpt_path and os.path.exists(best_ckpt_path):
#             s3_checkpoint_key = f"checkpoints/{os.path.basename(best_ckpt_path)}"
#             upload_to_s3_boto(best_ckpt_path, bucket_name, s3_checkpoint_key)
#             os.remove(best_ckpt_path)

#         if os.path.exists(metrics_file):
#             s3_metrics_key = f"matrics/{metrics_file}"
#             upload_to_s3_boto(metrics_file, bucket_name, s3_metrics_key)
#             os.remove(metrics_file)

#         print("---------------------------------------------------------")

# if __name__ == "__main__":
#     hyperparam_sweep()

import multiprocessing as mp
# 1) Force 'spawn' start method before CUDA or ProcessPoolExecutor is initialized
mp.set_start_method('spawn', force=True)

import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
import torch

from config4git import get_config
from train4_decoderonly_git import train_model

# hyperparameter ranges
d_model_values     = [32, 64, 128]
d_ff_values        = [32, 64, 128]
N_values           = [4, 6, 8]
num_heads_values   = [4, 6, 8]
lr_values          = [1e-4, 1e-5, 1e-6]
weight_values      = [4, 8]

# precompute the grid
HP_GRID = list(itertools.product(
    d_model_values,
    d_ff_values,
    N_values,
    num_heads_values,
    lr_values,
    weight_values
))

# Initialize S3 client
s3 = boto3.client("s3")

def upload_to_s3(local_path: str, bucket: str, key: str):
    s3.upload_file(local_path, bucket, key)

def run_one_experiment(params):
    d_model, d_ff, N, num_heads, lr, weight = params

    # 1) Build config
    config = get_config()
    config.update({
        'd_model':    d_model,
        'd_ff':       d_ff,
        'N':          N,
        'num_heads':  num_heads,
        'lr':         lr,
        'weight':     weight,
    })

    # 2) Unique identifier
    unique_id = (
        f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}"
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
    bucket = config["s3_bucket"]

    ckpt = results["best_checkpoint_path"]
    if ckpt and Path(ckpt).exists():
        upload_to_s3(ckpt, bucket, f"checkpoints/{Path(ckpt).name}")
        os.remove(ckpt)

    upload_to_s3(metrics_file, bucket, f"metrics/{metrics_file}")
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
