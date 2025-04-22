# # hyperparam_sweep.py

# import itertools
# import json
# import torch

# # Local modules
# from config4_decision_only_git import get_config
# from train4_decision_only_git import train_model

# import os
# import boto3

# # Define the hyperparameter ranges
# d_model_values = [32, 64, 128, 256]
# d_ff_values    = [32, 64, 128, 256]
# N_values       = [2, 4, 6, 8]
# num_heads_values = [2, 4, 8, 16]
# lr_values      = [0.001, 0.0001, 0.00001, 0.000001]
# weight_values  = [2, 4, 8, 16]

# # Initialize S3 client
# s3 = boto3.client("s3")

# def upload_to_s3_boto(local_path, bucket_name, s3_key):
#     print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key} ...")
#     s3.upload_file(local_path, bucket_name, s3_key)

# def hyperparam_sweep():
#     all_combinations = itertools.product(d_model_values, d_ff_values, N_values, num_heads_values, lr_values, weight_values)

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
#         unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_gamma{gamma}_lr{lr}_weight{weight}"
#         config['model_basename'] = f"DecisionOnly_{unique_id}"

#         # 4) Train model
#         final_metrics = train_model(config)

#         # 5) Save final metrics for this combo
#         metrics_out = {
#             "d_model": d_model,
#             "d_ff": d_ff,
#             "N": N,
#             "num_heads": num_heads,
#             # "gamma": gamma,
#             "lr": lr,
#             "weight": weight,
#             # "gamma": gamma,
#             "val_loss": final_metrics['val_loss'],
#             "val_ppl": final_metrics['val_ppl'],
#             "val_confusion_matrix": final_metrics['val_confusion_matrix'],
#             "val_hit_rate": final_metrics['val_hit_rate'],
#             "val_f1_score": final_metrics['val_f1_score'],
#             "val_auprc": final_metrics['val_auprc'],
#             "best_checkpoint_path": final_metrics['best_checkpoint_path']
#         }
#         metrics_file = f"DecisionOnly_{unique_id}.json"
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
#             s3_metrics_key = f"metrics/{metrics_file}"
#             upload_to_s3_boto(metrics_file, bucket_name, s3_metrics_key)
#             os.remove(metrics_file)

#         print("---------------------------------------------------------")

# if __name__ == "__main__":
#     hyperparam_sweep()
import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
import torch

from config4_decision_only_git import get_config
from train4_decision_only_git import train_model

# hyper‐parameter grids
d_model_values    = [32, 64, 128, 256]
d_ff_values       = [32, 64, 128, 256]
N_values          = [2, 4, 6, 8]
num_heads_values  = [2, 4, 8, 16]
lr_values         = [1e-3, 1e-4, 1e-5, 1e-6]
weight_values     = [2, 4, 8, 16]

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
    d_model, d_ff, N, num_heads, lr, weight = params

    # 1) Build config
    config = get_config()
    config.update({
        "d_model":    d_model,
        "d_ff":       d_ff,
        "N":          N,
        "num_heads":  num_heads,
        "lr":         lr,
        "weight":     weight,
    })

    # 2) Unique basename / id
    unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_lr{lr}_weight{weight}"
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
        upload_to_s3(ckpt, bucket, f"checkpoints/{Path(ckpt).name}")
        os.remove(ckpt)

    if Path(metrics_file).exists():
        upload_to_s3(metrics_file, bucket, f"metrics/{metrics_file}")
        os.remove(metrics_file)

    return unique_id

def hyperparam_sweep_parallel(max_workers=4):
    combos = itertools.product(
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

if __name__ == "__main__":
    # tune max_workers to however many GPUs (or CPU cores) you have
    hyperparam_sweep_parallel(max_workers=4)
