# hyperparam_sweep.py
# Local modules
from config4git import get_config
from train4_decoderonly_feature_git import train_model

import multiprocessing as mp
mp.set_start_method("spawn", force=True)    # ← must run only once, at import time

import os, random, itertools, json, uuid, boto3, torch
from concurrent.futures import ProcessPoolExecutor, as_completed

d_model_values     = [32]
d_ff_values        = [32]
N_values           = [2]
num_heads_values   = [2]
gamma_values       = [0]
lr_values          = [0.0001]
weight_values      = [10]

HP_GRID = list(itertools.product(
    d_model_values, d_ff_values, N_values,
    num_heads_values,  gamma_values, lr_values, weight_values))

s3        = boto3.client("s3")
BUCKET    = get_config()["s3_bucket"]


def upload_to_s3_boto(local_path, bucket_name, s3_key):
    print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key} ...")
    s3.upload_file(local_path, bucket_name, s3_key)

def hyperparam_sweep():
    all_combinations = itertools.product(d_model_values, d_ff_values, N_values, num_heads_values, gamma_values, lr_values,weight_values)

    for (d_model, d_ff, N, num_heads, gamma, lr, weight) in all_combinations:
        # 1) Get default config
        config = get_config()

        # 2) Override hyperparams
        config['d_model']   = d_model
        config['d_ff']      = d_ff
        config['N']         = N
        config['num_heads'] = num_heads
        config['gamma']     = gamma
        config['lr']        = lr
        config['weight']    = weight

        # 3) Unique name
        unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_gamma{gamma}_lr{lr}_weight{weight}"
        config['model_basename'] = f"MyProductGPT_{unique_id}"

        # 4) Train model
        final_metrics = train_model(config)

        # 5) Save final metrics for this combo
        metrics_out = {
            "d_model": d_model,
            "d_ff": d_ff,
            "N": N,
            "num_heads": num_heads,
            "gamma": gamma,
            "lr": lr,
            "weight": weight,
            "val_loss": final_metrics['val_loss'],
            "val_ppl": final_metrics['val_ppl'],
            "val_confusion_matrix": final_metrics['val_confusion_matrix'],
            "val_hit_rate": final_metrics['val_hit_rate'],
            "val_f1_score": final_metrics['val_f1_score'],
            "val_auprc": final_metrics['val_auprc'],
            "best_checkpoint_path": final_metrics['best_checkpoint_path']
        }
        metrics_file = f"FeatureBased_FullProductGPT_{unique_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_out, f, indent=2)

        print(f"[Done] {unique_id} -> {metrics_file}")

        # 6) Upload checkpoint + metrics to S3
        bucket_name = config["s3_bucket"]

        best_ckpt_path = final_metrics["best_checkpoint_path"]
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            s3_checkpoint_key = f"checkpoints/{os.path.basename(best_ckpt_path)}"
            upload_to_s3_boto(best_ckpt_path, bucket_name, s3_checkpoint_key)
            os.remove(best_ckpt_path)

        if os.path.exists(metrics_file):
            s3_metrics_key = f"metrics/{metrics_file}"
            upload_to_s3_boto(metrics_file, bucket_name, s3_metrics_key)
            os.remove(metrics_file)

        print("---------------------------------------------------------")

if __name__ == "__main__":
    hyperparam_sweep()