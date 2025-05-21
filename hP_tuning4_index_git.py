# hP_tuning4_index_git.py
# ---------------------------------------------------------------------------#
#  Multiprocessing / env setup                                               #
# ---------------------------------------------------------------------------#
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import itertools
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
import botocore
import torch

from config4_index_git      import get_config       # <- your config file
from train4_decoderonly_git import train_model      # <- your training loop

# ---------------------------------------------------------------------------#
#  Hyper-parameter grid                                                      #
# ---------------------------------------------------------------------------#
ctx_window_values  = [480, 960, 1920, 3840]
d_model_values     = [32, 64, 128]
d_ff_values        = [32, 64, 128]
N_values           = [4, 6, 8]
num_heads_values   = [4, 8]
lr_values          = [1e-4]
weight_values      = [2, 4, 8]

HP_GRID = list(itertools.product(
    ctx_window_values,
    d_model_values,
    d_ff_values,
    N_values,
    num_heads_values,
    lr_values,
    weight_values,
))

# ---------------------------------------------------------------------------#
#  S3 helpers                                                                #
# ---------------------------------------------------------------------------#
def get_s3_client():
    try:
        return boto3.client("s3")
    except botocore.exceptions.NoCredentialsError:
        print("[WARN] No AWS credentials – skipping uploads")
        return None

def upload_to_s3(local_path: Path, bucket: str, key: str, s3):
    if s3 is None:
        return False
    try:
        s3.upload_file(str(local_path), bucket, key)
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[WARN] S3 upload failed: {e}")
        return False

# ---------------------------------------------------------------------------#
#  One experiment                                                            #
# ---------------------------------------------------------------------------#
def run_one_experiment(params):
    (ctx_window, d_model, d_ff,
     N, num_heads, lr, weight) = params

    # ---------- build per-run config ------------------------------------
    cfg = get_config()
    cfg.update({
        "ctx_window": ctx_window,
        "seq_len_ai": ctx_window,       # keeps model & dataset in sync
        "seq_len_tgt": ctx_window // 15,
        "d_model":    d_model,
        "d_ff":       d_ff,
        "N":          N,
        "num_heads":  num_heads,
        "lr":         lr,
        "weight":     weight,
        "ai_rate":    15,
    })

    slots = ctx_window // cfg["ai_rate"]
    uid = (f"ctx{slots}_dmodel{d_model}_ff{d_ff}_N{N}_"
           f"heads{num_heads}_lr{lr}_weight{weight}")
    stem = f"MyProductGPT_{uid}"
    cfg["model_basename"] = stem

    # ---------- pin GPU --------------------------------------------------
    ngpu = torch.cuda.device_count()
    if ngpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hash(uid) % ngpu)

    # ---------- train ----------------------------------------------------
    result = train_model(cfg)          # returns dict with best_checkpoint_path

    # ---------- write metrics JSON --------------------------------------
    metrics_file = Path(f"{stem}.json")
    with metrics_file.open("w") as f:
        json.dump(result, f, indent=2)

    # ---------- upload to S3 --------------------------------------------
    s3          = get_s3_client()
    bucket      = cfg.get("s3_bucket", "")
    s3_prefix   = cfg.get("s3_prefix", "").rstrip("/")
    if s3_prefix:
        s3_prefix += "/"

    # checkpoint
    ckpt_path = Path(result["best_checkpoint_path"])
    if ckpt_path.exists():
        key = f"{s3_prefix}checkpoints/{ckpt_path.name}"
        if upload_to_s3(ckpt_path, bucket, key, s3):
            ckpt_path.unlink()

    # metrics
    key = f"{s3_prefix}metrics/{metrics_file.name}"
    if upload_to_s3(metrics_file, bucket, key, s3):
        metrics_file.unlink()

    return uid

# ---------------------------------------------------------------------------#
#  Sweep driver                                                              #
# ---------------------------------------------------------------------------#
def hyperparam_sweep_parallel(max_workers=None):
    if max_workers is None:
        max_workers = torch.cuda.device_count() or max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futs = {exe.submit(run_one_experiment, hp): hp for hp in HP_GRID}
        for fut in as_completed(futs):
            hp = futs[fut]
            try:
                name = fut.result()
                print(f"[Done] {name}")
            except Exception as e:
                print(f"[Error] params={hp} -> {e}")

# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    hyperparam_sweep_parallel()
