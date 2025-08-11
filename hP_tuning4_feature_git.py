# # hyperparam_sweep_parallel_feature_git.py
# from __future__ import annotations

# import multiprocessing as mp
# # 1) force 'spawn' before any CUDA or ProcessPoolExecutor is imported
# mp.set_start_method('spawn', force=True)

# from concurrent.futures import ProcessPoolExecutor, as_completed
# from pathlib import Path
# import itertools
# import os
# import json
# import uuid

# import itertools, json, os, socket 
# import boto3, botocore, torch
# import torch

# from config4 import get_config
# from train4_decoderonly_performer_feature_aws import train_model

# FOLD_ID  = 0
# SPEC_URI = "s3://productgptbucket/CV/folds.json"

# def load_fold_spec(uri: str):
#     if uri.startswith("s3://"):
#         bucket, key = uri[5:].split("/", 1)
#         body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
#         return json.loads(body)
#     else:
#         with open(uri, "r") as f:
#             return json.load(f)

# spec = load_fold_spec(SPEC_URI)

# # spec["assignment"] is assumed to be {uid: fold_idx, ...}
# uids_test = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
# uids_trainval = [u for u in spec["assignment"] if u not in uids_test]

# # Sanity check
# assert len(uids_test) > 0, "No test UIDs for the requested fold."
# assert len(uids_trainval) > 0, "No train/val UIDs for the requested fold."

# # Hyper‑parameter grids
# nb_features_values = [32]
# d_model_values     = [128]
# d_ff_values        = [128]
# N_values           = [6, 8]
# num_heads_values   = [4]
# gamma_values       = [1]
# lr_values          = [1e-3, 5e-4]
# weight_values      = [2]

# # Precompute every combo
# HP_GRID = list(itertools.product(
#     nb_features_values,
#     d_model_values,
#     d_ff_values,
#     N_values,
#     num_heads_values,
#     gamma_values,
#     lr_values,
#     weight_values
# ))

# s3 = boto3.client("s3")

# def upload_to_s3(local_path: str, bucket: str, key: str):
#     """Upload a local file to S3."""
#     s3.upload_file(local_path, bucket, key)

# def get_s3():                # single client per process
#     try:    return boto3.client("s3")
#     except botocore.exceptions.BotoCoreError: return None

# def s3_put(local, bucket, key, s3):
#     if s3 is None: return False
#     try:
#         s3.upload_file(str(local), bucket, key)
#         print(f"[S3] {local.name} → s3://{bucket}/{key}")
#         return True
#     except botocore.exceptions.BotoCoreError as e:
#         print(f"[S3-WARN] {e}"); return False
    
# def free_port():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(("",0)); return s.getsockname()[1]

# def run_one_experiment(params):
#     """
#     Unpack one hyperparam tuple, run train_model, save/upload results.
#     Returns unique_id.
#     """
#     nbf, d_model, d_ff, N, num_heads, gamma, lr, weight = params

#     # 1) Build config
#     config = get_config()

#     config.update({
#         "mode": "train",
#         "fold_id": FOLD_ID,
#         "uids_test": uids_test,
#         "uids_trainval": uids_trainval
#     })

#     config["num_epochs"] = 200
#     config["ai_rate"] = 15                          # <<<< FIX
#     config["seq_len_ai"] = config["ai_rate"] * config["seq_len_tgt"]

#     config.update({
#         'nb_features': nbf,
#         'd_model':    d_model,
#         'd_ff':       d_ff,
#         'N':          N,
#         'num_heads':  num_heads,
#         'gamma':      gamma,
#         'lr':         lr,
#         'weight':     weight,
#     })

#     # 2) Unique identifier
#     unique_id = f"featurebased_performerfeatures{config['nb_features']}_dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_gamma{gamma}_lr{lr}_weight{weight}"
#     config['model_basename'] = f"MyProductGPT_FeatureBased_{unique_id}"

#     if torch.cuda.device_count():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(hash(unique_id)%torch.cuda.device_count())
#     os.environ["MASTER_ADDR"] = "127.0.0.1"
#     os.environ["MASTER_PORT"] = str(free_port())

#     # 3) GPU pinning (one process per GPU)
#     ngpu = torch.cuda.device_count()
#     if ngpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(hash(unique_id) % ngpu)

#     # 4) Train
#     results = train_model(config)

#     # 5) Write metrics JSON
#     json_path = Path(results["best_checkpoint_path"]).with_suffix(".json")

#     if not json_path.exists():
#         print(f"[WARN] Expected JSON not found: {json_path}")
    
#     s3 = get_s3()
#     bucket = config['s3_bucket']

#     ckpt = Path(results["best_checkpoint_path"])
#     if ckpt.exists() and s3_put(ckpt, bucket, f"FullProductGPT/performer/Feature/checkpoints/{ckpt.name}", s3):
#          ckpt.unlink()

#     if s3_put(json_path, bucket, f"FullProductGPT/performer/Feature/metrics/{json_path.name}", s3):
#         json_path.unlink()
        
#     return unique_id

# def hyperparam_sweep_parallel(max_workers=None):
#     # default: one worker per GPU, or cpu_count-1 if no GPU
#     if max_workers is None:
#         if torch.cuda.is_available():
#             max_workers = torch.cuda.device_count()
#         else:
#             max_workers = max(1, mp.cpu_count() - 1)

#     with ProcessPoolExecutor(max_workers=max_workers) as exe:
#         futures = {exe.submit(run_one_experiment, params): params
#                    for params in HP_GRID}

#         for fut in as_completed(futures):
#             params = futures[fut]
#             try:
#                 uid = fut.result()
#                 print(f"[Done] {uid}")
#             except Exception as e:
#                 print(f"[Error] params={params} -> {e}")

# if __name__ == "__main__":
#     hyperparam_sweep_parallel()


# random_search_feature_performer.py
from __future__ import annotations
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os, json, uuid, math, random, socket
from pathlib import Path
import boto3, botocore, torch

from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model

FOLD_ID  = 0
SPEC_URI = "s3://productgptbucket/CV/folds.json"
S3_METRICS_PREFIX = "FullProductGPT/performer/Feature/metrics/"
S3_CKPT_PREFIX    = "FullProductGPT/performer/Feature/checkpoints/"

TRIALS          = 32           # total budget
MAX_EPOCHS      = 120          # cap to save time
PRUNE_AT_EPOCH  = 20           # prune quickly
PRUNE_METRIC    = "val_all_auprc"
PRUNE_TOPK      = 6            # keep running only if in top-K by epoch PRUNE_AT_EPOCH

random.seed(1337)

def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    else:
        with open(uri, "r") as f:
            return json.load(f)

spec = load_fold_spec(SPEC_URI)
uids_test     = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
uids_trainval = [u for u in spec["assignment"] if u not in uids_test]
assert uids_test and uids_trainval

s3 = boto3.client("s3")

def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False

def s3_put(local, bucket, key):
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name} → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-WARN] {e}")
        return False

def free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("",0))
        return s.getsockname()[1]

def sample_hp():
    # --- sample until constraints pass ---
    while True:
        nb_features = random.choice([32, 48, 64])
        d_model     = random.choice([96, 128, 160, 256])
        num_heads   = random.choice([4, 6, 8])
        if d_model % num_heads != 0: 
            continue
        head_dim = d_model // num_heads
        if head_dim < 16:
            continue

        N         = random.randint(4, 8)
        dff_mult  = random.choice([2, 3, 4])
        d_ff      = min(d_model * dff_mult, 512)

        dropout   = round(random.uniform(0.0, 0.2), 3)
        # log-uniform LR
        lr        = random.choice([1e-3, 5e-4, 1e-4, 5e-4])
        # 10 ** random.uniform(math.log10(3e-5), math.log10(5e-4))
        lr        = float(f"{lr:.6g}")

        weight    = random.choice([1, 2, 4, 6, 8])
        gamma     = round(random.uniform(0.8, 1.5), 3)
        label_smoothing = round(random.uniform(0.0, 0.1), 3)
        warmup_steps    = random.choice([500, 1000, 2000])

        return {
            "nb_features": nb_features,
            "d_model": d_model,
            "d_ff": d_ff,
            "N": N,
            "num_heads": num_heads,
            "dropout": dropout,
            "lr": lr,
            "weight": weight,
            "gamma": gamma,
            "label_smoothing": label_smoothing,
            "warmup_steps": warmup_steps
        }

def build_config(hp):
    cfg = get_config()
    cfg.update({
        "mode": "train",
        "fold_id": FOLD_ID,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval,
        "num_epochs": MAX_EPOCHS,
        # keep ai_rate modest to reduce seq_len_ai explosion
        "ai_rate": 10,
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    cfg.update(hp)

    # optional keys your code may read:
    cfg["dropout_attn"] = hp["dropout"]
    cfg["dropout_ffn"]  = hp["dropout"]
    cfg["label_smoothing"] = hp["label_smoothing"]
    cfg["warmup_steps"]    = hp["warmup_steps"]

    uid = (
        f"feat{hp['nb_features']}_dm{hp['d_model']}_ff{hp['d_ff']}_"
        f"N{hp['N']}_h{hp['num_heads']}_do{hp['dropout']}_"
        f"lr{hp['lr']}_w{hp['weight']}_g{hp['gamma']}_"
        f"ls{hp['label_smoothing']}_wu{hp['warmup_steps']}"
    )
    cfg["model_basename"] = f"MyProductGPT_FeatureBased_{uid}"
    return cfg, uid

def run_trial(hp):
    cfg, uid = build_config(hp)

    # single GPU; avoid multi-proc conflicts
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())

    results = train_model(cfg)  # your trainer should already do val-per-epoch & early stop

    metrics_path = Path(results["best_checkpoint_path"]).with_suffix(".json")
    if not metrics_path.exists():
        print(f"[WARN] Expected JSON not found: {metrics_path}")
        return uid, None

    # upload & cleanup
    bucket = cfg["s3_bucket"]
    ckpt = Path(results["best_checkpoint_path"])
    if ckpt.exists() and s3_put(ckpt, bucket, f"{S3_CKPT_PREFIX}{ckpt.name}"):
        ckpt.unlink()

    if s3_put(metrics_path, bucket, f"{S3_METRICS_PREFIX}{metrics_path.name}"):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics_path.unlink()
        return uid, metrics
    else:
        return uid, None

def random_search():
    # keep a sliding window of top-K metric at PRUNE_AT_EPOCH for pruning
    epoch20_cut = []

    for t in range(TRIALS):
        hp = sample_hp()
        cfg, uid = build_config(hp)

        # skip if metrics already on S3 (resume)
        if s3_key_exists(cfg["s3_bucket"], f"{S3_METRICS_PREFIX}{cfg['model_basename']}.json"):
            print(f"[Skip-existing] {uid}")
            continue

        print(f"\n[Trial {t+1}/{TRIALS}] {uid}\nHP: {hp}")

        # hint to your trainer to allow mid-run prune: set max epochs but
        # let trainer dump a mid-epoch JSON checkpoint with PRUNE_AT_EPOCH metrics
        os.environ["PGPT_PRUNE_AT_EPOCH"] = str(PRUNE_AT_EPOCH)

        uid, metrics = run_trial(hp)

        # Optional: update pruning frontier based on a sidecar JSON your trainer writes at epoch=PRUNE_AT_EPOCH
        # If your trainer doesn’t write that, you can rely on its own early stopping and skip this.
        # Example expectation:
        # /tmp/<uid>_epoch20.json containing { "val_all_auprc": x }
        sidecar = Path(f"/tmp/{uid}_epoch{PRUNE_AT_EPOCH}.json")
        if sidecar.exists():
            with open(sidecar) as f:
                m = json.load(f)
            epoch20_cut.append(m.get(PRUNE_METRIC, 0.0))
            epoch20_cut = sorted(epoch20_cut, reverse=True)[:PRUNE_TOPK]
            sidecar.unlink()

if __name__ == "__main__":
    random_search()
