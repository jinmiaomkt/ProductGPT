import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import itertools, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3, botocore
import numpy as np
import torch

from config4_decision_only_git import get_config
from train4_decision_only_aws  import train_model

# ───────────────────────────── hyper-parameter grid ──────────────────────
seq_len_ai_values = [32, 64, 128, 256]
d_model_values    = [256, 512]
d_ff_values       = [128, 256, 512]
N_values          = [6, 8, 10]
num_heads_values  = [8, 16, 32]
lr_values         = [1e-4]
weight_values     = [1, 2, 4]

HP_GRID = list(itertools.product(
    seq_len_ai_values,
    d_model_values,
    d_ff_values,
    N_values,
    num_heads_values,
    lr_values,
    weight_values,
))

# ───────────────────────────── S3 helpers ────────────────────────────────
def get_s3_client():
    try:
        return boto3.client("s3")
    except botocore.exceptions.BotoCoreError as e:
        print(f"[WARN] could not create S3 client: {e}")
        return None

def upload_to_s3(local: Path, bucket: str, key: str, s3) -> bool:
    if s3 is None:
        return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] uploaded → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-ERROR] {e}")
        return False

# ───────────────────────────── JSON helper ───────────────────────────────
def _json_safe(obj):
    """Recursively convert NumPy types → native Python so json.dump works."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

# ───────────────────────────── worker fn ─────────────────────────────────
def run_one_experiment(params):
    seq_len_ai, d_model, d_ff, N, num_heads, lr, weight = params

    # 1) build config
    cfg = get_config()
    cfg.update({
        "seq_len_ai" : seq_len_ai,
        "seq_len_tgt": seq_len_ai // cfg["ai_rate"],
        "d_model"    : d_model,
        "d_ff"       : d_ff,
        "N"          : N,
        "num_heads"  : num_heads,
        "lr"         : lr,
        "weight"     : weight,
    })

    uid = (f"ctx{seq_len_ai}_dmodel{d_model}_ff{d_ff}_N{N}_"
           f"heads{num_heads}_lr{lr}_weight{weight}")
    cfg["model_basename"] = f"DecisionOnly_{uid}"

    # pin GPU (round-robin)
    n_gpu = torch.cuda.device_count()
    if n_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hash(uid) % n_gpu)

    # 2) train
    metrics = train_model(cfg)              # returns dict

    # 3) write metrics JSON locally (NumPy → Python)
    metrics_path = Path(f"DecisionOnly_{uid}.json")
    with metrics_path.open("w") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    # 4) upload artefacts
    s3      = get_s3_client()
    bucket  = cfg["s3_bucket"]
    prefix  = (cfg.get("s3_prefix") or "").rstrip("/")
    if prefix:
        prefix += "/"

    # checkpoint
    ckpt_path = Path(metrics["best_checkpoint_path"])
    if ckpt_path.exists():
        ck_key = f"{prefix}DecisionOnly/checkpoints/{ckpt_path.name}"
        print("[INFO] artefacts will be saved to")
        print(f"  • s3://{bucket}/{ck_key}")
        ok = upload_to_s3(ckpt_path, bucket, ck_key, s3)
        if ok:
            ckpt_path.unlink()

    # metrics JSON
    json_key = f"{prefix}DecisionOnly/metrics/{metrics_path.name}"
    print(f"  • s3://{bucket}/{json_key}")
    if upload_to_s3(metrics_path, bucket, json_key, s3):
        metrics_path.unlink()

    return uid

# ───────────────────────────── sweep driver ──────────────────────────────
def hyperparam_sweep_parallel(max_workers=None):
    if max_workers is None:
        max_workers = torch.cuda.device_count() or max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_one_experiment, hp): hp for hp in HP_GRID}
        for fut in as_completed(futs):
            hp = futs[fut]
            try:
                print(f"[Done] {fut.result()}")
            except Exception as e:
                print(f"[Error] params={hp} -> {e}")

# ───────────────────────────── entry point ───────────────────────────────
if __name__ == "__main__":
    hyperparam_sweep_parallel()
