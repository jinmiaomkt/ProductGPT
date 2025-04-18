# hyperparam_sweep.py
# Local modules
from config4git import get_config
from train4_decoderonly_feature_git import train_model

import multiprocessing as mp
mp.set_start_method("spawn", force=True)    # ← must run only once, at import time

import os, random, itertools, json, uuid, boto3, torch
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------------------
# 1.  Search space
# ----------------------------------------------------------------------------------
d_model_values     = [64, 128, 256]
d_ff_values        = [64, 128, 256]
N_values           = [2, 4, 6, 8]
num_heads_values   = [2, 4, 8, 16]

d_model_values     = [256]
d_ff_values        = [256]
N_values           = [8]
num_heads_values   = [16]


lr_values          = [0.00001, 0.000001, 0.0000001]
weight_values      = [4]

HP_GRID = list(itertools.product(
    d_model_values, d_ff_values, N_values,
    num_heads_values, lr_values, weight_values))

# ----------------------------------------------------------------------------------
# 2.  S3 helper
# ----------------------------------------------------------------------------------
s3        = boto3.client("s3")
BUCKET    = get_config()["s3_bucket"]

def s3_upload(local_path, key):
    print(f"[S3] →  {key}")
    s3.upload_file(local_path, BUCKET, key)

# ----------------------------------------------------------------------------------
# 3.  Worker  (runs in a child process)
# ----------------------------------------------------------------------------------
def run_single_job(hp_tuple, gpu_id):
    """
    hp_tuple  = (d_model, d_ff, N, num_heads, lr, weight)
    gpu_id    = int, e.g. 0
    Returns metrics dict.
    """
    # ---  isolate this GPU  -------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_PORT"] = str(random.randint(12000, 20000))

    d_model, d_ff, N, num_heads, lr, weight = hp_tuple
    cfg = get_config()

    cfg.update(dict(
        d_model   = d_model,
        d_ff      = d_ff,
        N         = N,
        num_heads = num_heads,
        lr        = lr,
        weight    = weight,
    ))

    tag              = f"d{d_model}_ff{d_ff}_N{N}_h{num_heads}_lr{lr:.0e}_w{weight}"
    cfg["model_basename"] = f"ProdGPT_{tag}_{uuid.uuid4().hex[:6]}"

    # ---  launch training  --------------------------------------------------------
    metrics = train_model(cfg)

    # ---  save + upload  ----------------------------------------------------------
    out_fname = f"metrics_{tag}.json"
    with open(out_fname, "w") as f:
        json.dump(metrics, f, indent=2)

    # Upload checkpoint and metrics
    if metrics["best_checkpoint_path"] and os.path.exists(metrics["best_checkpoint_path"]):
        s3_upload(metrics["best_checkpoint_path"],
                  f"checkpoints/{os.path.basename(metrics['best_checkpoint_path'])}")
        os.remove(metrics["best_checkpoint_path"])

    s3_upload(out_fname, f"metrics/{out_fname}")
    os.remove(out_fname)

    return {**metrics,
            "tag": tag,
            "gpu": gpu_id}

# ----------------------------------------------------------------------------------
# 4.  Main launcher
# ----------------------------------------------------------------------------------
def main():
    gpu_list = [int(x) for x in os.environ.get("VISIBLE_GPUS", "0").split(",")]
    max_workers = len(gpu_list)
    if max_workers == 0:
        raise RuntimeError("Set VISIBLE_GPUS env var, e.g. VISIBLE_GPUS=0,1,2")

    print(f"Launching {len(HP_GRID)} jobs on GPUs {gpu_list}")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, hp in enumerate(HP_GRID):
            gpu_id = gpu_list[idx % max_workers]
            futures.append(pool.submit(run_single_job, hp, gpu_id))

        for fut in as_completed(futures):
            res = fut.result()     # will re‑raise exceptions if any
            print(f"[✓] {res['tag']}  (gpu={res['gpu']})  val_loss={res['val_loss']:.4f}")

if __name__ == "__main__":
    main()