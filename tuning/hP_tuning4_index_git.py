# hP_tuning4_index_git.py  – robust sweep runner
# ==============================================================
from __future__ import annotations

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import itertools, json, os, socket
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3, botocore, torch

from config4       import get_config
from train4_decoderonly_performer_index_aws  import train_model
import numpy as np

# ---------------- hyper-parameter grid -----------------------------------
nb_features_values = [16, 64, 128]
d_model_values    = [32, 64]
d_ff_values       = [32, 64]
N_values          = [6]
num_heads_values  = [4]
lr_values         = [1e-4]
weight_values     = [2, 4, 8]

HP_GRID = list(itertools.product(nb_features_values,
                                 d_model_values, 
                                 d_ff_values, 
                                 N_values,
                                 num_heads_values, 
                                 lr_values, 
                                 weight_values))

# ---------------- S3 helpers ---------------------------------------------
def get_s3():                # single client per process
    try:    return boto3.client("s3")
    except botocore.exceptions.BotoCoreError: return None

def s3_put(local, bucket, key, s3):
    if s3 is None: return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name} → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-WARN] {e}"); return False

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("",0)); return s.getsockname()[1]

# ---------------- one experiment -----------------------------------------
def run_one(params):
    nbf, dm, dff, N, H, lr, wt = params
    cfg = get_config()

    cfg["ai_rate"] = 15                          # <<<< FIX
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    cfg.update({"nb_features":nbf, 
                "d_model":dm, 
                "d_ff":dff, 
                "N":N,
                "num_heads":H, 
                "lr":lr, 
                "weight":wt})
    
    uid  = (f"indexbased_performerfeatures{cfg['nb_features']}_dmodel{dm}_ff{dff}_N{N}_heads{H}_lr{lr}_weight{wt}")
    stem = f"MyProductGPT_{uid}"
    cfg["model_basename"] = stem

    if torch.cuda.device_count():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hash(uid)%torch.cuda.device_count())
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())

    res = train_model(cfg)

    # ---------- JSON to disk (NumPy → list) -----------------------------
    # json_path = Path(f"{stem}.json")
    # with json_path.open("w") as fp:
    #     json.dump({k:(v.tolist() if isinstance(v,np.ndarray) else v)
    #                for k,v in res.items()}, fp, indent=2)

    json_path = Path(res["best_checkpoint_path"]).with_suffix(".json")
    if not json_path.exists():
        print(f"[WARN] Expected JSON not found: {json_path}")
    else:
        s3 = get_s3()
        bucket = cfg["s3_bucket"]

        ckpt = Path(res["best_checkpoint_path"])
        if ckpt.exists() and s3_put(ckpt, bucket,
            f"FullProductGPT/performer/Index/checkpoints/{ckpt.name}", s3):
            ckpt.unlink()

        if s3_put(json_path, bucket,
                f"FullProductGPT/performer/Index/metrics/{json_path.name}", s3):
            json_path.unlink()


    # ---------- S3 uploads ---------------------------------------------
    s3      = get_s3()
    bucket  = cfg["s3_bucket"]
    # prefix = cfg.get("s3_prefix","").rstrip("/")
    # if prefix: prefix += "/"

    ckpt = Path(res["best_checkpoint_path"])
    if ckpt.exists() and s3_put(ckpt, bucket,
        f"FullProductGPT/performer/Index/checkpoints/{ckpt.name}", s3):
        ckpt.unlink()

    if s3_put(json_path, bucket,
              f"FullProductGPT/performer/Index/metrics/{json_path.name}", s3):
        json_path.unlink()

    return uid

# ---------------- sweep driver -------------------------------------------
def sweep(max_workers=None):
    if max_workers is None:
        max_workers = torch.cuda.device_count() or max(1, mp.cpu_count()-1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one,hp): hp for hp in HP_GRID}
        for fut in as_completed(futs):
            hp = futs[fut]
            try:    print("[Done]", fut.result())
            except Exception as e:
                print(f"[Error] params={hp} → {e}")

if __name__ == "__main__":
    sweep()
