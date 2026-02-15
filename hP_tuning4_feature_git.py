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
