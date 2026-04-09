#!/usr/bin/env python3
"""
ray_tune4_flash_resume.py

Resumes the Flash attention Ray Tune sweep from where it left off.
- Does NOT repeat previously completed trials
- Adds regularization-focused hyperparameters
- HyperOpt uses previous results to guide the search
"""
from __future__ import annotations

import os, json, socket
from pathlib import Path
import boto3
import torch

from config4 import get_config
from train4_decoderonly_flash_feature_aws import train_model

import ray
from ray import tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

# -------------------- fold logic --------------------
FOLD_ID  = 0
SPEC_URI = "s3://productgptbucket/folds/productgptfolds.json"

def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(uri, "r") as f:
        return json.load(f)

spec = load_fold_spec(SPEC_URI)
uids_test     = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
uids_trainval = [u for u in spec["assignment"] if u not in uids_test]
assert uids_test and uids_trainval

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# -------------------- Trainable --------------------
def trainable_ray(config: dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    base_cfg = get_config()
    cfg = dict(base_cfg)

    cfg.update({
        "mode": "train",
        "fold_id": FOLD_ID,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval,
        "ai_rate": 15,
        "do_infer": config.get("do_infer", False),
        "num_epochs": config.get("num_epochs", 120),
        "data_frac": config.get("data_frac", 1.0),
        "subsample_seed": 33,
        "augment_train": config.get("augment_train", False),
        "permute_repeat": config.get("permute_repeat", 1),
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]
    cfg["batch_size"] = config.get("batch_size", cfg["batch_size"])

    d_model, num_heads = config["dm_heads"]
    cfg["d_model"] = d_model
    cfg["num_heads"] = num_heads
    cfg["d_ff"] = int(d_model * config["dff_mult"])

    # Pass weight_decay through to DeepSpeed
    cfg["weight_decay"] = config.get("weight_decay", 0.01)

    cfg.update({
        "nb_features": 0,
        "N": config.get("N", cfg["N"]),
        "dropout": config.get("dropout", cfg["dropout"]),
        "lr": config.get("lr", cfg["lr"]),
        "weight": 1,
        "gamma": config.get("gamma", 0.0),
        "tau": config.get("tau", 0.5),
        "warmup_steps": config.get("warmup_steps", cfg.get("warmup_steps", 500)),
        "label_smoothing": config.get("label_smoothing", 0.0),
    })

    try:
        trial_name = session.get_trial_name()
        in_tune = True
    except Exception:
        trial_name = "manual"
        in_tune = False

    cfg["model_basename"] = f"MyProductGPT_Flash_{trial_name}"

    def report_fn(m: dict):
        if in_tune:
            session.report(m)

    def stop_check_fn() -> bool:
        return False

    try:
        train_model(cfg, report_fn=report_fn if in_tune else None,
                    stop_check_fn=stop_check_fn)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[WARN] OOM — skipping this trial")
        if in_tune:
            session.report({"epoch": 0, "val_nll": float("inf"),
                            "val_hit": 0.0, "val_f1_macro": 0.0,
                            "val_auprc_macro": 0.0})

def main():
    ray.init(ignore_reinit_error=True)

    # ── Valid (d_model, heads) combos ──
    valid_dm_heads = []
    for dm in [64, 128, 256, 512]:
        for h in [2, 4, 8, 16]:
            if dm % h == 0 and (dm // h) >= 16:
                valid_dm_heads.append((dm, h))

    print(f"[INFO] Valid (d_model, heads) combos: {valid_dm_heads}")

    # ── Expanded search space with stronger regularization ──
    param_space = {
        # Stage-A tuning
        "num_epochs": 120,
        "data_frac": 0.3,
        "augment_train": tune.choice([False, True]),     # NEW: explore augmentation
        "permute_repeat": tune.choice([1, 3]),            # NEW: explore repetition

        # Architecture
        "dm_heads": tune.choice(valid_dm_heads),
        "N": tune.randint(2, 9),
        "dff_mult": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0.1, 0.5),               # WIDER: higher dropout
        "batch_size": tune.choice([4, 8, 16, 32]),

        # Optimization
        "lr": tune.loguniform(1e-5, 1e-3),
        "tau": tune.uniform(0.1, 0.7),                    # WIDER: lower tau explored
        "gamma": tune.choice([0.0, 1.0, 2.0]),
        "warmup_steps": tune.choice([500, 1000, 2000, 4000]),
        "label_smoothing": tune.uniform(0.0, 0.15),       # WIDER
        "weight_decay": tune.choice([0.01, 0.05, 0.1]),   # NEW

        # Fixed
        "weight": 1,
        "do_infer": False,
    }

    asha = ASHAScheduler(
        time_attr="epoch",
        metric="val_nll",
        mode="min",
        max_t=120,
        grace_period=10,
        reduction_factor=3,
    )

    algo = HyperOptSearch(
        metric="val_nll",
        mode="min",
    )

    storage_path = str(Path("./ray_results").resolve())

    tuner = tune.Tuner(
        tune.with_resources(trainable_ray, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=300,              # 300 NEW trials on top of previous 66
            max_concurrent_trials=1,
            search_alg=algo,
            scheduler=asha,
        ),
        run_config=ray.air.RunConfig(
            name="ProductGPT_Flash_RayTune",   # SAME name → resumes from existing
            storage_path=storage_path,
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    best = results.get_best_result(metric="val_nll", mode="min")
    print("\n===== BEST CONFIG (including previous + new trials) =====")
    print(best.config)
    print("best val_nll:", best.metrics.get("val_nll"))
    print("val_hit:", best.metrics.get("val_hit"))
    print("val_f1_macro:", best.metrics.get("val_f1_macro"))
    print("val_auprc_macro:", best.metrics.get("val_auprc_macro"))

    # ── Stage B ──
    best_cfg = best.config.copy()
    best_cfg.update({
        "data_frac": 1.0,
        "num_epochs": 200,
        "do_infer": True,
    })

    print("\n===== RETRAIN BEST ON FULL DATA =====")
    trainable_ray(best_cfg)

if __name__ == "__main__":
    main()