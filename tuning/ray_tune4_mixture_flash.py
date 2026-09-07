#!/usr/bin/env python3
"""
ray_tune4_mixture_flash.py

Two-stage hyperparameter search for FlashAttention + Mixture-Head model.
  Stage A: Fast search on 30% data with ASHA early stopping
  Stage B: Retrain best config on 100% data with full epochs

Usage:
  screen -S mixture_tune
  python3 ray_tune4_mixture_flash.py
  # Ctrl+A, D to detach
"""
from __future__ import annotations

import os, json, socket
from pathlib import Path
import boto3
import torch

from config4 import get_config
from train4_mixture_flash_aws import train_model

import ray
from ray import tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

# ────────────────── Fold logic ──────────────────
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

# ────────────────── Trainable ──────────────────
def trainable_ray(config: dict):
    """Called once per trial by Ray Tune."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    base_cfg = get_config()
    cfg = dict(base_cfg)

    # Fixed settings
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
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]
    cfg["batch_size"] = config.get("batch_size", 4)

    # Architecture
    d_model, num_heads = config["dm_heads"]
    cfg["d_model"] = d_model
    cfg["num_heads"] = num_heads
    cfg["d_ff"] = int(d_model * config["dff_mult"])

    # Mixture-specific
    cfg["num_mixture_heads"] = config.get("num_mixture_heads", 4)

    # Optimization
    cfg.update({
        "nb_features": 0,
        "N": config.get("N", 6),
        "dropout": config.get("dropout", 0.2),
        "lr": config.get("lr", 5e-4),
        "weight": 1,
        "gamma": config.get("gamma", 0.0),
        "tau": config.get("tau", 0.3),
        "warmup_steps": config.get("warmup_steps", 500),
        "label_smoothing": config.get("label_smoothing", 0.0),
        "weight_decay": config.get("weight_decay", 0.01),
        "patience": config.get("patience", 8),
    })

    # Detect Ray Tune session
    try:
        trial_name = session.get_trial_name()
        in_tune = True
    except Exception:
        trial_name = "manual"
        in_tune = False

    cfg["model_basename"] = f"MixtureFlash_{trial_name}"

    def report_fn(m: dict):
        if in_tune:
            session.report(m)

    def stop_check_fn() -> bool:
        return False

    try:
        result = train_model(cfg, report_fn=report_fn if in_tune else None,
                             stop_check_fn=stop_check_fn)
    
        if in_tune and result:
            session.report({
                "epoch": result.get("fold_id", 0),
                "val_nll": result.get("best_val_nll", float("inf")),
                "val_hit": result.get("val_hit", 0),
                "val_f1_macro": result.get("val_f1_macro", 0),
        })
        # return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[WARN] OOM — skipping this trial")
        if in_tune:
            session.report({"epoch": 0, "val_nll": float("inf"),
                            "val_hit": 0.0, "val_f1_macro": 0.0})


def main():
    ray.init(ignore_reinit_error=True)

    # ── Valid (d_model, attention_heads) combos ──
    valid_dm_heads = []
    for dm in [64, 128, 256]:
        for h in [2, 4, 8]:
            if dm % h == 0 and (dm // h) >= 16:
                valid_dm_heads.append((dm, h))
    print(f"[INFO] Valid (d_model, heads) combos: {valid_dm_heads}")

    # ── Search space ──
    param_space = {
        # Stage A: fast search
        "num_epochs": 60,
        "data_frac": 0.15,

        # Architecture
        "dm_heads": tune.choice(valid_dm_heads),
        "N": tune.randint(4, 9),          # 4 to 8 layers
        "dff_mult": tune.choice([2, 3]),
        "dropout": tune.uniform(0.1, 0.4),
        "batch_size": tune.choice([4, 8]),

        # Mixture-specific
        "num_mixture_heads": tune.choice([2, 4, 6, 8]),

        # Optimization
        "lr": tune.loguniform(1e-4, 1e-3),
        "tau": tune.uniform(0.1, 0.5),
        "gamma": 0.0,
        "warmup_steps": tune.choice([500, 1000, 2000]),
        "label_smoothing": tune.uniform(0.0, 0.12),
        "weight_decay": tune.choice([0.01, 0.05]),
        "patience": 8,

        # Fixed
        "do_infer": False,
    }

    # ── ASHA: early stop bad trials ──
    asha = ASHAScheduler(
        time_attr="epoch",
        metric="val_nll",
        mode="min",
        max_t=60,
        grace_period=5,
        reduction_factor=4,
    )

    # ── HyperOpt: Bayesian search ──
    algo = HyperOptSearch(
        metric="val_nll",
        mode="min",
    )

    storage_path = str(Path("./ray_results").resolve())

    tuner = tune.Tuner(
        tune.with_resources(trainable_ray, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=50,
            max_concurrent_trials=1,
            search_alg=algo,
            scheduler=asha,
        ),
        run_config=ray.air.RunConfig(
            name="MixtureFlash_RayTune",
            storage_path=storage_path,
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    # ── Print best ──
    best = results.get_best_result(metric="val_nll", mode="min")
    print("\n" + "=" * 60)
    print("BEST CONFIG (Stage A)")
    print("=" * 60)
    print(json.dumps({k: v for k, v in best.config.items()
                      if k not in ("do_infer",)}, indent=2, default=str))
    print(f"val_nll:       {best.metrics.get('val_nll', 'N/A')}")
    print(f"val_hit:       {best.metrics.get('val_hit', 'N/A')}")
    print(f"val_f1_macro:  {best.metrics.get('val_f1_macro', 'N/A')}")

    # ── Stage B: retrain best on full data ──
    print("\n" + "=" * 60)
    print("STAGE B: RETRAIN BEST ON FULL DATA")
    print("=" * 60)

    best_cfg = best.config.copy()
    best_cfg.update({
        "data_frac": 1.0,
        "num_epochs": 200,
        "do_infer": True,
        "patience": 10,
    })

    trainable_ray(best_cfg)


if __name__ == "__main__":
    main()