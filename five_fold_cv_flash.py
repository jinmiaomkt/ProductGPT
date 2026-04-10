#!/usr/bin/env python3
"""
five_fold_cv_flash.py

Runs 5-fold cross-validation for the FlashAttention model.
Each fold:
  1. Trains on 4/5 of campaigns
  2. Evaluates on the held-out 1/5
  3. Saves metrics

At the end, reports mean ± std across all 5 folds.

Usage:
  nohup python3 five_fold_cv_flash.py > cv_flash.log 2>&1 &
"""
from __future__ import annotations

import json, os, socket, math
from pathlib import Path
import boto3
import numpy as np
import torch

from config4 import get_config
from train4_decoderonly_flash_feature_aws import train_model

# ═══════════════════════════════════════════════════════════
# Configuration — best hyperparameters from Phase A
# ═══════════════════════════════════════════════════════════
BEST_HP = {
    "dm_heads": (128, 8),
    "N": 6,
    "dff_mult": 3,
    "dropout": 0.221486,
    "lr": 0.00089497,
    "tau": 0.303938,
    "gamma": 0.0,
    "warmup_steps": 500,
    "batch_size": 4,
    "label_smoothing": 0.0930536,
}

NUM_FOLDS = 5
SPEC_URI = "s3://productgptbucket/folds/productgptfolds.json"

# ═══════════════════════════════════════════════════════════
# Fold spec
# ═══════════════════════════════════════════════════════════
def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(uri, "r") as f:
        return json.load(f)

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# ═══════════════════════════════════════════════════════════
# Train one fold
# ═══════════════════════════════════════════════════════════
def train_one_fold(fold_id: int, spec: dict) -> dict:
    """Train and evaluate on one fold. Returns test metrics."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_id} / {NUM_FOLDS - 1}")
    print(f"{'='*60}")

    # DeepSpeed env
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    uids_test = [u for u, f in spec["assignment"].items() if f == fold_id]
    uids_trainval = [u for u in spec["assignment"] if u not in uids_test]

    base_cfg = get_config()
    cfg = dict(base_cfg)

    # Fixed settings
    cfg.update({
        "mode": "train",
        "fold_id": fold_id,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval,
        "ai_rate": 15,
        "do_infer": False,
        "num_epochs": 200,
        "data_frac": 1.0,
        "subsample_seed": 33,
        "augment_train": False,
        "permute_repeat": 1,
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    # Best hyperparameters
    d_model, num_heads = BEST_HP["dm_heads"]
    cfg["d_model"] = d_model
    cfg["num_heads"] = num_heads
    cfg["d_ff"] = int(d_model * BEST_HP["dff_mult"])
    cfg["batch_size"] = BEST_HP["batch_size"]
    cfg["nb_features"] = 0

    cfg.update({
        "N": BEST_HP["N"],
        "dropout": BEST_HP["dropout"],
        "lr": BEST_HP["lr"],
        "weight": 1,
        "gamma": BEST_HP["gamma"],
        "tau": BEST_HP["tau"],
        "warmup_steps": BEST_HP["warmup_steps"],
        "label_smoothing": BEST_HP["label_smoothing"],
    })

    cfg["model_basename"] = f"MyProductGPT_Flash_CV_fold{fold_id}"

    result = train_model(cfg, report_fn=None, stop_check_fn=None)
    return result

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    spec = load_fold_spec(SPEC_URI)
    # spec = create_fold_spec_if_needed(spec, NUM_FOLDS)

    all_results = []

    for fold_id in range(NUM_FOLDS):
        try:
            result = train_one_fold(fold_id, spec)
            all_results.append(result)
            print(f"\n[FOLD {fold_id}] val_nll={result.get('best_val_nll', 'N/A'):.4f} "
                  f"test_hit={result.get('test_hit', 'N/A'):.4f} "
                  f"test_f1={result.get('test_f1_macro', 'N/A'):.4f}")
        except Exception as e:
            print(f"\n[FOLD {fold_id}] FAILED: {e}")
            all_results.append({"fold_id": fold_id, "error": str(e)})

        # Clear GPU memory between folds
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    metrics = ["best_val_nll", "val_hit", "val_f1_macro", "val_auprc_macro",
               "test_hit", "test_f1_macro", "test_auprc_macro"]

    # Print per-fold
    header = f"{'Fold':>6}"
    for m in metrics:
        header += f"  {m:>16}"
    print(header)
    print("-" * len(header))

    valid_results = [r for r in all_results if "error" not in r]
    for r in valid_results:
        line = f"{r.get('fold_id', '?'):>6}"
        for m in metrics:
            v = r.get(m, float("nan"))
            line += f"  {float(v):>16.4f}"
        print(line)

    # Print mean ± std
    print("-" * len(header))
    means_line = f"{'Mean':>6}"
    stds_line = f"{'Std':>6}"
    for m in metrics:
        vals = [float(r.get(m, float("nan"))) for r in valid_results]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            means_line += f"  {np.mean(vals):>16.4f}"
            stds_line += f"  {np.std(vals):>16.4f}"
        else:
            means_line += f"  {'N/A':>16}"
            stds_line += f"  {'N/A':>16}"
    print(means_line)
    print(stds_line)

    # Save summary
    summary = {
        "model": "FlashAttention",
        "config": BEST_HP,
        "num_folds": NUM_FOLDS,
        "per_fold": all_results,
        "mean": {},
        "std": {},
    }
    for m in metrics:
        vals = [float(r.get(m, float("nan"))) for r in valid_results]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            summary["mean"][m] = float(np.mean(vals))
            summary["std"][m] = float(np.std(vals))

    summary_path = Path("/tmp/flash_5fold_cv_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {summary_path}")

    # Upload to S3
    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(summary_path), "productgptbucket",
                       "FullProductGPT/flash/CV/5fold_summary.json")
        print("[S3] summary → s3://productgptbucket/FullProductGPT/flash/CV/5fold_summary.json")
    except Exception as e:
        print(f"[WARN] S3 upload failed: {e}")

if __name__ == "__main__":
    main()