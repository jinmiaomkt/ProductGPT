#!/usr/bin/env python3
"""
Phase B: Retrain the best Flash attention config on full data.
Run with:  nohup python3 run_phase_b_flash.py > phase_b.log 2>&1 &
"""
from ray_tune4_flash import trainable_ray

best_cfg = {
    # ── Architecture (from best Phase A trial: trainable_ray_50f71d4f) ──
    "dm_heads": (128, 8),
    "N": 6,
    "dff_mult": 3,
    "dropout": 0.221486,

    # ── Optimization ──
    "lr": 0.00089497,
    "tau": 0.303938,
    "gamma": 0.0,
    "warmup_steps": 500,
    "batch_size": 4,
    "label_smoothing": 0.0930536,

    # ── Phase B: full data, longer training, run inference ──
    "data_frac": 1.0,
    "num_epochs": 200,
    "do_infer": True,
    "augment_train": False,
    "permute_repeat": 1,
}

if __name__ == "__main__":
    trainable_ray(best_cfg)