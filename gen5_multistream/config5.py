"""
Generation-5 (multi-stream state-space) configuration.

TWO PROFILES
------------
  get_config("pilot")  - sized for the 6 GB RTX PRO 500 laptop. Small model,
                         truncated event window, batch 2. For "does it learn
                         at all", not for results.
  get_config("hpcc")   - full sequences, larger model, for the cluster.

THE MEMORY CONSTRAINT (read before changing max_events)
-------------------------------------------------------
OfferInventoryCrossAttention builds a score tensor of shape

    (B, H, S, lto_len, S * obtained_len)

so cost grows with S**2, with a constant of lto_len*obtained_len*H = 160 at
H=4. For B=1, H=4, fp32, that single tensor is:

    S=256    42 MB        S=1024   671 MB
    S=512   168 MB        S=2048   2.7 GB

and autograd keeps roughly 3x that alive through the backward pass. At the
IPT cap of S=2048 this needs on the order of 11 GB for a single sequence,
which is why the laptop pilot truncates. Raising max_events is the single
biggest lever on memory here - much bigger than d_model or batch size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import paths as _paths

# ─────────────────────────── shared vocabulary ───────────────────────────
VOCAB_SRC = 68   # token id space: PAD, decisions, specials, products
VOCAB_TGT = 18   # projection width; decision labels live at 1..9

AI_RATE = 15     # 4 LTO + 10 obtained + 1 previous decision
LTO_LEN = 4
OBTAINED_LEN = 10
PREV_DEC_LEN = 1


def _base() -> Dict[str, Any]:
    return {
        # ---------- data ----------
        # Both profiles default to the IPT file. It has no LTO_* fields, but
        # dataset_multistream.py derives those streams from AggregateInput,
        # so this works on the old simple6 file too - just swap the name.
        "data_file": "clean_list_int_wide4_simple6_IPT.json",
        "vocab_size_src": VOCAB_SRC,
        "vocab_size_tgt": VOCAB_TGT,

        # ---------- event block layout ----------
        "ai_rate": AI_RATE,
        "lto_len": LTO_LEN,
        "obtained_len": OBTAINED_LEN,
        "prev_dec_len": PREV_DEC_LEN,

        # ---------- optimisation ----------
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "grad_clip": 1.0,
        "warmup_frac": 0.05,
        "gamma": 0.0,      # focal loss gamma; 0 = plain weighted cross-entropy
        "tau": 0.5,        # inverse-frequency class-weight temperature
        "patience": 5,
        "seed": 33,

        # ---------- splits ----------
        # "temporal": hold out later campaigns for every user, using the
        #   FeatureBasedHoldout / IndexBasedHoldout flags the R generator
        #   writes (train <= 27, val = 28, test >= 29). This is the holdout
        #   the dataset was designed for, and it is what makes the per-user
        #   embedding meaningful -- every user is seen during training.
        # "user": hold out whole users. Answers a different question ("can we
        #   predict a stranger?") and makes the user embedding useless at
        #   evaluation, since unseen users all fall back to index 0.
        # "both" (default): cross the two, giving three test cells --
        #   trained_users_future  a customer we trained on, later campaigns
        #   heldout_users_past    a customer we did not train on, same period
        #   heldout_users_future  neither: not trained on, later campaigns
        #   Held-out users share the index-0 unknown-user embedding, so the
        #   cells measure "no personal parameter" rather than random noise.
        #   NOTE these are not cold starts: a held-out user's history is still
        #   visible to the model as input, only their labels were withheld
        #   from training.
        "split_mode": "both",
        # Which campaign boundary separates calibration from holdout. The R
        # generator defines one per model family:
        #   "feature" -> FeatureBasedHoldout, campaigns >= 28
        #   "index"   -> IndexBasedHoldout,   campaigns >= 29
        # Gen 5 uses the product feature table, so "feature" is the matching
        # choice and campaign 28 belongs to the holdout period.
        "holdout_flag": "feature",
        # Out-of-sample customers, withheld from training entirely.
        # Lu & Kannan (JMR 2025) hold out 50%.
        "user_holdout_frac": 0.5,
        # Validation customers, taken from WITHIN the in-sample half and
        # scored on the calibration period only. Model selection therefore
        # never touches the holdout period.
        "val_user_frac": 0.1,
        # How validation is drawn (EXPERIMENTS.md R13).
        #   "customers": held-out customers x whole calibration period. The
        #                Lu & Kannan design -- but it shares a time regime with
        #                training, so it CANNOT see temporal overfitting: in R13
        #                it improved for 34 epochs while holdout NLL worsened.
        #   "late":      in-sample customers x the last calibration campaigns
        #                (>= val_from). Sits just before the holdout in time.
        # Default stays "customers" until the late mode is shown to track the
        # holdout; flip it once that is confirmed.
        "val_mode": "customers",
        # Campaign 27 alone is ~378k rows, about the size of the entire
        # holdout block (28-30, ~440k). Campaigns 25-27 would be 32% of ALL
        # rows and strip the most recent third of calibration from training.
        "val_from": 27,
        # Seeds the customer partition only. Kept separate from "seed" so
        # multi-seed runs share one partition.
        "split_seed": 33,
        # train_frac/val_frac apply to split_mode="user" only.
        "train_frac": 0.8,
        "val_frac": 0.1,

        # ---------- correctness ----------
        # Roll the obtained stream so event t sees o_{t-1} rather than o_t.
        # Without this the all-zero obtained block on inserted no-buy rows
        # determines y_t == 9 exactly, and the model reads the label off its
        # own input. Only set False to reproduce a pre-fix run.
        "shift_obtained": True,

        # ---------- augmentation ----------
        "augment_permute_obtained": False,

        # ---------- model ----------
        "use_user_embedding": True,
        # Lu & Kannan's heterogeneous mixture: H output projections combined
        # by per-customer weights alpha_n = softmax(user_mix_logits[n]).
        # 0 disables it and uses a single shared projection. Their application
        # uses a small H; start around 4-8. Out-of-sample customers receive
        # alpha_bar, the mean over trained customers.
        # Unlike a plain embedding, alpha_n is interpretable: it is a soft
        # membership over H behavioural patterns.
        "num_mix_heads": 0,
        "dropout": 0.1,
        # "transformer" (the gen-5 model), or "gru" / "lstm" baselines that
        # share its feature pipeline and differ only in the sequence encoder.
        "arch": "transformer",

        # ---------- output ----------
        "run_name": "gen5_multistream",
    }


def get_config(profile: str = "pilot") -> Dict[str, Any]:
    cfg = _base()

    if profile == "pilot":
        cfg.update({
            "profile": "pilot",
            # Model: deliberately small. The point is to exercise the wiring.
            "d_model": 64,
            "N": 2,
            "num_heads": 4,
            "d_ff": 128,
            # Memory: S=256 keeps the cross-attention score tensor near 42 MB
            # per sequence, which leaves room on a 6 GB card.
            "max_events": 256,
            "batch_size": 2,
            "grad_accum": 4,          # effective batch 8
            "num_epochs": 3,
            "amp": True,              # bf16 autocast where supported
            # Subsample users so an epoch is minutes, not hours.
            "max_users": 400,
            "num_workers": 0,         # >0 is unreliable on Windows
        })
    elif profile == "hpcc":
        cfg.update({
            "profile": "hpcc",
            "d_model": 128,
            "N": 4,
            "num_heads": 8,
            "d_ff": 384,
            # Still not the full 2048: see the memory note at the top of this
            # file. Raise deliberately, watching nvidia-smi.
            "max_events": 512,
            "batch_size": 4,
            "grad_accum": 4,
            "num_epochs": 60,
            "amp": True,
            "max_users": None,
            "num_workers": 4,
        })
    else:
        raise ValueError(f"unknown profile {profile!r}; use 'pilot' or 'hpcc'")

    return cfg


def data_path(cfg: Dict[str, Any]) -> Path:
    """Resolve the training file through PRODUCTGPT_DATA (see paths.py)."""
    return _paths.data_file(cfg["data_file"])


def feature_path() -> Path:
    return _paths.product_feature_xlsx()


def output_dir(cfg: Dict[str, Any]) -> Path:
    d = _paths.output_dir() / cfg["run_name"] / cfg["profile"]
    d.mkdir(parents=True, exist_ok=True)
    return d
