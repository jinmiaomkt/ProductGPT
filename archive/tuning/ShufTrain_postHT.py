from __future__ import annotations

import argparse
import ast
import json
import os
import socket
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model

# -------------------- fold logic (same as Ray Tune script) --------------------
FOLD_ID = 0
SPEC_URI = "s3://productgptbucket/folds/productgptfolds.json"

def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(uri, "r") as f:
        return json.load(f)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _parse_dm_heads(x: Any) -> tuple[int, int]:
    """
    Accepts formats like:
      - "[96, 6]" (CSV string)
      - "(96, 6)" (string)
      - [96, 6] (already parsed)
      - (96, 6)
    """
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])

    if isinstance(x, str):
        y = ast.literal_eval(x)
        if isinstance(y, (list, tuple)) and len(y) == 2:
            return int(y[0]), int(y[1])

    raise ValueError(f"Cannot parse cfg_dm_heads={x!r}")


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {x!r}")


def _pick_row(csv_path: Path, rank: int) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "val_nll" not in df.columns:
        raise ValueError(f"'val_nll' not found in {csv_path}")
    df = df.sort_values("val_nll", ascending=True).reset_index(drop=True)

    if rank < 0 or rank >= len(df):
        raise IndexError(f"rank={rank} out of range (0..{len(df)-1})")

    return df.iloc[rank]


def build_phase_b_cfg_from_row(
    row: pd.Series,
    *,
    fold_id: int,
    uids_test: list[str],
    uids_trainval: list[str],
    num_epochs: int,
    data_frac: float,
    augment_train: bool,
    permute_repeat: int,
    do_infer: bool,
    batch_size: int | None,
    patience: int | None,
    upload_live_curve: bool,
    plot_every: int,
) -> dict:
    """
    Reconstruct a training config compatible with your train_model().
    """
    base_cfg = get_config()
    cfg = dict(base_cfg)

    # ---- fixed settings ----
    cfg.update({
        "mode": "train",
        "fold_id": fold_id,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval,
        "ai_rate": 15,
        "num_epochs": int(num_epochs),
        "data_frac": float(data_frac),
        "subsample_seed": 33,
        "augment_train": bool(augment_train),
        "permute_repeat": int(permute_repeat),
        "do_infer": bool(do_infer),

        # live curve plotting (requires train_model patch)
        "upload_live_curve": bool(upload_live_curve),
        "plot_every": int(plot_every),
    })

    if batch_size is not None:
        cfg["batch_size"] = int(batch_size)
    if patience is not None:
        cfg["patience"] = int(patience)

    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    # ---- best hyperparameters from Phase A ranking ----
    # Required columns from your phaseA_ranked_trials.csv:
    # cfg_N, cfg_nb_features, cfg_dm_heads, cfg_dropout, cfg_lr,
    # cfg_weight, cfg_warmup_steps, cfg_dff_mult
    d_model, num_heads = _parse_dm_heads(row["cfg_dm_heads"])
    dff_mult = int(row["cfg_dff_mult"]) if "cfg_dff_mult" in row.index else 2

    cfg["d_model"] = d_model
    cfg["num_heads"] = num_heads
    cfg["num_users"] = len(uids_trainval)
    cfg["d_ff"] = min(int(d_model * dff_mult), 512)

    cfg.update({
        "nb_features": int(row["cfg_nb_features"]),
        "N": int(row["cfg_N"]),
        "dropout": float(row["cfg_dropout"]),
        "lr": float(row["cfg_lr"]),
        "weight": float(row["cfg_weight"]),
        "warmup_steps": int(row["cfg_warmup_steps"]),

        # Defaults / optional values used by your trainer
        "gamma": float(row["cfg_gamma"]) if "cfg_gamma" in row.index and pd.notna(row["cfg_gamma"]) else 0.0,
        "label_smoothing": float(row["cfg_label_smoothing"]) if "cfg_label_smoothing" in row.index and pd.notna(row["cfg_label_smoothing"]) else 0.0,

        # Convenience aliases your trainer may read
        "dropout_attn": float(row["cfg_dropout"]),
        "dropout_ffn": float(row["cfg_dropout"]),
    })

    # Make this run easy to distinguish in checkpoints
    rank_tag = f"phaseB_rank{int(row.name)}"
    cfg["model_basename"] = f"MyProductGPT_{rank_tag}"

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Phase B retrain using best Phase-A config from CSV ranking.")
    ap.add_argument(
        "--phasea-csv",
        type=str,
        default="/home/ec2-user/ProductGPT/ray_results/ProductGPT_RayTune/phaseA_ranked_trials.csv",
        help="Path to phaseA_ranked_trials.csv",
    )
    ap.add_argument("--rank", type=int, default=0, help="Which ranked Phase-A config to retrain (0=best)")
    ap.add_argument("--epochs", type=int, default=200, help="Phase B num_epochs")
    ap.add_argument("--data-frac", type=float, default=1.0, help="Phase B data fraction (1.0 = full)")
    ap.add_argument("--augment-train", type=str, default="true", help="Turn on shuffling augmentation (true/false)")
    ap.add_argument("--permute-repeat", type=int, default=1, help="RepeatWithPermutation factor")
    ap.add_argument("--do-infer", type=str, default="true", help="Run final inference after training (true/false)")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch_size from get_config()")
    ap.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    ap.add_argument("--plot-every", type=int, default=1, help="Save live convergence plot every N epochs")
    ap.add_argument("--upload-live-curve", type=str, default="true", help="Upload live curve CSV/PNG to S3 each plot step")
    ap.add_argument("--dry-run", action="store_true", help="Print final config and exit")
    ap.add_argument("--smoke", action="store_true", help="Convenience: 3 epochs, no infer, data_frac=0.02")
    args = ap.parse_args()

    # ---- single-GPU DeepSpeed env (like your Ray trainable) ----
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    # Optional: keep the same runtime knobs you used elsewhere
    os.environ.setdefault("DS_BUILD_OPS", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ---- load folds ----
    spec = load_fold_spec(SPEC_URI)
    uids_test = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
    uids_trainval = [u for u in spec["assignment"] if u not in uids_test]
    assert uids_test and uids_trainval, "Empty split detected"

    # ---- pick ranked Phase-A row ----
    csv_path = Path(args.phasea_csv)
    row = _pick_row(csv_path, args.rank)

    # ---- apply smoke overrides if requested ----
    epochs = args.epochs
    data_frac = args.data_frac
    do_infer = _to_bool(args.do_infer)

    if args.smoke:
        epochs = 3
        data_frac = min(data_frac, 0.02)
        do_infer = False

    # ---- build cfg ----
    cfg = build_phase_b_cfg_from_row(
        row,
        fold_id=FOLD_ID,
        uids_test=uids_test,
        uids_trainval=uids_trainval,
        num_epochs=epochs,
        data_frac=data_frac,
        augment_train=_to_bool(args.augment_train),   # Phase B should be True
        permute_repeat=args.permute_repeat,
        do_infer=do_infer,
        batch_size=args.batch_size,
        patience=args.patience,
        upload_live_curve=_to_bool(args.upload_live_curve),
        plot_every=args.plot_every,
    )

    # ---- print summary ----
    print("\n===== SELECTED PHASE-A ROW =====")
    print(f"rank           : {args.rank}")
    print(f"trial_name      : {row.get('trial_name', 'N/A')}")
    print(f"val_nll         : {row.get('val_nll', 'N/A')}")
    print(f"epoch_at_best   : {row.get('epoch_at_best', 'N/A')}")
    print(f"val_hit         : {row.get('val_hit', 'N/A')}")
    print(f"val_f1_macro    : {row.get('val_f1_macro', 'N/A')}")
    print(f"val_auprc_macro : {row.get('val_auprc_macro', 'N/A')}")

    print("\n===== PHASE-B CONFIG (important fields) =====")
    keys_to_show = [
        "fold_id", "ai_rate", "seq_len_tgt", "seq_len_ai",
        "batch_size", "num_epochs", "patience",
        "data_frac", "augment_train", "permute_repeat", "do_infer",
        "d_model", "num_heads", "d_ff", "N", "nb_features",
        "dropout", "lr", "weight", "gamma", "warmup_steps",
        "upload_live_curve", "plot_every",
        "model_folder", "model_basename",
    ]
    for k in keys_to_show:
        if k in cfg:
            print(f"{k:>18}: {cfg[k]}")

    if args.dry_run:
        print("\n[dry-run] Exiting without training.")
        return

    print("\n===== START PHASE B RETRAIN =====")
    print("Note: Phase B uses augment_train=True to increase effective sample size via shuffling/permutation.")

    out = train_model(cfg)

    print("\n===== PHASE B DONE =====")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()