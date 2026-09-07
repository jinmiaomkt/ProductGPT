"""
Re-evaluate a finished gen-5 checkpoint and report per-class metrics.

Use this to get the per-class breakdown for a run that finished BEFORE that
breakdown was added to the trainer -- no retraining needed. It loads best.pt,
rebuilds the model from the config stored inside it, reconstructs the same
test split, and reports precision / recall / F1 / AUPRC for each of the nine
decisions.

USAGE
    python scripts/eval_gen5_per_class.py \\
        --ckpt /storage/home/jinmiao/ProductGPT/runs/gen5_hpcc_S1024_b4/gen5_multistream/hpcc/best.pt

    # If you have frozen uid files, pass them so the split is guaranteed
    # identical rather than merely reproducible:
    python scripts/eval_gen5_per_class.py --ckpt .../best.pt --uids-dir eval_inputs/gen5

Writes per_class.json next to the checkpoint and prints a table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import config5
from shared.features import load_feature_tensor
from model_multistream_state_space import build_transformer
from train5_multistream import (
    amp_dtype,
    build_loaders,
    evaluate,
    format_per_class,
    pick_device,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--uids-dir", default=None,
                    help="frozen split directory (recommended)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default=None,
                    help="output json (default: per_class.json beside the ckpt)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    device = pick_device()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    num_users = state["num_users"]
    print(f"[ckpt] {ckpt_path}")
    print(f"[ckpt] epoch {state.get('epoch')}  val_nll {state.get('val_nll')}")
    print(f"[cfg]  d_model={cfg['d_model']} N={cfg['N']} heads={cfg['num_heads']} "
          f"max_events={cfg['max_events']} data={cfg['data_file']}")

    set_seed(cfg["seed"])
    uids_dir = Path(args.uids_dir) if args.uids_dir else None
    _, val_dl, test_dl, rebuilt_users = build_loaders(cfg, uids_dir=uids_dir)

    if rebuilt_users != num_users:
        print(f"[warn] user count differs: checkpoint {num_users}, rebuilt "
              f"{rebuilt_users}. The split may not match the training run; "
              f"pass --uids-dir to pin it.")

    feat = load_feature_tensor(config5.feature_path())
    model = build_transformer(
        vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
        max_seq_len=cfg["max_events"], d_model=cfg["d_model"], n_layers=cfg["N"],
        n_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        feature_tensor=feat, ai_rate=cfg["ai_rate"], num_users=num_users,
        lto_len=cfg["lto_len"], obtained_len=cfg["obtained_len"],
        prev_dec_len=cfg["prev_dec_len"],
        use_user_embedding=cfg["use_user_embedding"],
    ).to(device)

    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict mismatch -- missing={list(missing)[:5]} "
              f"unexpected={list(unexpected)[:5]}")
    else:
        print("[model] state_dict loaded cleanly (no missing/unexpected keys)")

    loader = test_dl if args.split == "test" else val_dl
    m = evaluate(model, loader, device, amp_dtype(device) if cfg.get("amp") else None)

    print(f"\n** {args.split.upper()} ** n={int(m['n']):,}  nll={m['nll']:.4f}  "
          f"hit={m['hit']:.4f}  macroF1={m['f1_macro']:.4f}  "
          f"macroAUPRC={m['auprc_macro']:.4f}  revMAE={m['rev_mae']:.4f}\n")
    print(format_per_class(m["per_class"]))

    out = Path(args.out) if args.out else ckpt_path.parent / f"per_class_{args.split}.json"
    out.write_text(json.dumps(
        {"checkpoint": str(ckpt_path), "split": args.split,
         "epoch": state.get("epoch"), "uids_dir": str(uids_dir) if uids_dir else None,
         **{k: v for k, v in m.items()}},
        indent=2), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
