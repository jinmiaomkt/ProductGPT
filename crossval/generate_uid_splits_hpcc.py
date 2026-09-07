#!/usr/bin/env python3
"""Create validation/test UID files from a local OMEGA fold specification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic UID split files from productgptfolds.json."
    )
    parser.add_argument("--fold-spec", required=True, help="Local productgptfolds.json path")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be strictly between 0 and 1")

    fold_spec = Path(args.fold_spec).expanduser()
    if not fold_spec.is_file():
        raise FileNotFoundError(f"Fold specification not found: {fold_spec}")

    spec = json.loads(fold_spec.read_text())
    assignment = spec.get("assignment")
    if not isinstance(assignment, dict) or not assignment:
        raise ValueError("Fold specification must contain a non-empty 'assignment' object")

    available_folds = sorted({int(fold) for fold in assignment.values()})
    if args.fold_id not in available_folds:
        raise ValueError(
            f"fold {args.fold_id} is absent; available folds: {available_folds}"
        )

    # Preserve JSON insertion order to reproduce the original AWS script exactly.
    uids_test = [str(uid) for uid, fold in assignment.items() if int(fold) == args.fold_id]
    uids_trainval = [
        str(uid) for uid, fold in assignment.items() if int(fold) != args.fold_id
    ]

    rng = random.Random(args.seed)
    rng.shuffle(uids_trainval)
    n_val = max(1, int(args.val_fraction * len(uids_trainval)))
    uids_val = uids_trainval[:n_val]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = output_dir / "uids_val.txt"
    test_path = output_dir / "uids_test.txt"
    summary_path = output_dir / "split_summary.json"

    val_path.write_text("\n".join(uids_val) + "\n")
    test_path.write_text("\n".join(uids_test) + "\n")
    summary = {
        "fold_spec": str(fold_spec),
        "fold_id": args.fold_id,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "val_uids": len(uids_val),
        "test_uids": len(uids_test),
        "train_uids": len(uids_trainval) - len(uids_val),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Validation UIDs: {len(uids_val)} -> {val_path}")
    print(f"Test UIDs:       {len(uids_test)} -> {test_path}")
    print(f"Training UIDs:   {len(uids_trainval) - len(uids_val)}")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
