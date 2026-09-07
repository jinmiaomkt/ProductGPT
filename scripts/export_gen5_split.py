"""
Freeze the gen-5 train/val/test split to explicit UID files.

WHY
---
train5_multistream.py derives its split by shuffling record INDICES with a
fixed seed. That is deterministic for one particular file, but the partition
depends on the order of records inside the JSON. Regenerate the data, change
the cohort size, or pass --max-users, and the split silently shifts -- no
error, just a different test set.

That matters because the gen-5 LSTM and GRU baselines must hold out the same
users as the gen-5 transformer, or the comparison between them is not valid.

Running this script with the same --profile / --seed / --data-file as a
training run reproduces that run's split exactly and writes it down, so every
later model can read the same held-out users regardless of file order.

USAGE
    # Capture the split used by the current hpcc run (seed 33, IPT, no cap)
    python scripts/export_gen5_split.py --out eval_inputs/gen5

    # Then train against it
    python gen5_multistream/train5_multistream.py --profile hpcc \\
        --uids-dir eval_inputs/gen5

Writes uids_train.txt, uids_val.txt, uids_test.txt (one uid per line) plus
split_meta.json recording exactly how the split was produced.

Prints only counts and a checksum -- never uid values or record contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import config5
from dataset_multistream import TransformerDataset, load_json_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="hpcc", choices=["pilot", "hpcc"])
    ap.add_argument("--out", default="eval_inputs/gen5",
                    help="directory to write the uid files into")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg seed (default: the profile's seed)")
    ap.add_argument("--data-file", default=None,
                    help="override cfg data_file")
    ap.add_argument("--max-users", type=int, default=None,
                    help="replicate a run that used --max-users")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing uid files")
    args = ap.parse_args()

    cfg = config5.get_config(args.profile)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.data_file is not None:
        cfg["data_file"] = args.data_file
    max_users = args.max_users if args.max_users is not None else cfg.get("max_users")

    path = config5.data_path(cfg)
    print(f"[data] {path}")
    raw = load_json_dataset(str(path))
    print(f"[data] {len(raw)} users in file")

    # ---- EXACTLY the partition logic in train5_multistream.build_loaders ----
    rng = random.Random(cfg["seed"])
    idx = list(range(len(raw)))
    rng.shuffle(idx)
    if max_users:
        idx = idx[: int(max_users)]
        print(f"[data] subsampled to {len(idx)} users (max_users)")

    n = len(idx)
    n_tr = int(cfg["train_frac"] * n)
    n_va = int(cfg["val_frac"] * n)
    parts = {
        "train": idx[:n_tr],
        "val": idx[n_tr:n_tr + n_va],
        "test": idx[n_tr + n_va:],
    }
    # ------------------------------------------------------------------------

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in ("uids_train.txt", "uids_val.txt", "uids_test.txt")
                if (out_dir / p).exists()]
    if existing and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing split files in {out_dir}: {existing}\n"
            "A split that models have already trained against should not change "
            "silently. Pass --force if you really mean to replace it."
        )

    meta_counts = {}
    digest = hashlib.sha256()
    for name, ii in parts.items():
        uids = [TransformerDataset._uid(raw[i]) for i in ii]
        if len(set(uids)) != len(uids):
            print(f"[warn] {name}: {len(uids) - len(set(uids))} duplicate uid(s)")
        f = out_dir / f"uids_{name}.txt"
        f.write_text("\n".join(uids) + "\n", encoding="utf-8")
        meta_counts[name] = len(uids)
        # Hash the split membership so a later run can prove it matches,
        # without the hash revealing any uid.
        for u in sorted(uids):
            digest.update(u.encode("utf-8"))
            digest.update(b"|")
        digest.update(f"::{name}::".encode("utf-8"))
        print(f"[out] {f}  ({len(uids)} uids)")

    meta = {
        "profile": args.profile,
        "seed": cfg["seed"],
        "data_file": cfg["data_file"],
        "n_users_in_file": len(raw),
        "max_users": max_users,
        "train_frac": cfg["train_frac"],
        "val_frac": cfg["val_frac"],
        "counts": meta_counts,
        "split_sha256": digest.hexdigest(),
        "note": ("Reproduces train5_multistream.build_loaders' index shuffle. "
                 "Any model comparing against this split must read these files."),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[out] {out_dir/'split_meta.json'}")
    print(f"\nsplit sha256: {meta['split_sha256'][:16]}...")
    print(f"counts: {meta_counts}")


if __name__ == "__main__":
    main()
