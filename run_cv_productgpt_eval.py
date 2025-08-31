#!/usr/bin/env python3
"""
run_cv_productgpt_eval.py

10-fold CV orchestrator that:
  • creates per-fold train/val/test UID sets
  • uploads fold UID lists to S3 (val/test) for exact matching
  • trains a model per fold via train_model(cfg_k) (your existing trainer)
  • runs predict_productgpt_and_eval.py per fold, passing the S3 UID files
  • stores eval CSVs/preds in S3 under .../eval/fold{K}/

Requirements:
  - Your repository must include train4_decoderonly_performer_feature_aws.train_model
  - predict_productgpt_and_eval.py must be reachable and executable
  - IAM role or AWS creds present on the machine
"""

from __future__ import annotations
import argparse, io, json, os, sys, time, subprocess, tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from config4 import get_config
from dataset4_productgpt import load_json_dataset

# Optional boto3 (falls back to aws CLI if missing)
try:
    import boto3
except Exception:
    boto3 = None

# ---- S3 helpers ----------------------------------------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Invalid S3 uri: {uri}"
    no = uri[5:]
    return (no.split("/", 1) + [""])[:2] if "/" in no else (no, "")

def s3_join(prefix: str, name: str) -> str:
    if not prefix.startswith("s3://"): raise ValueError(prefix)
    if not prefix.endswith("/"): prefix += "/"
    return prefix + name

def s3_join_folder(prefix: str, folder: str) -> str:
    folder = folder.strip("/")
    if not prefix.endswith("/"): prefix += "/"
    return prefix + folder + "/"

def s3_put_text(s3_uri: str, text: str):
    bucket, key = parse_s3_uri(s3_uri)
    if boto3:
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain")
    else:
        p = subprocess.run(["aws", "s3", "cp", "-", s3_uri], input=text.encode("utf-8"))
        if p.returncode != 0:
            raise RuntimeError(f"aws s3 cp to {s3_uri} failed")

def s3_download_to(s3_uri: str, local_path: str | Path):
    if boto3:
        bucket, key = parse_s3_uri(s3_uri)
        boto3.client("s3").download_file(bucket, key, str(local_path))
    else:
        rc = os.system(f"aws s3 cp '{s3_uri}' '{local_path}'")
        if rc != 0:
            raise RuntimeError(f"aws s3 cp from {s3_uri} failed")

# ---- CV utilities --------------------------------------------------
def _flat_uid(u) -> str:
    return str(u[0] if isinstance(u, list) else u)

def make_folds(uids: List[str], K: int, seed: int) -> List[List[str]]:
    rs = np.random.RandomState(seed)
    arr = np.array(sorted(set(uids)))
    rs.shuffle(arr)
    return [arr[i::K].tolist() for i in range(K)]

# ---- CLI -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--predict-eval-script", type=str, required=True,
                   help="Path to predict_productgpt_and_eval.py")
    p.add_argument("--labels", type=str, required=True,
                   help="JSON with labels (and AggregateInput if you also use it as --data)")
    p.add_argument("--data", type=str, default=None,
                   help="Data NDJSON/JSON for inference; default: cfg['filepath']")
    p.add_argument("--feat-xlsx", type=str, default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")
    p.add_argument("--ckpt-name", type=str, default=None,
                   help="If train_model returns an S3 path in 'ckpt', leave None. "
                        "If you want to override, supply a full local/s3 path template.")
    p.add_argument("--ai-rate", type=int, default=None, help="Override ai_rate for eval (else cfg)")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--s3-bucket", type=str, required=True)
    p.add_argument("--s3-prefix", type=str, required=True,
                   help="Base prefix, e.g. ProductGPT/CV/exp_001 (no s3://, bucket separate)")
    p.add_argument("--keep-local-ckpt", action="store_true", help="Do not delete downloaded ckpts from /tmp")
    return p.parse_args()

# ---- main ----------------------------------------------------------
def main():
    args = parse_args()
    cfg: Dict[str, Any] = get_config()

    # Paths & S3 layout
    bucket = args.s3_bucket
    root   = f"s3://{bucket}/{args.s3_prefix.strip('/')}/"
    s3_train_root = s3_join_folder(root, "train")
    s3_eval_root  = s3_join_folder(root, "eval")

    data_path   = args.data or cfg["filepath"]
    labels_path = args.labels

    # Load dataset once to enumerate users
    raw = load_json_dataset(data_path, keep_uids=None)
    all_uids = [_flat_uid(r["uid"]) for r in raw]
    folds = make_folds(all_uids, args.num_folds, args.seed)
    print(f"[INFO] CV with {args.num_folds} folds on {len(set(all_uids))} unique users.")
    print(f"[INFO] S3 train root: {s3_train_root}")
    print(f"[INFO] S3 eval  root: {s3_eval_root}")

    # Trainer (your existing pipeline)
    from train4_decoderonly_performer_feature_aws import train_model  # uses patched build_dataloaders

    all_rows = []
    t0 = time.time()

    for k in range(args.num_folds):
        test_u = set(folds[k])
        val_u  = set(folds[(k + 1) % args.num_folds])
        train_u = set(all_uids) - test_u - val_u

        print(f"\n========== FOLD {k+1}/{args.num_folds} ==========")
        print(f"[INFO] users: train={len(train_u)}  val={len(val_u)}  test={len(test_u)}")

        # --- 1) write & upload UID lists for exact-match reproducibility
        uid_val_txt  = "\n".join(sorted(val_u))  + "\n"
        uid_test_txt = "\n".join(sorted(test_u)) + "\n"

        s3_fold_train = s3_join_folder(s3_train_root, f"fold{k}")
        s3_val_uri  = s3_join(s3_fold_train, "uids_val.txt")
        s3_test_uri = s3_join(s3_fold_train, "uids_test.txt")
        s3_put_text(s3_val_uri, uid_val_txt)
        s3_put_text(s3_test_uri, uid_test_txt)
        print(f"[S3] uploaded UID lists: {s3_val_uri} , {s3_test_uri}")

        # --- 2) train model for this fold (pass explicit UID sets)
        cfg_k = dict(cfg)
        cfg_k.update({
            "fold_id": k,
            "uids_train": list(train_u),
            "uids_val":   list(val_u),
            "uids_test":  list(test_u),
            # (anything else your trainer expects can be set here)
        })
        summary = train_model(cfg_k)
        # expected keys (your trainer): 'ckpt' (s3 or local), maybe metrics
        ckpt_ref = summary.get("ckpt")
        if not ckpt_ref:
            raise RuntimeError("train_model did not return 'ckpt' in summary")

        # If checkpoint is on S3, download it locally for the eval script
        if ckpt_ref.startswith("s3://"):
            local_ckpt = Path(tempfile.gettempdir()) / f"fold{k}_model.pt"
            s3_download_to(ckpt_ref, local_ckpt)
            ckpt_local_path = str(local_ckpt)
        else:
            ckpt_local_path = ckpt_ref

        # --- 3) run prediction + evaluation with EXACT UID files
        s3_fold_eval = s3_join_folder(s3_eval_root, f"fold{k}")
        preds_local  = Path(tempfile.gettempdir()) / f"fold{k}_preds.jsonl.gz"

        # Build the eval command
        cmd = [
            sys.executable, args.predict_eval_script,
            "--data",   data_path,
            "--ckpt",   ckpt_local_path,
            "--labels", labels_path,
            "--feat-xlsx", args.feat_xlsx,
            "--s3",     s3_fold_eval,       # base eval folder; script nests again with --fold-id if you keep it
            "--pred-out", str(preds_local),
            "--uids-val",  s3_val_uri,
            "--uids-test", s3_test_uri,
            "--fold-id", str(k),
            "--batch-size", str(args.eval_batch_size),
            "--thresh", str(args.thresh),
        ]
        if args.ai_rate is not None:
            cmd += ["--ai-rate", str(args.ai_rate)]

        print("[CMD]", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"predict_productgpt_and_eval.py failed for fold {k} (rc={rc})")

        # optional: clean local ckpt
        if (not args.keep_local_ckpt) and ckpt_local_path.startswith("/tmp"):
            try: os.remove(ckpt_local_path)
            except Exception: pass

        # collect any quick numbers your trainer returned
        row = {
            "fold": k,
            "val_loss": summary.get("val_loss"),
            "val_f1": summary.get("val_f1"),
            "val_auprc": summary.get("val_auprc"),
            "test_f1": summary.get("test_f1"),
            "test_auprc": summary.get("test_auprc"),
            "ckpt": ckpt_ref,
            "val_uids_s3": s3_val_uri,
            "test_uids_s3": s3_test_uri,
        }
        all_rows.append(row)

    # ---- summary CSV (local + S3)
    df = pd.DataFrame(all_rows)
    print("\n=============  CV SUMMARY (training quick metrics)  ==========")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))
    print("==============================================================")

    # Upload consolidated training summary (eval CSVs were uploaded by the eval script)
    ts = time.strftime("%Y%m%d-%H%M%S")
    local_csv = Path("/tmp") / f"cv_train_summary_{ts}.csv"
    df.to_csv(local_csv, index=False)
    s3_summary = s3_join(s3_train_root, f"cv_train_summary_{ts}.csv")
    if boto3:
        bucket, key = parse_s3_uri(s3_summary)
        boto3.client("s3").put_object(Bucket=bucket, Key=key,
                                      Body=local_csv.read_bytes(), ContentType="text/csv")
    else:
        os.system(f"aws s3 cp '{local_csv}' '{s3_summary}'")
    print(f"[S3] uploaded: {s3_summary}")

    dur = time.time() - t0
    print(f"[DONE] {args.num_folds}-fold CV finished in {dur/60:.1f} min.")

if __name__ == "__main__":
    sys.exit(main())
