#!/usr/bin/env python3
"""
run_cv_productgpt_eval.py

10-fold CV orchestrator for FullProductGPT that:
  • creates per-fold train/val/test UID sets
  • uploads fold UID lists to S3 (val/test) for exact matching
  • trains a model per fold only if a matching ckpt is NOT already on S3
  • runs predict_productgpt_and_eval.py per fold, passing the S3 UID files
  • stores eval CSVs/preds in S3 under .../eval/fold{K}/

Model spec defaults come from config4.get_config(), but we will
parse the spec from the checkpoint name whenever we run evaluation.
"""

from __future__ import annotations
import argparse, json, os, sys, time, subprocess, tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Project bits
from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model  # safe: we skip if S3 ckpt exists

# Optional boto3 (falls back to aws CLI if missing)
try:
    import boto3
except Exception:
    boto3 = None

import re

# ───────────────────────── hparam parsing from ckpt name ─────────────────────────
def _parse_hparams_from_ckpt_name(name: str):
    """
    Parse things like:
      FullProductGPT_featurebased_performerfeatures16_dmodel128_ff128_N6_heads4_lr0.0001_w2_fold0.pt
    Returns dict with any keys it finds; missing ones are left out.
    """
    m = re.search(r"features(\d+)_dmodel(\d+)_ff(\d+)_N(\d+)_heads(\d+)", name)
    if not m:
        return {}
    feats, dmodel, dff, N, heads = map(int, m.groups())
    return {
        "nb_features": feats,
        "d_model": dmodel,
        "d_ff": dff,
        "N": N,
        "num_heads": heads,
    }

# ───────────────────────── S3 helpers ─────────────────────────
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Invalid S3 uri: {uri}"
    no = uri[5:]
    if "/" in no:
        b, k = no.split("/", 1)
    else:
        b, k = no, ""
    return b, k

def s3_join(prefix: str, name: str) -> str:
    if not prefix.startswith("s3://"):
        raise ValueError(prefix)
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + name

def s3_join_folder(prefix: str, folder: str) -> str:
    if not prefix.endswith("/"):
        prefix += "/"
    folder = folder.strip("/")
    return prefix + folder + "/"

def s3_put_text(s3_uri: str, text: str):
    if boto3:
        b, k = parse_s3_uri(s3_uri)
        boto3.client("s3").put_object(
            Bucket=b, Key=k, Body=text.encode("utf-8"), ContentType="text/plain"
        )
    else:
        p = subprocess.run(["aws", "s3", "cp", "-", s3_uri], input=text.encode("utf-8"))
        if p.returncode != 0:
            raise RuntimeError(f"aws s3 cp to {s3_uri} failed")

def s3_exists(s3_uri: str) -> bool:
    if boto3:
        b, k = parse_s3_uri(s3_uri)
        s3 = boto3.client("s3")
        try:
            s3.head_object(Bucket=b, Key=k)
            return True
        except Exception:
            return False
    # best-effort fallback
    return os.system(f"aws s3 ls '{s3_uri}' >/dev/null 2>&1") == 0

def s3_download_to(s3_uri: str, local_path: str | Path):
    if boto3:
        b, k = parse_s3_uri(s3_uri)
        boto3.client("s3").download_file(b, k, str(local_path))
    else:
        rc = os.system(f"aws s3 cp '{s3_uri}' '{local_path}'")
        if rc != 0:
            raise RuntimeError(f"aws s3 cp from {s3_uri} failed")

# ──────────────────────── CV utilities ────────────────────────
def _flat_uid(u) -> str:
    # UID can be scalar or [scalar]
    if isinstance(u, list):
        return str(u[0])
    return str(u)

def load_all_uids_from_labels(labels_path: str | Path) -> List[str]:
    """
    Works with your clean_list_int_wide4_simple6.json format:
    - dict of arrays with key 'uid' (length = num users), or
    - list of records with key 'uid' (each scalar or [scalar])
    """
    obj = json.loads(Path(labels_path).read_text())
    if isinstance(obj, dict) and "uid" in obj:
        return [_flat_uid(u) for u in obj["uid"]]
    if isinstance(obj, list):
        out = []
        for rec in obj:
            out.append(_flat_uid(rec.get("uid")))
        return out
    raise ValueError("Unrecognized labels JSON structure; expected dict['uid'] or list of records with 'uid'].")

def make_folds(uids: List[str], K: int, seed: int) -> List[List[str]]:
    rs = np.random.RandomState(seed)
    arr = np.array(sorted(set(uids)))
    rs.shuffle(arr)
    return [arr[i::K].tolist() for i in range(K)]

# ───────────────────────── CLI ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=33)

    p.add_argument("--predict-eval-script", type=str, required=True,
                   help="Path to predict_productgpt_and_eval.py")
    p.add_argument("--labels", type=str, required=True,
                   help="JSON with labels (can also be used as --data for eval script)")
    p.add_argument("--data", type=str, default=None,
                   help="Data NDJSON/JSON for inference; default: --labels")
    p.add_argument("--feat-xlsx", type=str, default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")

    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--thresh", type=float, default=0.5)

    # Fixed model spec (FullProductGPT uses ai_rate)
    p.add_argument("--nb-features", type=int, default=16)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)

    p.add_argument("--ai-rate", type=int, default=15, help="Aggregate stride; seq_len_ai ≈ ai_rate * seq_len_tgt")

    p.add_argument("--s3-bucket", type=str, required=True)
    p.add_argument("--s3-prefix", type=str, required=True,
                   help="Base prefix, e.g. FullProductGPT/CV/exp_001 (no s3://, bucket separate)")

    p.add_argument("--keep-local-ckpt", action="store_true", help="Do not delete downloaded ckpts from /tmp")
    return p.parse_args()

# ───────────────────────── main ─────────────────────────
def main():
    args = parse_args()
    cfg: Dict[str, Any] = get_config()

    # Lock the model spec to the requested settings
    cfg.update({
        "nb_features": args.nb_features,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "N": args.N,
        "num_heads": args.num_heads,
    })

    cfg["ai_rate"] = args.ai_rate
    cfg["seq_len_ai"] = cfg.get("seq_len_ai", cfg["ai_rate"] * cfg["seq_len_tgt"])

    # Paths & S3 layout
    bucket = args.s3_bucket
    root   = f"s3://{bucket}/{args.s3_prefix.strip('/')}/"
    s3_train_root = s3_join_folder(root, "train")
    s3_eval_root  = s3_join_folder(root, "eval")

    data_path   = args.data or args.labels   # your eval script happily takes JSON here
    labels_path = args.labels

    # Load UIDs once from labels
    all_uids = load_all_uids_from_labels(labels_path)
    folds = make_folds(all_uids, args.num_folds, args.seed)
    uniq = len(set(all_uids))
    print(f"[INFO] CV with {args.num_folds} folds on {uniq} unique users.")
    print(f"[INFO] S3 train root: {s3_train_root}")
    print(f"[INFO] S3 eval  root: {s3_eval_root}")
    print(f"[INFO] Locked model spec → features={cfg['nb_features']} d_model={cfg['d_model']} "
          f"d_ff={cfg['d_ff']} N={cfg['N']} heads={cfg['num_heads']}  ai_rate={cfg['ai_rate']}  "
          f"seq_len_ai={cfg['seq_len_ai']}")

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

        # --- 2) either train (if no ckpt yet) or skip to eval (resume)
        cfg_k = dict(cfg)
        cfg_k.update({
            "mode": "train",
            "fold_id": k,
            "uids_train": list(train_u),
            "uids_val":   list(val_u),
            "uids_test":  list(test_u),
        })

        # Build deterministic basename that includes fold & spec
        basename = (f"FullProductGPT_featurebased_performer"
                    f"features{cfg_k['nb_features']}_dmodel{cfg_k['d_model']}"
                    f"_ff{cfg_k['d_ff']}_N{cfg_k['N']}_heads{cfg_k['num_heads']}"
                    f"_lr{cfg.get('lr', 1e-4)}_w{cfg.get('weight', 2)}_fold{k}")
        cfg_k["model_basename"] = basename

        # Expected S3 checkpoint location
        ckpt_name = f"{basename}.pt"
        ckpt_s3 = f"s3://{bucket}/FullProductGPT/performer/FeatureBased/checkpoints/{ckpt_name}"
        metrics_s3 = f"s3://{bucket}/FullProductGPT/performer/FeatureBased/metrics/{basename}.json"

        have_ckpt = s3_exists(ckpt_s3)
        if have_ckpt:
            print(f"[RESUME] Found existing ckpt on S3: {ckpt_s3}  → skipping training.")
            summary = {
                "ckpt": ckpt_s3,  # we’ll download locally for eval
                "val_loss": float("nan"),
                "val_f1": float("nan"),
                "val_auprc": float("nan"),
                "test_f1": float("nan"),
                "test_auprc": float("nan"),
            }
        else:
            print("[TRAIN] No checkpoint found; training this fold.")
            summary = train_model(cfg_k)
            # normalize to full S3 URI if needed
            if summary.get("ckpt", "").startswith("s3://") is False:
                summary["ckpt"] = ckpt_s3

        # --- 3) run prediction + evaluation with EXACT UID files
        s3_fold_eval = s3_join_folder(s3_eval_root, f"fold{k}")
        preds_local  = Path(tempfile.gettempdir()) / f"fold{k}_preds.jsonl.gz"

        # ensure local ckpt file
        if str(summary["ckpt"]).startswith("s3://"):
            local_ckpt = Path(tempfile.gettempdir()) / f"fold{k}_model.pt"
            try:
                s3_download_to(summary["ckpt"], local_ckpt)
            except Exception as e:
                # maybe summary only returned a filename; try the deterministic S3
                s3_download_to(ckpt_s3, local_ckpt)
            ckpt_local_path = str(local_ckpt)
        else:
            ckpt_local_path = str(summary["ckpt"])

        # parse hparams from the ckpt filename to pass to predict script
        hp = _parse_hparams_from_ckpt_name(Path(ckpt_local_path).name)

        cmd = [
            sys.executable, args.predict_eval_script,
            "--data",   data_path,
            "--ckpt",   ckpt_local_path,
            "--labels", labels_path,
            "--feat-xlsx", args.feat_xlsx,
            "--s3",     s3_fold_eval,
            "--pred-out", str(preds_local),
            "--uids-val",  s3_val_uri,
            "--uids-test", s3_test_uri,
            "--fold-id",   str(k),
            "--batch-size", str(args.eval_batch_size),
            "--thresh", str(args.thresh),
            "--ai-rate", str(cfg_k["ai_rate"]),
        ]

        # Append model size overrides if parse succeeded
        for key, flag in [
            ("nb_features", "--nb-features"),
            ("d_model", "--d-model"),
            ("d_ff", "--d-ff"),
            ("N", "--N"),
            ("num_heads", "--num-heads"),
        ]:
            if key in hp:
                cmd.extend([flag, str(hp[key])])

        print("[CMD]", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"predict_productgpt_and_eval.py failed for fold {k} (rc={rc})")

        # optional: clean local ckpt
        if (not args.keep_local_ckpt) and ckpt_local_path.startswith("/tmp"):
            try:
                os.remove(ckpt_local_path)
            except Exception:
                pass

        # collect quick numbers trainer returned (may be NaN if we skipped)
        row = {
            "fold": k,
            "val_loss": summary.get("val_loss"),
            "val_f1": summary.get("val_f1"),
            "val_auprc": summary.get("val_auprc"),
            "test_f1": summary.get("test_f1"),
            "test_auprc": summary.get("test_auprc"),
            "ckpt": summary.get("ckpt", ckpt_s3),
            "val_uids_s3": s3_val_uri,
            "test_uids_s3": s3_test_uri,
            "model_spec": f"feat{hp.get('nb_features', cfg_k['nb_features'])}"
                          f"_dm{hp.get('d_model', cfg_k['d_model'])}"
                          f"_ff{hp.get('d_ff', cfg_k['d_ff'])}"
                          f"_N{hp.get('N', cfg_k['N'])}"
                          f"_h{hp.get('num_heads', cfg_k['num_heads'])}"
                          f"_ai{cfg_k['ai_rate']}",
        }
        all_rows.append(row)

    # ---- summary CSV (local + S3/train)
    df = pd.DataFrame(all_rows)
    print("\n=============  CV SUMMARY (training quick metrics)  ==========")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))
    print("==============================================================")

    ts = time.strftime("%Y%m%d-%H%M%S")
    local_csv = Path("/tmp") / f"cv_train_summary_{ts}.csv"
    df.to_csv(local_csv, index=False)
    s3_summary = s3_join(s3_train_root, f"cv_train_summary_{ts}.csv")
    if boto3:
        b, k = parse_s3_uri(s3_summary)
        boto3.client("s3").put_object(Bucket=b, Key=k,
                                      Body=local_csv.read_bytes(), ContentType="text/csv")
    else:
        os.system(f"aws s3 cp '{local_csv}' '{s3_summary}'")
    print(f"[S3] uploaded: {s3_summary}")

    dur = time.time() - t0
    print(f"[DONE] {args.num_folds}-fold CV finished in {dur/60:.1f} min.")

if __name__ == "__main__":
    sys.exit(main())
