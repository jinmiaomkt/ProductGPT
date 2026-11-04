#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cv_lstm_eval.py

K-fold CV orchestrator for a raw-feature LSTM pipeline.

Per fold k:
  • create train/val/test UID sets (test = fold k, val = fold (k+1) mod K)
  • upload UID lists (val/test) to S3 for exact-match reproducibility
  • train an LSTM on the train set, early-stop on the val set
  • save best checkpoint locally (optionally upload to S3)
  • run the provided predict+eval script for that fold (points at the S3 UID files)
  • evaluation CSVs (+ optional predictions) land under .../eval/fold{k}/ on S3
  • print a compact per-fold training summary and write an overall CV summary CSV

Assumptions:
  - Training JSON is a list of records with:
      uid, AggregateInput[0] (space-separated features), Decision[0] (space-separated labels)
  - Labels JSON includes uid, Decision, IndexBasedHoldout, FeatureBasedHoldout.
  - predict-eval script matches a raw LSTM checkpoint signature (LSTM + fc),
    e.g., predict_lstm_and_eval_raw.py.

Example:
  python3 run_cv_lstm_eval.py \
    --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
    --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
    --predict-eval-script /home/ec2-user/ProductGPT/predict_lstm_and_eval_raw.py \
    --s3-bucket productgptbucket \
    --s3-prefix LSTM/CV/h128_lr0.0001_bs4 \
    --hidden-size 128 --lr 0.0001 --train-batch-size 4 --eval-batch-size 128 \
    --epochs 80 --class9-weight 5.0 --input-dim 15 \
    --num-folds 10 --seed 33 \
    --upload-ckpt --pred-out
"""

from __future__ import annotations
import argparse, json, os, sys, time, math, tempfile, subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ------------------------- S3 helpers -------------------------
try:
    import boto3  # optional; will fall back to aws CLI if unavailable
except Exception:
    boto3 = None

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://"), f"Invalid S3 uri: {uri}"
    no = uri[5:]
    return (no.split("/", 1) + [""])[:2] if "/" in no else (no, "")

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
        boto3.client("s3").put_object(Bucket=b, Key=k, Body=text.encode("utf-8"), ContentType="text/plain")
    else:
        p = subprocess.run(["aws", "s3", "cp", "-", s3_uri], input=text.encode("utf-8"))
        if p.returncode != 0:
            raise RuntimeError(f"aws s3 cp to {s3_uri} failed")

def s3_upload_file(local_path: str | Path, s3_uri: str):
    if boto3:
        b, k = parse_s3_uri(s3_uri)
        boto3.client("s3").upload_file(str(local_path), b, k)
    else:
        rc = os.system(f"aws s3 cp '{local_path}' '{s3_uri}'")
        if rc != 0:
            raise RuntimeError(f"aws s3 cp '{local_path}' '{s3_uri}' failed")

# ------------------------- UID & folds -------------------------
def _flat_uid(u) -> str:
    return str(u[0] if isinstance(u, list) else u)

def load_all_uids_from_labels(labels_path: str | Path) -> List[str]:
    obj = json.loads(Path(labels_path).read_text())
    if isinstance(obj, dict) and "uid" in obj:
        return [_flat_uid(u) for u in obj["uid"]]
    if isinstance(obj, list):
        return [_flat_uid(rec.get("uid")) for rec in obj]
    raise ValueError("Unrecognized labels JSON; expected dict['uid'] or list of records with 'uid'.")

def make_folds(uids: List[str], K: int, seed: int) -> List[List[str]]:
    rs = np.random.RandomState(seed)
    arr = np.array(sorted(set(uids)))
    rs.shuffle(arr)
    return [arr[i::K].tolist() for i in range(K)]

# ------------------------- Dataset -------------------------
class LSTMSequenceDataset(Dataset):
    """
    Reads training JSON (list of records) and filters by include_uids set.
    Each record requires:
      - 'uid'
      - 'AggregateInput': [ "t1 t2 ... tK" ] with K multiple of input_dim
      - 'Decision'      : [ "d1 d2 ... dT" ] aligned 1:1 with timesteps
    """
    def __init__(self, json_path: str | Path, input_dim: int, include_uids: Set[str]):
        rows = json.loads(Path(json_path).read_text())
        if not isinstance(rows, list):
            raise ValueError("Training JSON must be a list of records")
        self.input_dim = input_dim
        self.x, self.y = [], []
        for rec in rows:
            uid = _flat_uid(rec.get("uid"))
            if include_uids and uid not in include_uids:
                continue
            feat_str = rec["AggregateInput"][0] if isinstance(rec["AggregateInput"], list) else rec["AggregateInput"]
            dec_str  = rec["Decision"][0]       if isinstance(rec["Decision"],       list) else rec["Decision"]
            flat = [0.0 if t == "NA" else float(t) for t in str(feat_str).split()]
            dec  = [0   if t == "NA" else int(t)   for t in str(dec_str).split()]
            T = min(len(flat) // input_dim, len(dec))
            if T <= 0:
                continue
            xt = torch.tensor(flat[:T*input_dim], dtype=torch.float32).view(T, input_dim)
            yt = torch.tensor(dec[:T], dtype=torch.long)
            self.x.append(xt)
            self.y.append(yt)
        for xi, yi in zip(self.x, self.y):
            assert len(xi) == len(yi), "x/y length mismatch in dataset"

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def collate_pad(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return x_pad, y_pad

# ------------------------- Model & eval -------------------------
class LSTMDecisionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)      # (B, T, H)
        return self.fc(out)        # (B, T, C)

def _evaluate(loader, model, device, loss_fn) -> Tuple[float, float]:
    model.eval()
    tot_loss_mean = 0.0
    nll_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                     # (B, T, C)
            B, T, C = logits.shape
            flat_logits = logits.view(-1, C)
            flat_labels = yb.view(-1)
            loss = loss_fn(flat_logits, flat_labels)  # mean over tokens (ignores 0)
            tot_loss_mean += loss.item()
            logp = F.log_softmax(flat_logits, dim=-1)
            nll_sum += F.nll_loss(logp, flat_labels, ignore_index=0, reduction='sum').item()
            token_count += (flat_labels != 0).sum().item()
    avg_loss = tot_loss_mean / max(1, len(loader))
    avg_ppl  = math.exp(nll_sum / max(1, token_count)) if token_count > 0 else float("inf")
    return avg_loss, avg_ppl

def train_one_fold(
    data_json: str | Path,
    uids_train: Set[str],
    uids_val: Set[str],
    uids_test: Set[str],
    hidden_size: int,
    lr: float,
    input_dim: int,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    class9_weight: float,
    device: torch.device,
    ckpt_out: str | Path,
    patience_limit: int = 10,
) -> Dict[str, Any]:
    # datasets
    ds_tr = LSTMSequenceDataset(data_json, input_dim, uids_train)
    ds_va = LSTMSequenceDataset(data_json, input_dim, uids_val)
    ds_te = LSTMSequenceDataset(data_json, input_dim, uids_test)

    LD = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh, collate_fn=collate_pad)
    tr_ld = LD(ds_tr, train_batch_size, True)
    va_ld = LD(ds_va, eval_batch_size,  False)
    te_ld = LD(ds_te, eval_batch_size,  False)

    # model/opt
    model = LSTMDecisionModel(input_dim=input_dim, hidden_size=hidden_size, num_classes=10).to(device)
    w = torch.ones(10, device=device)
    w[9] = class9_weight   # class index 9 == decision "9" (since class 0 is PAD)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=0)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"val_loss": None, "val_ppl": None}
    patience = 0
    ckpt_out = Path(ckpt_out)

    for ep in range(1, epochs+1):
        model.train(); run_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).reshape(-1, 10)
            labels = yb.reshape(-1)
            loss   = loss_fn(logits, labels)
            optim.zero_grad(); loss.backward(); optim.step()
            run_loss += loss.item()

        v_loss, v_ppl = _evaluate(va_ld, model, device, loss_fn)
        print(f"[fold] Ep{ep:02d} Train={run_loss/len(tr_ld):.4f}  Val={v_loss:.4f}  PPL={v_ppl:.4f}")
        if best["val_loss"] is None or v_loss < best["val_loss"]:
            best.update(val_loss=v_loss, val_ppl=v_ppl)
            torch.save(model.state_dict(), ckpt_out)
            patience = 0
            print("  [*] new best saved")
        else:
            patience += 1
            if patience >= patience_limit:
                print("  [early-stop]")
                break

    # test with best
    model.load_state_dict(torch.load(ckpt_out, map_location=device))
    t_loss, t_ppl = _evaluate(te_ld, model, device, loss_fn)
    best.update(test_loss=t_loss, test_ppl=t_ppl)
    return best

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # data & evaluation
    p.add_argument("--labels", required=True, help="Labels JSON (used to derive UID universe)")
    p.add_argument("--data",   required=True, help="Training JSON (AggregateInput + Decision per user)")
    p.add_argument("--predict-eval-script", required=True, help="Path to predict_lstm_and_eval_raw.py (or compatible)")
    p.add_argument("--input-dim", type=int, default=15)

    # CV
    p.add_argument("--num-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=33)

    # LSTM hparams
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--class9-weight", type=float, default=5.0)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size",  type=int, default=128)

    # S3 layout
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--s3-prefix", required=True, help="Base prefix, e.g. LSTM/CV/h128_lr0.0001_bs4 (no s3://, bucket separate)")
    p.add_argument("--upload-ckpt", action="store_true", help="Also upload fold checkpoints to S3 train area")
    p.add_argument("--keep-local-ckpt", action="store_true", help="Do not delete /tmp fold checkpoints")

    # predict+eval knobs
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--pred-out", action="store_true", help="Save fold predictions and upload to S3 eval area")
    return p.parse_args()

# ------------------------- main -------------------------
def main():
    args = parse_args()

    # S3 roots
    root = f"s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}/"
    s3_train_root = s3_join_folder(root, "train")
    s3_eval_root  = s3_join_folder(root, "eval")

    # UID folds
    all_uids = load_all_uids_from_labels(args.labels)
    folds = make_folds(all_uids, args.num_folds, args.seed)
    print(f"[INFO] CV with {args.num_folds} folds on {len(set(all_uids))} unique users.")
    print(f"[INFO] S3 train root: {s3_train_root}")
    print(f"[INFO] S3 eval  root: {s3_eval_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    rows = []
    t0 = time.time()

    for k in range(args.num_folds):
        test_u = set(folds[k])
        val_u  = set(folds[(k + 1) % args.num_folds])
        train_u = set(all_uids) - test_u - val_u

        print(f"\n========== FOLD {k+1}/{args.num_folds} ==========")
        print(f"[INFO] users: train={len(train_u)}  val={len(val_u)}  test={len(test_u)}")

        # 1) upload fold UID lists for exact-match usage by predict script
        s3_fold_train = s3_join_folder(s3_train_root, f"fold{k}")
        s3_val_uri  = s3_join(s3_fold_train, "uids_val.txt")
        s3_test_uri = s3_join(s3_fold_train, "uids_test.txt")
        s3_put_text(s3_val_uri,  "\n".join(sorted(val_u))  + "\n")
        s3_put_text(s3_test_uri, "\n".join(sorted(test_u)) + "\n")
        print(f"[S3] uploaded UID lists: {s3_val_uri} , {s3_test_uri}")

        # 2) train LSTM on this fold
        ckpt_local = Path(tempfile.gettempdir()) / f"lstm_fold{k}_h{args.hidden_size}_lr{args.lr}_bs{args.train_batch_size}.pt"
        stats = train_one_fold(
            data_json=args.data,
            uids_train=train_u,
            uids_val=val_u,
            uids_test=test_u,
            hidden_size=args.hidden_size,
            lr=args.lr,
            input_dim=args.input_dim,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            class9_weight=args.class9_weight,
            device=device,
            ckpt_out=ckpt_local,
            patience_limit=10,
        )
        print(f"[INFO] best: val_loss={stats['val_loss']:.4f}  val_ppl={stats['val_ppl']:.4f}  "
              f"test_loss={stats['test_loss']:.4f}  test_ppl={stats['test_ppl']:.4f}")

        # optionally upload checkpoint
        if args.upload_ckpt:
            s3_fold_ckpt = s3_join(s3_fold_train, ckpt_local.name)
            s3_upload_file(ckpt_local, s3_fold_ckpt)
            print(f"[S3] uploaded checkpoint: {s3_fold_ckpt}")

        # 3) run predict + eval
        s3_fold_eval = s3_join_folder(s3_eval_root, f"fold{k}")
        preds_local = Path(tempfile.gettempdir()) / f"lstm_fold{k}_preds.jsonl.gz"
        cmd = [
            sys.executable, args.predict_eval_script,
            "--data", args.data,
            "--ckpt", str(ckpt_local),
            "--hidden-size", str(args.hidden_size),
            "--input-dim", str(args.input_dim),
            "--batch-size", str(args.eval_batch_size),
            "--labels", args.labels,
            "--s3",     s3_fold_eval,
            "--uids-val",  s3_val_uri,
            "--uids-test", s3_test_uri,
            "--fold-id",   str(k),
            "--thresh",    str(args.thresh),
        ]
        if args.pred_out:
            cmd.extend(["--pred-out", str(preds_local)])
        print("[CMD]", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"predict LSTM eval script failed for fold {k} (rc={rc})")

        # if we saved preds, push them to S3 eval root
        if args.pred_out:
            s3_preds = s3_join(s3_fold_eval, preds_local.name)
            s3_upload_file(preds_local, s3_preds)
            print(f"[S3] uploaded preds: {s3_preds}")

        # optionally clean local ckpt
        if (not args.keep_local_ckpt) and str(ckpt_local).startswith("/tmp"):
            try: os.remove(ckpt_local)
            except Exception: pass

        # add to CV summary
        rows.append({
            "fold": k,
            "val_loss": stats["val_loss"],
            "val_ppl":  stats["val_ppl"],
            "test_loss": stats["test_loss"],
            "test_ppl":  stats["test_ppl"],
            "val_uids_s3": s3_val_uri,
            "test_uids_s3": s3_test_uri,
        })

    # 4) CV summary CSV
    df = pd.DataFrame(rows)
    print("\n=============  CV SUMMARY (training quick metrics)  ==========")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))
    print("==============================================================")

    ts = time.strftime("%Y%m%d-%H%M%S")
    local_csv = Path("/tmp") / f"lstm_cv_train_summary_{ts}.csv"
    df.to_csv(local_csv, index=False)
    s3_summary = s3_join(s3_train_root, local_csv.name)
    s3_upload_file(local_csv, s3_summary)
    print(f"[S3] uploaded: {s3_summary}")

    dur = time.time() - t0
    print(f"[DONE] {args.num_folds}-fold CV finished in {dur/60:.1f} min.")

if __name__ == "__main__":
    sys.exit(main())
