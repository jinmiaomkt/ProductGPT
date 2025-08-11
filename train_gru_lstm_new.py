#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ONE GRU fold and upload metrics JSON to S3.

Usage (example):
python3 train_gru_lstm_new.py \
  --model gru --fold 2 --bucket productgptbucket --prefix CV_GRU \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --ckpt /tmp/ckpt_fold2.pt --out /tmp/out_fold2.txt \
  --hidden_size 128 --input_dim 15 --batch_size 4 --lr 1e-4 \
  --uids_trainval '["uid_a","uid_b"]' \
  --uids_test     '["uid_x","uid_y"]'
# or (recommended for large UID lists):
python3 train_gru_lstm_new.py \
  --model gru --fold 2 --bucket productgptbucket --prefix CV_GRU \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --ckpt /tmp/ckpt_fold2.pt --out /tmp/out_fold2.txt \
  --hidden_size 128 --input_dim 15 --batch_size 4 --lr 1e-4 \
  --uids_trainval_file /tmp/trainval_uids.json \
  --uids_test_file     /tmp/test_uids.json
"""
from __future__ import annotations

import argparse, json, math, os, random, pathlib
from typing import List, Dict, Optional, Tuple

import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

# ───────── Repro ─────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ───────── Dataset ───────────────────────────────────────────────────
NUM_CLASSES = 10        # 0 = PAD, 1-9 = real classes

class SequenceDataset(Dataset):
    def __init__(self, rows: List[Dict], input_dim: int,
                 allowed_uids: Optional[set] = None):
        self.x, self.y = [], []

        def _norm_uid_val(v):
            # rows may store uid as ["abc"] or "abc" or numeric
            if isinstance(v, (list, tuple)):
                v = v[0] if len(v) else ""
            return str(v)

        # Detect whether rows have a uid/UID key at all
        contains_uid = any(("uid" in r) or ("UID" in r) for r in rows)

        # If we were given a filter, normalize both sides to strings
        if allowed_uids is not None and contains_uid:
            norm_allowed = { _norm_uid_val(u) for u in allowed_uids }
            def _uid(r):
                return _norm_uid_val(r.get("uid", r.get("UID")))
            rows = [r for r in rows if _uid(r) in norm_allowed]
            if not rows:
                print("[WARN] No rows kept after UID filtering; falling back to empty dataset.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return x_pad, y_pad

# ───────── Model ─────────────────────────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, H)
        return self.fc(out)           # (B, T, C)

# ───────── Metrics / Eval ────────────────────────────────────────────
REV_VEC = torch.tensor([1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=torch.float32)

def _subset(pred, lbl, probs, rev_err, mask, classes=np.arange(1, 10)):
    if mask.sum() == 0:
        nan = float("nan")
        return dict(hit=nan, f1=nan, auprc=nan, rev_mae=nan)

    p, l, pr, re = pred[mask], lbl[mask], probs[mask], rev_err[mask]
    hit = accuracy_score(l, p)
    f1  = f1_score(l, p, average="macro")
    try:
        auprc = average_precision_score(
            label_binarize(l, classes=classes), pr[:, 1:10], average="macro")
    except ValueError:
        auprc = float("nan")
    return dict(hit=hit, f1=f1, auprc=auprc, rev_mae=re.mean())

@torch.no_grad()
def evaluate(loader, model, device, loss_fn):
    model.eval()
    P, L, PR, RE = [], [], [], []
    tot_loss_batch_mean = 0.0
    nll_sum = 0.0
    token_count = 0
    rev_vec = REV_VEC.to(device)

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                       # (B, T, C)
        B, T, C = logits.shape
        flat_logits = logits.reshape(-1, C)
        flat_labels = yb.reshape(-1)

        # Mean CE (ignore PAD=0) — for logging
        loss = loss_fn(flat_logits, flat_labels)
        tot_loss_batch_mean += loss.item()

        # Token-level NLL sum for proper perplexity
        log_probs = F.log_softmax(flat_logits, dim=-1)
        nll_sum += F.nll_loss(
            log_probs, flat_labels, ignore_index=0, reduction='sum'
        ).item()
        token_count += (flat_labels != 0).sum().item()

        # Other metrics on non-pad tokens
        probs = log_probs.exp()
        probs_np = probs.cpu().numpy()
        preds_np = probs_np.argmax(1)
        labs_np  = flat_labels.cpu().numpy()
        mask     = labs_np != 0

        P.append(preds_np[mask])
        L.append(labs_np[mask])
        PR.append(probs_np[mask])

        exp_rev  = (probs[:, 1:10] * rev_vec).sum(-1)
        true_rev = rev_vec[(flat_labels-1).clamp(min=0, max=8)]
        RE.append(torch.abs(exp_rev-true_rev).cpu().numpy()[mask])

    if len(P) == 0:
        # empty loader
        out = {"all": dict(hit=float("nan"), f1=float("nan"),
                           auprc=float("nan"), rev_mae=float("nan"))}
        avg_loss = float("nan")
        avg_ppl  = float("nan")
        return avg_loss, avg_ppl, out

    P, L, PR, RE = map(np.concatenate, (P, L, PR, RE))
    masks = dict(all=np.ones_like(P, dtype=bool))
    out = {k: _subset(P, L, PR, RE, m) for k, m in masks.items()}

    avg_loss = tot_loss_batch_mean / max(1, len(loader))
    avg_ppl  = math.exp(nll_sum / max(1, token_count))
    return avg_loss, avg_ppl, out

# ───────── Training loop ────────────────────────────────────────────
def train_one_gru_fold(*,
                       data_path: str,
                       input_dim: int,
                       hidden_size: int,
                       batch_size: int,
                       lr: float,
                       uids_trainval: Optional[List[str]],
                       uids_test: Optional[List[str]],
                       epochs: int = 80,
                       class_9_weight: float = 5.0,
                       seed: int = 33,
                       device: Optional[torch.device] = None
                       ) -> Tuple[Dict, Dict, nn.Module]:
    set_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load rows
    rows = json.loads(pathlib.Path(data_path).read_text())

    # Build datasets
    # Split train/val using uids_trainval if possible; otherwise random row split
    contains_uid = any(("uid" in r) or ("UID" in r) for r in rows)
    train_ds = val_ds = test_ds = None

    # ... after: uids_trainval, uids_test are loaded lists ...
    def _norm_uid_list(L):
        out = []
        for v in (L or []):
            if isinstance(v, (list, tuple)):
                v = v[0] if len(v) else ""
            out.append(str(v))
        return out

    uids_trainval = _norm_uid_list(uids_trainval)
    uids_test     = _norm_uid_list(uids_test)

    if uids_trainval and contains_uid:
        u_trainval = set(uids_trainval)
        u_test     = set(uids_test or [])
        # Further split trainval uids into train/val (90/10)
        u_trainval = list(u_trainval)
        u_trainval.sort()
        rng = random.Random(seed)
        rng.shuffle(u_trainval)
        k = max(1, int(0.9 * len(u_trainval)))
        u_train = set(u_trainval[:k])
        u_val   = set(u_trainval[k:])
        train_ds = SequenceDataset(rows, input_dim, allowed_uids=u_train)
        val_ds   = SequenceDataset(rows, input_dim, allowed_uids=u_val)
        test_ds  = SequenceDataset(rows, input_dim, allowed_uids=u_test)
    else:
        # Fallback: random split rows (80/10/10)
        print("[WARN] UID split not applied (no UID keys or no lists provided). "
              "Using random row split.")
        full = SequenceDataset(rows, input_dim, allowed_uids=None)
        n = len(full)
        tr_n, va_n = int(.8*n), int(.1*n)
        train_ds, val_ds, test_ds = random_split(
            full, [tr_n, va_n, n-tr_n-va_n],
            generator=torch.Generator().manual_seed(seed)
        )

    # DataLoaders
    LD = lambda d, sh: DataLoader(d, batch_size, shuffle=sh,
                                  collate_fn=collate_fn, num_workers=2,
                                  pin_memory=torch.cuda.is_available())
    tr_ld, va_ld, te_ld = LD(train_ds, True), LD(val_ds, False), LD(test_ds, False)

    # Model / loss / opt
    model = GRUClassifier(input_dim, hidden_size).to(dev)
    w = torch.ones(NUM_CLASSES, device=dev)
    w[9] = class_9_weight
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=0)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    # Train with early stopping on val loss
    best_loss, best_snapshot = None, None
    patience = 0
    PATIENCE_LIMIT = 10

    for ep in range(1, epochs+1):
        model.train(); run_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb).reshape(-1, NUM_CLASSES)
            labels = yb.reshape(-1)
            loss   = loss_fn(logits, labels)

            optim.zero_grad(); loss.backward(); optim.step()
            run_loss += loss.item()

        v_loss, v_ppl, v = evaluate(va_ld, model, dev, loss_fn)
        print(f"[GRU h{hidden_size} lr{lr} bs{batch_size}] "
              f"Ep{ep:02d} Train={run_loss/max(1,len(tr_ld)):.4f} "
              f"Val={v_loss:.4f} PPL={v_ppl:.4f}")

        if best_loss is None or v_loss < best_loss:
            best_loss, patience = v_loss, 0
            best_snapshot = {
                "state_dict": {k: v.cpu() if isinstance(v, torch.Tensor) else v
                               for k, v in model.state_dict().items()},
                "val": {"loss": v_loss, "ppl": v_ppl, "metrics": v}
            }
            print("  [*] new best saved (in memory)")
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print("  [early-stop]")
                break

    # Load best and evaluate on test
    if best_snapshot is not None:
        model.load_state_dict({k: v for k, v in best_snapshot["state_dict"].items()})
    t_loss, t_ppl, t = evaluate(te_ld, model, dev, loss_fn)

    val_metrics = {
        "val_loss": best_snapshot["val"]["loss"] if best_snapshot else float("nan"),
        "val_ppl":  best_snapshot["val"]["ppl"] if best_snapshot else float("nan"),
        **{f"val_{k}_{m}": best_snapshot["val"]["metrics"][k][m]
           for k in (best_snapshot["val"]["metrics"] if best_snapshot else {"all": {}})
           for m in (best_snapshot["val"]["metrics"][k] if best_snapshot else {})}
    }
    test_metrics = {
        "test_loss": t_loss, "test_ppl": t_ppl,
        **{f"test_{k}_{m}": t[k][m] for k in t for m in t[k]}
    }
    return val_metrics, test_metrics, model

# ───────── CLI / Main ───────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gru"])
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="CV_GRU")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=False, default="")
    ap.add_argument("--out",  required=False, default="")
    ap.add_argument("--hidden_size", required=True, type=int)
    ap.add_argument("--input_dim",   required=True, type=int)
    ap.add_argument("--batch_size",  required=True, type=int)
    ap.add_argument("--lr",          required=True, type=float)

    # Old inline JSON args (now optional)
    ap.add_argument("--uids_trainval", type=str, default=None,
                    help="JSON string of train/val UIDs")
    ap.add_argument("--uids_test",     type=str, default=None,
                    help="JSON string of test UIDs")

    # NEW: file-based args (preferred when provided)
    ap.add_argument("--uids_trainval_file", type=str, default=None,
                    help="Path to JSON file of train/val UIDs")
    ap.add_argument("--uids_test_file",     type=str, default=None,
                    help="Path to JSON file of test UIDs")

    args = ap.parse_args()

    # Resolve UID lists (prefer *_file if provided)
    if args.uids_trainval_file:
        with open(args.uids_trainval_file) as f:
            uids_trainval = json.load(f)
    else:
        uids_trainval = json.loads(args.uids_trainval) if args.uids_trainval else None

    if args.uids_test_file:
        with open(args.uids_test_file) as f:
            uids_test = json.load(f)
    else:
        uids_test = json.loads(args.uids_test) if args.uids_test else None

    # Basic checks
    if (uids_trainval is None) or (uids_test is None):
        raise SystemExit("Provide uids via --uids_* or --uids_*_file")

    val, test, model = train_one_gru_fold(
        data_path=args.data,
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        uids_trainval=uids_trainval,
        uids_test=uids_test
    )

    # Save checkpoint if requested
    if args.ckpt:
        ckpt_path = pathlib.Path(args.ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[ckpt] saved -> {ckpt_path}")

    # Build metrics dict
    metrics = {
        "fold": args.fold,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        **val, **test
    }

    # Write local metrics JSON and upload to S3
    name  = f"gru_h{args.hidden_size}_lr{args.lr}_bs{args.batch_size}_fold{args.fold}.json"
    local = pathlib.Path(name)
    local.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] wrote -> {local}")

    s3 = boto3.client("s3")
    s3.upload_file(str(local), args.bucket, f"{args.prefix}/metrics/{name}")
    print(f"[s3] uploaded -> s3://{args.bucket}/{args.prefix}/metrics/{name}")

if __name__ == "__main__":
    main()

#