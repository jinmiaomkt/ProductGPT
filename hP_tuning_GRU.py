# hP_tuning_GRU.py  ────────────────────────────────────────────────
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU hyper-parameter sweep
────────────────────────────────────────────────────────────
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# ───────── stdlib / third-party ─────────
import itertools, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import math

import boto3, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ═════════ 0.  Hyper-params ═════════════
HIDDEN_SIZES = [32, 64, 128]
LR_VALUES    = [1e-3, 1e-4, 1e-5]
BATCH_SIZES  = [2, 4]
HP_GRID      = list(itertools.product(HIDDEN_SIZES, LR_VALUES, BATCH_SIZES))

INPUT_DIM      = 15          # feature dim per timestep
NUM_CLASSES    = 10          # 0 = PAD, 1-9 = decisions
EPOCHS         = 80
CLASS_9_WEIGHT = 5.0

JSON_PATH = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
S3_BUCKET = "productgptbucket"
S3_PREFIX = "GRU"            # top-level prefix for this model

LOCAL_TMP = Path("/home/ec2-user/tmp_gru")
LOCAL_TMP.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

# ═════════ 1.  Dataset ══════════════════
class SequenceDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            rows = json.load(f)

        self.x, self.y = [], []
        for row in rows:
            # 1) Parse inputs → shape [T, INPUT_DIM]
            flat = [0 if t == "NA" else int(t)
                    for t in row["AggregateInput"][0].split()]
            T = len(flat) // INPUT_DIM
            x = torch.tensor(flat, dtype=torch.float32).view(T, INPUT_DIM)

            # 2) Parse decisions (already "next decision" per timestep)
            dec = [0 if t == "NA" else int(t)
                   for t in row["Decision"][0].split()]

            # 3) Align lengths WITHOUT any shift
            valid = min(T, len(dec))
            if valid == 0:
                continue  # skip empty sequences just in case

            y = torch.tensor(dec[:valid], dtype=torch.long)

            # 4) Store aligned slices: x[t] ↔ y[t]
            self.x.append(x[:valid])
            self.y.append(y)

        # (optional) integrity check
        for xi, yi in zip(self.x, self.y):
            assert len(xi) == len(yi)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return x_pad, y_pad

# ═════════ 2.  GRU model ════════════════
class GRUClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, H)
        return self.fc(out)           # (B, T, C)

# ═════════ 3.  Metric helpers ═══════════
REV_VEC = torch.tensor([1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=torch.float32)

def transition_mask(seq):
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev

def _json_safe(o):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)): return o.cpu().tolist()
    if isinstance(o, _np.ndarray):  return o.tolist()
    if isinstance(o, (_np.floating, _np.integer)):   return o.item()
    if isinstance(o, dict):   return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    return o

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

# ═════════ 4.  Evaluation ═══════════════
def evaluate(loader, model, device, loss_fn):
    model.eval()
    P, L, PR, RE = [], [], [], []
    tot_loss_batch_mean = 0.0          # keeps your printed "avg_loss" behavior
    nll_sum = 0.0                      # token-level sum NLL for PPL
    token_count = 0
    rev_vec = REV_VEC.to(device)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                       # (B, T, C)
            B, T, C = logits.shape
            flat_logits = logits.reshape(-1, C)
            flat_labels = yb.reshape(-1)

            # 1) Your normal (mean) loss for logging/early stop
            loss = loss_fn(flat_logits, flat_labels)  # ignore_index=0 inside
            tot_loss_batch_mean += loss.item()

            # 2) Proper PPL pieces: sum NLL over non-pad tokens
            log_probs = F.log_softmax(flat_logits, dim=-1)
            nll_sum += F.nll_loss(
                log_probs, flat_labels,
                ignore_index=0, reduction='sum'
            ).item()
            token_count += (flat_labels != 0).sum().item()

            # --- metrics (mask out pads) ---
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

    P, L, PR, RE = map(np.concatenate, (P, L, PR, RE))
    masks = dict(all=np.ones_like(P, dtype=bool))
    out = {k: _subset(P, L, PR, RE, m) for k, m in masks.items()}

    # Keep your displayed avg_loss semantics (mean over batches of the mean loss)
    avg_loss = tot_loss_batch_mean / len(loader)

    # True perplexity over all non-pad tokens
    avg_ppl = math.exp(nll_sum / max(1, token_count))

    return avg_loss, avg_ppl, out

# ═════════ 5.  One sweep job ════════════
def run_one(params):
    hidden, lr, bs = params
    uid = f"h{hidden}_lr{lr}_bs{bs}"

    # ----- data -----
    ds = SequenceDataset(JSON_PATH)
    n = len(ds)
    tr_n, va_n = int(.8*n), int(.1*n)
    tr, va, te = random_split(ds, [tr_n, va_n, n-tr_n-va_n])
    LD = lambda d, sh: DataLoader(d, bs, shuffle=sh, collate_fn=collate_fn)
    tr_ld, va_ld, te_ld = LD(tr, True), LD(va, False), LD(te, False)

    # ----- model & opt -----
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(hidden).to(dev)

    w = torch.ones(NUM_CLASSES, device=dev)
    w[9] = CLASS_9_WEIGHT
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=0)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = LOCAL_TMP / f"gru_{uid}.pt"
    json_path = LOCAL_TMP / f"metrics_{uid}.json"

    best_loss, best_metrics = None, {}
    patience = 0
    PATIENCE_LIMIT = 10

    # ----- training -----
    for ep in range(1, EPOCHS+1):
        model.train(); run_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb).reshape(-1, NUM_CLASSES)
            labels = yb.reshape(-1)
            loss   = loss_fn(logits, labels)

            optim.zero_grad(); loss.backward(); optim.step()
            run_loss += loss.item()

        v_loss, v_ppl, v = evaluate(va_ld, model, dev, loss_fn)
        print(f"[{uid}] Ep{ep:02d} Train={run_loss/len(tr_ld):.4f} "
              f"Val={v_loss:.4f} PPL={v_ppl:.4f}")

        if best_loss is None or v_loss < best_loss:
            best_loss, patience = v_loss, 0
            best_metrics = {"val_loss": v_loss, "val_ppl": v_ppl,
                            **{f"val_{k}_{m}": v[k][m]
                               for k in v for m in v[k]}}
            torch.save(model.state_dict(), ckpt_path)
            json_path.write_text(json.dumps(_json_safe(best_metrics), indent=2))
            print("  [*] new best saved")
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print("  [early-stop]")
                break

    # ----- test -----
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    t_loss, t_ppl, t = evaluate(te_ld, model, dev, loss_fn)
    metrics = {"hidden_size": hidden, "lr": lr, "batch_size": bs,
               **best_metrics,
               "test_loss": t_loss, "test_ppl": t_ppl,
               **{f"test_{k}_{m}": t[k][m] for k in t for m in t[k]}}
    json_path.write_text(json.dumps(_json_safe(metrics), indent=2))

    # ----- upload -----
    uploads = [
        (ckpt_path, f"{S3_PREFIX}/checkpoints/{ckpt_path.name}"),
        (json_path, f"{S3_PREFIX}/metrics/{json_path.name}"),
    ]
    for local, key in uploads:
        s3.upload_file(str(local), S3_BUCKET, key)
        local.unlink(missing_ok=True)
        print(f"[S3] {local.name} → s3://{S3_BUCKET}/{key}")

    return uid

# ═════════ 6.  Parallel sweep ═══════════
def sweep(max_workers=None):
    if max_workers is None:
        max_workers = torch.cuda.device_count() or mp.cpu_count()-1
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(run_one, p): p for p in HP_GRID}
        for f in as_completed(fut):
            p = fut[f]
            try:
                print(f"[Done] {f.result()}")
            except Exception as e:
                print(f"[Error] params={p} → {e}")

# ═════════ entry-point ═════════════════
if __name__ == "__main__":
    sweep()
