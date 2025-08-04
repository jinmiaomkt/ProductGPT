#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM hyper-parameter sweep
────────────────────────────────────────────────────────────
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# ───────── stdlib / third-party ─────────
import itertools, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
S3_PREFIX = "LSTM"           # top-level prefix for this model

LOCAL_TMP = Path("/home/ec2-user/tmp_lstm")
LOCAL_TMP.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

# ═════════ 1.  Dataset ══════════════════
class SequenceDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            rows = json.load(f)

        # self.x, self.y = [], []
        # for row in rows:
        #     flat = [0 if t == "NA" else int(t)
        #             for t in row["AggregateInput"][0].split()]
        #     T = len(flat) // INPUT_DIM
        #     x = torch.tensor(flat, dtype=torch.float32).view(T, INPUT_DIM)

        #     dec   = [0 if t == "NA" else int(t)
        #              for t in row["Decision"][0].split()]
        #     valid = min(T, len(dec)) - 1
        #     y = torch.tensor(dec[1:valid + 1], dtype=torch.long)

        #     self.x.append(x[:valid])
        #     self.y.append(y)


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

        # (optional) quick integrity check:
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

# ═════════ 2.  LSTM model ═══════════════
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, hidden_size,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)            # (B, T, C)

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
    # m_stop, m_after, m_tr = [], [], []
    tot_loss = tot_ppl = 0.0
    rev_vec = REV_VEC.to(device)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                       # (B, T, C)
            B, T, C = logits.shape
            flat_logits = logits.reshape(-1, C)
            flat_labels = yb.reshape(-1)

            loss = loss_fn(flat_logits, flat_labels)
            tot_loss += loss.item()

            probs = F.softmax(flat_logits, dim=-1)
            prob_true = probs[torch.arange(len(flat_labels)),
                               flat_labels.clamp(max=C-1)]
            tot_ppl += torch.exp(-torch.log(prob_true + 1e-9).mean()).item()

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

            labs2d = yb.cpu().numpy()
            # m_stop.append((labs2d == 9).reshape(-1)[mask])
            prev = np.pad(labs2d, ((0,0),(1,0)), constant_values=-1)[:, :-1]
            # m_after.append((prev == 9).reshape(-1)[mask])
            # m_tr.append(transition_mask(yb).cpu().numpy().reshape(-1)[mask])

    P, L, PR, RE = map(np.concatenate, (P, L, PR, RE))
    masks = dict(
        all=np.ones_like(P, dtype=bool),
        # stop_cur=np.concatenate(m_stop),
        # after_stop=np.concatenate(m_after),
        # trans=np.concatenate(m_tr),
    )
    out = {k: _subset(P, L, PR, RE, m) for k, m in masks.items()}
    avg_loss = tot_loss / len(loader)
    avg_ppl  = tot_ppl  / len(loader)
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
    model = LSTMClassifier(hidden).to(dev)

    w = torch.ones(NUM_CLASSES, device=dev)
    w[9] = CLASS_9_WEIGHT
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=0)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = LOCAL_TMP / f"lstm_{uid}.pt"
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
