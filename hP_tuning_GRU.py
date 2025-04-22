#!/usr/bin/env python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------------
# 0) Hyperparameter grid & constants
# ------------------------------------------------------------------
HIDDEN_SIZES = [32, 64, 128]
LR_VALUES    = [1e-3, 1e-4, 1e-5]
BATCH_SIZES  = [2, 4]
HP_GRID      = list(itertools.product(HIDDEN_SIZES, LR_VALUES, BATCH_SIZES))

INPUT_DIM      = 15
NUM_CLASSES    = 10    # 0=PAD, 1–9 real decisions
EPOCHS         = 20
CLASS_9_WEIGHT = 5.0

JSON_PATH = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
S3_BUCKET = "productgptbucket"
S3_PREFIX = "GRU"

s3 = boto3.client("s3")

# ------------------------------------------------------------------
# 1) Dataset + collate
# ------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            records = json.load(f)

        self.x_seqs, self.y_seqs = [], []
        for row in records:
            flat = [0 if t == "NA" else int(t)
                    for t in row["AggregateInput"][0].split()]
            T    = len(flat) // INPUT_DIM
            x    = torch.tensor(flat, dtype=torch.float32).view(T, INPUT_DIM)

            dec  = [0 if t == "NA" else int(t)
                    for t in row["Decision"][0].split()]
            valid = min(T, len(dec)) - 1
            y    = torch.tensor(dec[1:valid+1], dtype=torch.long)

            self.x_seqs.append(x[:valid])
            self.y_seqs.append(y)

    def __len__(self):
        return len(self.x_seqs)

    def __getitem__(self, idx):
        return self.x_seqs[idx], self.y_seqs[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return x_pad, y_pad

# ------------------------------------------------------------------
# 2) GRU model
# ------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)         # (B, T, hidden_size)
        return self.fc(out)          # (B, T, NUM_CLASSES)

# ------------------------------------------------------------------
# 3) Validation-only evaluation
# ------------------------------------------------------------------
def evaluate(loader, model, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_ppl  = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x = x_batch.to(device)       # (B, T, 15)
            y = y_batch.to(device)       # (B, T)

            logits = model(x)            # (B, T, C)
            B, T, C = logits.shape
            flat_logits = logits.reshape(-1, C)
            flat_labels = y.reshape(-1)

            loss = loss_fn(flat_logits, flat_labels)
            total_loss += loss.item()

            probs = F.softmax(flat_logits, dim=-1)
            true_p = probs[torch.arange(len(flat_labels)), flat_labels]
            total_ppl += torch.exp(-torch.log(true_p + 1e-9).mean()).item()

            preds = probs.argmax(dim=-1).cpu().numpy()
            labs  = flat_labels.cpu().numpy()
            mask  = labs != 0   # drop PAD=0
            all_preds.append(preds[mask])
            all_labels.append(labs[mask])
            all_probs.append(probs.cpu().numpy()[mask, :])

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)

    avg_loss = total_loss / len(loader)
    avg_ppl  = total_ppl  / len(loader)

    cls_ids   = np.arange(1, NUM_CLASSES)
    conf_mat  = confusion_matrix(all_labels, all_preds, labels=cls_ids)
    hit_rate  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="macro")
    y_bin     = label_binarize(all_labels, classes=cls_ids)
    auprc     = average_precision_score(y_bin, all_probs[:,1:], average="macro")

    return avg_loss, conf_mat, avg_ppl, hit_rate, f1, auprc

# ------------------------------------------------------------------
# 4) Single-run: train on 80%, validate on 10%, print per-epoch
# ------------------------------------------------------------------
def run_one_experiment(params):
    hidden_size, lr, batch_size = params
    uid = f"h{hidden_size}_lr{lr}_bs{batch_size}"

    # split
    ds = SequenceDataset(JSON_PATH)
    n  = len(ds)
    train_size = int(0.8 * n)
    val_size   = int(0.1 * n)
    # test_size = n - train_size - val_size

    train_ds, val_ds, _ = random_split(ds, [train_size, val_size, n-train_size-val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GRUClassifier(hidden_size).to(device)

    weights = torch.ones(NUM_CLASSES, device=device)
    weights[9] = CLASS_9_WEIGHT
    loss_fn   = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).reshape(-1, NUM_CLASSES)
            labels = yb.reshape(-1)
            loss   = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # evaluate after this epoch
        val_loss, val_cm, val_ppl, val_hit, val_f1, val_auprc = evaluate(
            val_loader, model, device, loss_fn
        )

        # print metrics
        print(f"\n[{uid}] Epoch {epoch}")
        print(f"Train Loss={avg_train_loss:.4f}")
        print(f"Val   Loss={val_loss:.4f}")
        print(f"Val   PPL={val_ppl:.4f}")
        print(f"Val   Hit Rate={val_hit:.4f}")
        print(f"Val   F1 Score={val_f1:.4f}")
        print(f"Val   AUPRC={val_auprc:.4f}")
        print("Val Confusion Matrix:\n", val_cm)

    # final save & upload
    ckpt  = f"gru_{uid}.pt"
    torch.save(model.state_dict(), ckpt)
    metrics = {
        "hidden_size":  hidden_size,
        "lr":           lr,
        "batch_size":   batch_size,
        "val_loss":     val_loss,
        "val_ppl":      val_ppl,
        "val_hit_rate": val_hit,
        "val_f1_score": val_f1,
        "val_auprc":    val_auprc,
        "checkpoint":   ckpt
    }
    mfile = f"metrics_{uid}.json"
    with open(mfile, "w") as f:
        json.dump(metrics, f, indent=2)

    for local, key in [(ckpt, f"{S3_PREFIX}/{ckpt}"), (mfile, f"{S3_PREFIX}/{mfile}")]:
        s3.upload_file(local, S3_BUCKET, key)
        os.remove(local)

    return uid

# ------------------------------------------------------------------
# 5) Parallel sweep
# ------------------------------------------------------------------
def hyperparam_sweep_parallel(max_workers=None):
    if max_workers is None:
        max_workers = torch.cuda.device_count() or max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_experiment, p): p for p in HP_GRID}
        for fut in as_completed(futures):
            params = futures[fut]
            try:
                uid = fut.result()
                print(f"[Done] {uid}")
            except Exception as e:
                print(f"[Error] params={params} → {e}")

if __name__ == "__main__":
    hyperparam_sweep_parallel()
