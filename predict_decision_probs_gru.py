#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_decision_probs_gru.py

Loads a trained GRUClassifier and produces per‐timestep decision probabilities
from a JSON‐array file of records.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ──────────────────────────── 1.  Dataset ──────────────────────────────
class PredictDataset(Dataset):
    def __init__(self, json_path: Path, input_dim: int):
        raw = json.loads(json_path.read_text())
        if not isinstance(raw, list):
            raise ValueError("Input must be a JSON array of objects")
        self.rows = raw
        self.input_dim = input_dim

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        uid = rec["uid"][0] if isinstance(rec["uid"], list) else rec["uid"]
        feat_str = rec["AggregateInput"][0]
        flat = [0.0 if tok == "NA" else float(tok)
                for tok in feat_str.strip().split()]
        T = len(flat) // self.input_dim
        x = torch.tensor(flat, dtype=torch.float32).view(T, self.input_dim)
        return {"uid": uid, "x": x}

def collate(batch):
    uids = [b["uid"] for b in batch]
    xs   = [b["x"] for b in batch]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return {"uid": uids, "x": x_pad}

# ──────────────────────────── 2.  GRU model ─────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)  # (B, T, num_classes)

# ──────────────────────────── 3.  CLI ─────────────────────────────────
parser = argparse.ArgumentParser(
    description="Predict per‐timestep decision probabilities with GRU"
)
parser.add_argument("--data",        required=True,
                    help="Path to JSON array file of sequences")
parser.add_argument("--ckpt",        required=True,
                    help="Path to the .pt checkpoint")
parser.add_argument("--hidden_size", type=int, required=True,
                    help="Hidden size used at training time (must match ckpt)")
parser.add_argument("--input_dim",   type=int, default=15,
                    help="Feature dimension per timestep")
parser.add_argument("--batch_size",  type=int, default=128,
                    help="Batch size for inference")
parser.add_argument("--out",         required=True,
                    help="Output JSONL path")
args = parser.parse_args()

# ──────────────────────────── 4.  Setup ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUClassifier(
    input_dim   = args.input_dim,
    hidden_size = args.hidden_size,
    num_classes = 10
).to(device).eval()

# load checkpoint (handles both raw and wrapped state_dict)
state = torch.load(args.ckpt, map_location=device)
sd = state.get("model_state_dict", state)
model.load_state_dict(sd, strict=True)

dataset = PredictDataset(Path(args.data), input_dim=args.input_dim)
loader  = DataLoader(dataset,
                     batch_size=args.batch_size,
                     shuffle=False,
                     collate_fn=collate)

# ──────────────────────────── 5.  Inference ──────────────────────────
out_path = Path(args.out)
with out_path.open("w") as fout, torch.no_grad():
    for batch in tqdm(loader, desc="Predict"):
        x    = batch["x"].to(device)    # (B, T, D)
        uids = batch["uid"]

        logits = model(x)               # (B, T, 10)
        probs  = F.softmax(logits, dim=-1)  # (B, T, 10)

        # ─── NEW: drop the dummy logit 0 and renormalise ──────────────────
        probs = probs[..., 1:]                # keep columns 1-9  → (B,T,9)
        probs = probs / probs.sum(-1, keepdim=True)

        # Safe renorm
        row_sum = torch.where(row_sum == 0,
                            torch.ones_like(row_sum),
                            row_sum)
        probs   = probs / row_sum

        for i, uid in enumerate(uids):
            fout.write(json.dumps({
                "uid":   uid,
                "probs": probs[i].cpu().tolist()
            }) + "\n")

print(f"[✓] Predictions written → {out_path}")
