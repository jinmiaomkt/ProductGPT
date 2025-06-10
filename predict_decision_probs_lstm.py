#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_lstm.py

Loads a trained LSTMClassifier and produces per‐timestep decision probabilities.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ──────────────────────────── 1.  Dataset ──────────────────────────────
class PredictDataset(Dataset):
    """
    Expects an ND‐JSON file where each line is a dict with:
      - "uid": unique identifier
      - "AggregateInput": e.g. ["0 1 2 3 …"]
    """
    def __init__(self, jsonl_path: Path):
        self.rows = [json.loads(line) for line in jsonl_path.open()]
        self.pad_id = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        uid = rec["uid"]
        # parse the same way as training:
        tok_str = rec["AggregateInput"][0]
        toks = [int(t) if t != "NA" else 0 for t in tok_str.strip().split()]
        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.float32)}

def collate(batch):
    uids = [b["uid"] for b in batch]
    xs   = [b["x"] for b in batch]
    # pad with zeros
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)  # (B, T, INPUT_DIM)
    return {"uid": uids, "x": x_pad}

# ──────────────────────────── 2.  Model ───────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out)  # (B, T, num_classes)

# ──────────────────────────── 3.  CLI ─────────────────────────────────
parser = argparse.ArgumentParser(
    description="Predict per‐timestep decision probabilities with LSTM"
)
parser.add_argument("--data",       required=True,
                    help="Input ND‐JSON file of sequences")
parser.add_argument("--ckpt",       required=True,
                    help="Path to the .pt checkpoint")
parser.add_argument("--hidden_size",type=int, required=True,
                    help="Hidden size used at training time")
parser.add_argument("--input_dim",  type=int, default=15,
                    help="Feature dimension per timestep (default: 15)")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for inference")
parser.add_argument("--out",        required=True,
                    help="Output JSONL path")
args = parser.parse_args()

# ──────────────────────────── 4.  Setup ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build model & load checkpoint
model = LSTMClassifier(
    input_dim   = args.input_dim,
    hidden_size = args.hidden_size,
    num_classes = 10
).to(device).eval()

# load state dict
state = torch.load(args.ckpt, map_location=device)
# tolerate both raw and wrapped dicts
sd = state.get("model_state_dict", state)
model.load_state_dict(sd, strict=True)

# prepare data loader
dataset = PredictDataset(Path(args.data))
loader  = DataLoader(dataset,
                     batch_size=args.batch_size,
                     shuffle=False,
                     collate_fn=collate)

# ──────────────────────────── 5.  Inference ──────────────────────────
out_path = Path(args.out)
with out_path.open("w") as fout, torch.no_grad():
    for batch in tqdm(loader, desc="Predict"):
        x    = batch["x"].to(device)   # (B, T, D)
        uids = batch["uid"]

        logits = model(x)              # (B, T, 10)
        probs  = F.softmax(logits, dim=-1)  # (B, T, 10)

        # write one JSON line per sequence
        for i, uid in enumerate(uids):
            # convert to nested lists
            prof = probs[i].cpu().tolist()  # list of T lists of length 10
            fout.write(json.dumps({
                "uid":   uid,
                "probs": prof
            }) + "\n")

print(f"[✓] Written predictions to {out_path}")
