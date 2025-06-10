#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_decision_probs_lstm.py

Loads a trained LSTMClassifier and produces per‐timestep decision probabilities
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
    """
    Expects a single JSON array in `json_path`, where each element is a dict
    containing at least:
      - "uid": either a string or a single‐element list
      - "AggregateInput": a list whose first element is the whitespace-delimited feature string
    """
    def __init__(self, json_path: Path):
        raw = json.loads(json_path.read_text())
        if not isinstance(raw, list):
            raise ValueError("Input must be a JSON array of objects")
        self.rows = raw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        # handle uid being a list or str
        uid = rec["uid"][0] if isinstance(rec["uid"], list) else rec["uid"]
        # extract the feature string and convert to floats
        feat_str = rec["AggregateInput"][0]
        toks = [float(x) for x in feat_str.strip().split()]
        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.float32)}

def collate(batch):
    # batch is a list of {"uid":…, "x": tensor(T,D)}
    uids = [b["uid"] for b in batch]
    xs   = [b["x"] for b in batch]
    # pad sequences to [B, T_max, D]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return {"uid": uids, "x": x_pad}

# ──────────────────────────── 2.  Model ───────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int = 10):
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
parser.add_argument("--data",        required=True,
                    help="Path to JSON array file of sequences")
parser.add_argument("--ckpt",        required=True,
                    help="Path to the .pt checkpoint")
parser.add_argument("--hidden_size", type=int, required=True,
                    help="Hidden size used at training time")
parser.add_argument("--input_dim",   type=int, default=15,
                    help="Feature dimension per timestep")
parser.add_argument("--batch_size",  type=int, default=128,
                    help="Batch size for inference")
parser.add_argument("--out",         required=True,
                    help="Output JSONL path")
args = parser.parse_args()

# ──────────────────────────── 4.  Setup ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    input_dim   = args.input_dim,
    hidden_size = args.hidden_size,
    num_classes = 10
).to(device).eval()

# load checkpoint (supports either raw state_dict or {"model_state_dict":…})
state = torch.load(args.ckpt, map_location=device)
sd = state.get("model_state_dict", state)
model.load_state_dict(sd, strict=True)

dataset = PredictDataset(Path(args.data))
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

        for i, uid in enumerate(uids):
            # per‐timestep 10‐way probabilities
            prob_list = probs[i].cpu().tolist()
            fout.write(json.dumps({
                "uid":   uid,
                "probs": prob_list
            }) + "\n")

print(f"[✓] Predictions written → {out_path}")
