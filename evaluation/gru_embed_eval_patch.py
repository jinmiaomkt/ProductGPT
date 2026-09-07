#!/usr/bin/env python3
"""
gru_embed_eval_patch.py

Adds GRU_embed adapter to unified_model_eval.py.

Add to unified_model_eval.py:
  1. Near the top:  from gru_embed_eval_patch import GRUEmbedAdapter
  2. In make_adapter:  if family == "gru_embed": return GRUEmbedAdapter(spec, args)

models.json entry example:
  {
    "name": "GRU-Embed",
    "model_family": "gru_embed",
    "ckpt": "/tmp/gru_embed_dm64_h256_lr0.001_bs16.pt",
    "feat_xlsx": "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx",
    "d_model": 64,
    "hidden_size": 256,
    "ai_rate": 15,
    "batch_size": 4
  }
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer, models, pre_tokenizers

# Reuse shared helpers from the eval script
from unified_model_eval import (
    BaseAdapter,
    flat_uid,
    FEATURE_COLS,
    FIRST_PROD_ID, LAST_PROD_ID,
)

# ── Constants ──
PAD_ID = 0
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
UNK_PROD_ID = 59
MAX_TOKEN_ID = 68
VOCAB_SIZE_SRC = 68
NUM_CLASSES = 10
SEQ_LEN_TGT = 1024

# ── Feature tensor ──
def load_feature_tensor(xls_path):
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)

# ── Tokenizers ──
def _build_tokenizer_src():
    vocab = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},
        "[SOS]": SOS_DEC_ID, "[EOS]": EOS_DEC_ID, "[UNK]": UNK_DEC_ID,
        **{str(i): i for i in range(13, UNK_PROD_ID + 1)},
    }
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ── Embedding (same as transformer) ──
class SpecialPlusFeatureLookup(nn.Module):
    def __init__(self, d_model, feature_tensor, product_ids, vocab_size_src):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_tensor.size(1)
        self.id_embed = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)
        self.register_buffer("feat_tbl", feature_tensor, persistent=False)
        prod_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        prod_mask[product_ids] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids):
        ids_long = ids.long()
        id_vec = self.id_embed(ids_long)
        raw_feat = self.feat_tbl[ids_long]
        feat_vec = self.feat_proj(raw_feat)
        keep = self.prod_mask[ids_long]
        feat_vec = feat_vec * keep.unsqueeze(-1)
        return id_vec + self.gamma * feat_vec

# ── GRU model (same as training script) ──
class GRUEmbedClassifier(nn.Module):
    def __init__(self, d_model, hidden_size, feature_tensor, ai_rate=15):
        super().__init__()
        self.ai_rate = ai_rate
        self.embed = SpecialPlusFeatureLookup(
            d_model=d_model,
            feature_tensor=feature_tensor,
            product_ids=list(range(FIRST_PROD_ID, LAST_PROD_ID + 1)) + [UNK_PROD_ID],
            vocab_size_src=VOCAB_SIZE_SRC,
        )
        self.gru = nn.GRU(d_model, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.gru(emb)
        logits = self.fc(out)
        pos = torch.arange(self.ai_rate - 1, x.size(1), self.ai_rate, device=x.device)
        return logits[:, pos, :]

# ── Dataset ──
class GRUEmbedPredictDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tok_src, seq_len_ai, pad_id=0):
        import json
        with open(data_path, "r") as f:
            raw = json.load(f)

        self.samples = []
        for rec in raw:
            uid = flat_uid(rec.get("uid", ""))
            agg = rec["AggregateInput"]
            src_txt = " ".join(map(str, agg)) if isinstance(agg, (list, tuple)) else str(agg)
            ids = tok_src.encode(src_txt).ids[:seq_len_ai]
            if len(ids) < seq_len_ai:
                ids = ids + [pad_id] * (seq_len_ai - len(ids))
            self.samples.append({"uid": uid, "x": torch.tensor(ids, dtype=torch.long)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def _collate(batch):
    uids = [b["uid"] for b in batch]
    x = torch.stack([b["x"] for b in batch])
    lens = [len(b["x"]) for b in batch]
    return {"uid": uids, "x": x, "lens": lens}

# ── Adapter ──
class GRUEmbedAdapter(BaseAdapter):
    def __init__(self, spec: Dict[str, Any], args):
        super().__init__(spec, args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = Path(spec["ckpt"])
        self.feat_path = Path(spec["feat_xlsx"])
        self.d_model = int(spec["d_model"])
        self.hidden_size = int(spec["hidden_size"])
        self.ai_rate = int(spec.get("ai_rate", 15))
        self.batch_size = int(spec.get("batch_size", 4))
        self.seq_len_ai = SEQ_LEN_TGT * self.ai_rate

        self.tok_src = _build_tokenizer_src()

        # Build and load model
        feat_tensor = load_feature_tensor(self.feat_path)
        self.model = GRUEmbedClassifier(
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            feature_tensor=feat_tensor,
            ai_rate=self.ai_rate,
        ).to(self.device).eval()

        state = torch.load(self.ckpt, map_location=self.device)
        # Handle both raw state_dict and wrapped checkpoint
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        else:
            sd = state
        clean_sd = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        self.model.load_state_dict(clean_sd, strict=True)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] {self.name}: loaded GRU-Embed model "
              f"(d_model={self.d_model}, hidden={self.hidden_size}, params={n_params:,})")

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        ds = GRUEmbedPredictDataset(
            self.args.data, self.tok_src, self.seq_len_ai, PAD_ID
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, collate_fn=_collate,
        )

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                uids = batch["uid"]
                lens = batch["lens"]

                logits = self.model(x)          # (B, n_decisions, 10)
                probs = F.softmax(logits[..., 1:], dim=-1)  # (B, n_decisions, 9)

                yield {
                    "uid": uids,
                    "lens": lens,
                    "probs_dec_9": probs.float().detach().cpu().numpy(),
                    "ai_rate": self.ai_rate,
                }