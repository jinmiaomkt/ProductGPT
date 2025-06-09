#!/usr/bin/env python
"""
predict_decision_probs_decisiononly.py

Inference for “Decision-Only” Performer model.
Outputs one JSON line per user with the 9-way decision distribution.

FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2.pt
"""

import argparse, json, torch, deepspeed          # ← add deepspeed import
from pathlib import Path
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config4 import get_config
from model4_decoderonly_index_performer import build_transformer
from train4_decision_only_performer_aws import (
        _ensure_jsonl, JsonLineDataset, _build_tok)

import torch

ckpt = torch.load("FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2.pt", map_location="cpu")
print(set(ckpt))                # what’s inside?  (‘epoch’, ‘best_val_loss’, ‘model_state_dict’, …)

state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
print(len(state), "tensor blobs saved")

# ───────────── CLI ─────────────
cli = argparse.ArgumentParser()
cli.add_argument("--data", required=True, help="ND-JSON events file")
cli.add_argument("--ckpt", required=True, help="*.pt checkpoint")
cli.add_argument("--out",  required=True, help="output *.jsonl path")
args = cli.parse_args()

# ─────────── Config ────────────
cfg              = get_config()
cfg["ai_rate"]   = 15
cfg["batch_size"] = 512

# ──────── Tokeniser ────────────
tok_path = Path(cfg["model_folder"]) / "tokenizer_tgt.json"
tok_tgt  = (Tokenizer.from_file(str(tok_path))
            if tok_path.exists() else _build_tok())
pad_id   = tok_tgt.token_to_id("[PAD]")

# ─────── Dataset helpers ───────
def to_int_or_pad(tok: str) -> int:
    try:             return int(tok)
    except ValueError:return pad_id

class PredictDataset(JsonLineDataset):
    def __getitem__(self, idx):
        row     = super().__getitem__(idx)
        seq_raw = row["AggregateInput"]

        if isinstance(seq_raw, list):
            seq_str = (" ".join(map(str, seq_raw))
                       if not (len(seq_raw)==1 and isinstance(seq_raw[0], str))
                       else seq_raw[0])
        else:  seq_str = seq_raw

        toks  = [to_int_or_pad(t) for t in seq_str.strip().split()]
        uid   = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.long)}

def collate(batch):
    uids = [b["uid"] for b in batch]
    lens = [len(b["x"]) for b in batch]
    Lmax = max(lens)
    X    = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
    for i,(item,L) in enumerate(zip(batch,lens)):
        X[i,:L] = item["x"]
    return {"uid": uids, "x": X}

loader = DataLoader(
            PredictDataset(_ensure_jsonl(args.data)),
            batch_size=cfg["batch_size"],
            collate_fn=collate,
            shuffle=False)

# ───────── Build model ──────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2
model  = build_transformer(
            vocab_size  = cfg["vocab_size_tgt"],
            d_model     = 32,
            n_layers    = 6,
            n_heads     = 4,
            d_ff        = 32,
            dropout     = 0.0,
            nb_features = 16,
            max_seq_len = cfg["seq_len_ai"],
            kernel_type = cfg["kernel_type"]).to(device).eval()

# ─────── LOAD CHECKPOINT ───────
# NB: PyTorch 2.6 default is now weights_only=True → we override it.
state = torch.load(args.ckpt, map_location=device, weights_only=False)

# ----- sanitise a DeepSpeed-QAT checkpoint -------------------
def clean_state_dict(raw):
    """Remove 'module.' prefix and QAT artefacts."""
    def strip_prefix(k):         # 1️⃣ remove 'module.'
        return k[7:] if k.startswith("module.") else k

    ignore = ("weight_fake_quant", "activation_post_process")
    return {strip_prefix(k): v   # 2️⃣ drop QAT junk
            for k, v in raw.items()
            if not any(tok in k for tok in ignore)}

# decide which blob inside the pickle is the actual state-dict
state_dict = (state["module"]
              if isinstance(state, dict) and "module" in state
              else state)
state_dict = clean_state_dict(state_dict)

model.load_state_dict(state_dict, strict=True)    # 3️⃣ now works

# ─────── LOAD CHECKPOINT (DeepSpeed + QAT compatible) ───────
state  = torch.load(args.ckpt, map_location=device, weights_only=False)

def clean_state_dict(raw):
    def strip_prefix(k): return k[7:] if k.startswith("module.") else k
    ignore = ("weight_fake_quant", "activation_post_process")
    return {strip_prefix(k): v
            for k, v in raw.items()
            if not any(tok in k for tok in ignore)}

state_dict = state["module"] if "module" in state else state
state_dict = clean_state_dict(state_dict)

model.load_state_dict(state_dict, strict=True)

# # tolerate either of these common layouts
# if "model_state_dict" in state:
#     model.load_state_dict(state["model_state_dict"], strict=True)
# elif "module" in state:                           # DS ZeRO engine save
#     model.load_state_dict(state["module"], strict=False)
# else:                                             # raw state-dict
#     model.load_state_dict(state, strict=True)

focus_ids = torch.arange(1, 10, device=device)  # decision classes 1-9

# ───────── Inference loop ───────
out_path = Path(args.out).expanduser()
with out_path.open("w") as fout, torch.no_grad():
    for batch in loader:
        x    = batch["x"].to(device)
        uids = batch["uid"]

        probs = torch.softmax(model(x), dim=-1)
        pos   = torch.arange(cfg["ai_rate"]-1, x.size(1), cfg["ai_rate"],
                             device=device)

        prob_dec_focus = probs[:, pos, :][..., focus_ids]  # (B, N, 9)

        for i, uid in enumerate(uids):
            fout.write(json.dumps({
                "uid": uid,
                "probs": prob_dec_focus[i].cpu().tolist()
            }) + "\n")

print(f"[✓] predictions written → {out_path}")
