#!/usr/bin/env python
"""
predict_decision_probs.py

Inference script for the “Decision-Only” performer model.
Writes a ND-JSON file identical to the input plus a ‘probs’ field
that contains ONLY the probabilities for decision classes 1-9.

DecisionOnly_dmodel128_ff128_N8_heads8_lr0.0001_weight2.pt
"""

import argparse, json, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config4 import get_config
from model4_decoderonly_index_performer_original import build_transformer
from train4_decision_only_performer_aws import (
        _ensure_jsonl, JsonLineDataset, _build_tok)

# ────────────────────────── CLI ──────────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--data", required=True, help="ND-JSON events file")
cli.add_argument("--ckpt", required=True, help="*.pt checkpoint")
cli.add_argument("--out",  required=True, help="output *.jsonl path")
args = cli.parse_args()

# ─────────────────────── config tweaks ───────────────────────
cfg = get_config()
cfg["ai_rate"]   = 1
cfg["batch_size"] = 512          # larger batches → faster inference

# ───────────────────────── tokenizer ─────────────────────────
tok_path = Path(cfg["model_folder"]) / "tokenizer_tgt.json"
tok_tgt  = (Tokenizer.from_file(str(tok_path))
            if tok_path.exists() else _build_tok())

pad_id  = tok_tgt.token_to_id("[PAD]")

# ───────────────────────── dataset ───────────────────────────
def to_int_or_pad(tok: str) -> int:
    try:
        return int(tok)
    except ValueError:
        return pad_id

class PredictDataset(JsonLineDataset):
    def __getitem__(self, idx):
        row = super().__getitem__(idx)

        seq_raw = row["PreviousDecision"]
        if isinstance(seq_raw, list):
            seq_str = (" ".join(map(str, seq_raw))
                       if not (len(seq_raw) == 1 and isinstance(seq_raw[0], str))
                       else seq_raw[0])
        else:
            seq_str = seq_raw
        toks = [to_int_or_pad(t) for t in seq_str.strip().split()]

        uid = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.long)}

def collate(batch):
    uids = [b["uid"] for b in batch]
    lens = [len(b["x"]) for b in batch]
    Lmax = max(lens)
    X = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
    for i, (item, L) in enumerate(zip(batch, lens)):
        X[i, :L] = item["x"]
    return {"uid": uids, "x": X}

datafile = _ensure_jsonl(args.data)
loader   = DataLoader(PredictDataset(datafile),
                      batch_size=cfg["batch_size"],
                      collate_fn=collate,
                      shuffle=False)

# ───────────────────────── model ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DecisionOnly_dmodel128_ff128_N8_heads8_lr0.0001_weight2.pt
model = build_transformer(
            vocab_size  = cfg["vocab_size_tgt"],
            d_model     = 128,
            n_layers    = 8,
            n_heads     = 8,
            d_ff        = 128,
            dropout     = 0.0,
            max_seq_len = cfg["seq_len_ai"],
            # nb_features = 16,
            kernel_type = cfg["kernel_type"]).to(device).eval()

state = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state["model_state_dict"], strict=True)

# indices we care about (decision classes 1-9)
focus_ids = torch.arange(1, 10, device=device)   # tensor([1,2,…,9])

# ─────────────────────── inference ───────────────────────────
out_path = Path(args.out).expanduser()
with out_path.open("w") as fout, torch.no_grad():
    for batch in loader:
        x   = batch["x"].to(device)          # (B, T)
        uids = batch["uid"]

        logits = model(x)                    # (B, T, V)
        probs  = torch.softmax(logits, dim=-1)

        # slots that correspond to real decisions
        pos = torch.arange(cfg["ai_rate"]-1,
                           x.size(1),
                           cfg["ai_rate"],
                           device=device)

        prob_dec       = probs[:, pos, :]            # (B, N, V)
        prob_dec_focus = prob_dec[..., focus_ids]    # (B, N, 9)

        # --- OPTIONAL: renormalise so row sums == 1 ----------------------
        # total = prob_dec_focus.sum(dim=-1, keepdim=True)
        # prob_dec_focus = torch.where(total > 0, prob_dec_focus / total, prob_dec_focus)
        # -----------------------------------------------------------------

        for i, uid in enumerate(uids):
            fout.write(json.dumps({
                "uid": uid,
                "probs": prob_dec_focus[i].cpu().tolist()   # [slot][decision 1-9]
            }) + "\n")

print(f"[✓] predictions written → {out_path}")
