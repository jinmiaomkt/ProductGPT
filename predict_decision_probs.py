# predict_decision_probs.py
# This script expects you to run it from the command line like:
# python infer.py --data your_test_data.jsonl --ckpt path_to_model.pt --out output_preds.jsonl

# DecisionOnly_performer_nb_features16_dmodel64_ff64_N8_heads8_lr0.0001_weight2.pt

# ---------------------------------------------------------------
# 1.  python predict_decision_probs.py \
#         --data  path/to/your_dataset.jsonl \
#         --ckpt  DecisionOnly_<uid>.pt \
#         --out   decision_probs.jsonl
# ---------------------------------------------------------------

import argparse, json, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config4 import get_config
from model4_decoderonly_index_performer import build_transformer
from train4_decision_only_performer_aws import (_ensure_jsonl, JsonLineDataset, _build_tok)          # ← already defined

# ───────────────────────── arguments ───────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data", required=True,  help="ND-JSON file with your events")
p.add_argument("--ckpt", required=True,  help="*.pt checkpoint produced at training time")
p.add_argument("--out",  required=True,  help="where to write the jsonl predictions")
args = p.parse_args()

# ───────────────────── config – only 4 keys matter ─────────────
cfg = get_config()                     # loads the dict you trained with
# cfg["test-filepath"] = args.data            # points to **new** dataset
cfg["ai_rate"]  = 1                    # must match training
cfg["batch_size"] = 512                # inference can use big batches
TOK_PATH = Path(cfg["model_folder"]) / "tokenizer_tgt.json"

# ───────────────────── tokenizer (identical ordering!) ─────────
tok_tgt = Tokenizer.from_file(str(TOK_PATH)) if TOK_PATH.exists() else _build_tok()

pad_id   = tok_tgt.token_to_id("[PAD]")
special  = {pad_id,
            tok_tgt.token_to_id("[SOS]"),
            tok_tgt.token_to_id("[UNK]")}

# ───────────────────── dataset / loader (streaming) ────────────
# class PredictDataset(JsonLineDataset):
#     def __getitem__(self, idx):
#         row = super().__getitem__(idx)
#         # integerise the token string that feed the decoder
#         toks = [int(t) for t in row["AggregateInput"].split()]
#         return {"uid": row["uid"][0], "x": torch.tensor(toks, dtype=torch.long)}

class PredictDataset(JsonLineDataset):
    def __getitem__(self, idx):
        row = super().__getitem__(idx)

        # --- robust decoding of AggregateInput -----------------------------
        seq_raw = row["AggregateInput"]               # could be str, [str], or [int]
        if isinstance(seq_raw, list):
            if len(seq_raw) == 1 and isinstance(seq_raw[0], str):
                seq_str = seq_raw[0]                  # ["'10 20 30'"]  →  '10 20 30'
            else:                                     # [10, 20, 30]
                seq_str = " ".join(map(str, seq_raw)) # → '10 20 30'
        else:                                         # already a str
            seq_str = seq_raw

        toks = [int(t) for t in seq_str.strip().split()]
        # -------------------------------------------------------------------

        uid = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.long)}


def collate(batch):
    uids = [b["uid"] for b in batch]
    lens = [len(b["x"]) for b in batch]
    Lmax = max(lens)
    X = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
    for i, (seq, L) in enumerate(zip(batch, lens)):
        X[i, :L] = seq["x"]
    return {"uid": uids, "x": X}

datafile = _ensure_jsonl(cfg["filepath"])
loader   = DataLoader(PredictDataset(datafile),
                      batch_size=cfg["batch_size"],
                      collate_fn=collate,
                      shuffle=False)

# ───────────────────── model + checkpoint ──────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_transformer(
            vocab_size  = cfg["vocab_size_tgt"],
            # d_model     = cfg["d_model"],
            d_model     = 64, 
            # n_layers    = cfg["N"],
            n_layers    = 8, 
            # n_heads     = cfg["num_heads"],
            n_heads     = 8,
            # d_ff        = cfg["d_ff"],
            d_ff        = 64, 
            dropout     = 0.0,               # no dropout at test time
            max_seq_len = cfg["seq_len_ai"],
            # nb_features = cfg["nb_features"],
            nb_features = 16,
            kernel_type = cfg["kernel_type"]).to(device).eval()

state = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state["model_state_dict"], strict=True)

# ───────────────────── inference loop ──────────────────────────
out_path = Path(args.out).expanduser()
with out_path.open("w") as fout, torch.no_grad():
    for batch in loader:
        x   = batch["x"].to(device)                       # (B, T)
        uid = batch["uid"]

        logits = model(x)                                 # (B, T, V)
        prob   = torch.softmax(logits, dim=-1)            # (B, T, V)

        # positions containing **real decisions** – match training rule
        pos = torch.arange(cfg["ai_rate"]-1,
                           x.size(1),
                           cfg["ai_rate"],
                           device=device)                 # shape (n_slots,)

        prob_dec = prob[:, pos, :]                        # (B, n_slots, V)

        # serialise -- JSON-safe, one line per consumer
        for i, u in enumerate(uid):
            fout.write(json.dumps({
                "uid": u,
                "probs": prob_dec[i].cpu().tolist()       # nested list: [slot][vocab_id]
            }) + "\n")

print(f"predictions written → {out_path}")


