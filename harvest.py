#!/usr/bin/env python
# harvest_s3_models.py – run on EC2
# --------------------------------------------------------------
import os
import re
import json
import boto3
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from torch.utils.data import random_split, DataLoader
from dataset4_decoderonly import TransformerDataset, load_json_dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from config4git import get_config

# -----------------  Load training config  -----------------------
config = get_config()

# -----------------  S3 parameters  -------------------------------
BUCKET   = config.get("s3_bucket", "productgptbucket")
PREFIX   = config.get("s3_prefix", "winningmodel/")
LOCALDIR = Path(config.get("checkpoint_dir", "/home/ec2-user/ProductGPT/checkpoints"))
LOCALDIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

# -----------------  Download .pt files from S3  -------------------
resp  = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
ckpts = [obj["Key"] for obj in resp.get("Contents", []) if obj.get("Key", "").endswith(".pt")]
if not ckpts:
    raise RuntimeError("No .pt files found under that prefix!")
for key in ckpts:
    dest = LOCALDIR / Path(key).name
    if not dest.exists():
        print(f"▶ downloading s3://{BUCKET}/{key} → {dest}")
        s3.download_file(BUCKET, key, str(dest))
    else:
        print(f"✓ already have {dest}")

# -----------------  Dataloader setup  ----------------------------
RAW_JSON    = config["filepath"]
SEQ_LEN_AI  = config["seq_len_ai"]
SEQ_LEN_TGT = config["seq_len_tgt"]
NUM_HEADS   = config["num_heads"]
AI_RATE     = config["ai_rate"]
BATCH_SIZE  = config["batch_size"]
SEED        = config.get("seed", 33)

# Build fixed-vocab tokenizers
def build_tokenizer_fixed():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab_size = config.get("vocab_size_src")
    vocab = {str(i): i for i in range(vocab_size)}
    vocab.update({"[PAD]":0, "[SOS]":10, "[EOS]":11, "[UNK]":12})
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

tokenizer_ai  = build_tokenizer_fixed()
tokenizer_tgt = build_tokenizer_fixed()

# Load and split full dataset into train/val/test
full_data = load_json_dataset(RAW_JSON)
train_sz  = int(0.8 * len(full_data))
val_sz    = int(0.1 * len(full_data))
test_sz   = len(full_data) - train_sz - val_sz

torch.manual_seed(SEED)
_, val_data, _ = random_split(
    full_data, [train_sz, val_sz, test_sz],
    generator=torch.Generator().manual_seed(SEED)
)

val_ds = TransformerDataset(
    val_data, tokenizer_ai, tokenizer_tgt,
    SEQ_LEN_AI, SEQ_LEN_TGT, NUM_HEADS, AI_RATE
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------  Load feature_tensor  -------------------------
df = pd.read_excel(config["feature_file"], sheet_name=0)
feature_cols    = config["feature_cols"]
FIRST_PROD_ID   = config["first_prod_id"]
LAST_PROD_ID    = config["last_prod_id"]
VOCAB_SIZE_SRC  = config["vocab_size_src"]

feature_array = np.zeros((VOCAB_SIZE_SRC, len(feature_cols)), dtype=np.float32)
for _, row in df.iterrows():
    tid = int(row[config.get("prod_index_col")])
    if FIRST_PROD_ID <= tid <= LAST_PROD_ID:
        feature_array[tid] = row[feature_cols].values.astype(np.float32)
feature_tensor = torch.from_numpy(feature_array)

# Define special token IDs
PAD_ID      = tokenizer_tgt.token_to_id("[PAD]")
SOS_DEC_ID  = tokenizer_tgt.token_to_id("[SOS]")
EOS_DEC_ID  = tokenizer_tgt.token_to_id("[EOS]")
UNK_DEC_ID  = tokenizer_tgt.token_to_id("[UNK]")
EOS_PROD_ID = LAST_PROD_ID + 1
SOS_PROD_ID = LAST_PROD_ID + 2
UNK_PROD_ID = LAST_PROD_ID + 3
SPECIAL_IDS = [PAD_ID, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

# -----------------  Model builders  ------------------------------
def _parse_hidden_size(name):
    m = re.search(r"h(\d+)", name)
    return int(m.group(1)) if m else config.get("hidden_size")

def build_gru(ckpt_name):
    from hP_tuning_GRU import GRUClassifier
    return GRUClassifier(_parse_hidden_size(ckpt_name))

def build_lstm(_):
    from LSTM import LSTMClassifier
    return LSTMClassifier()

def build_feature_transformer(_=None):
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src    = config["vocab_size_src"],
        vocab_size_tgt    = config["vocab_size_tgt"],
        max_seq_len       = SEQ_LEN_AI,
        d_model           = config["d_model"],
        d_ff              = config["d_ff"],
        n_layers          = config["N"],
        n_heads           = config["num_heads"],
        dropout           = config["dropout"],
        kernel_type       = config["kernel_type"],
        feature_tensor    = feature_tensor,
        special_token_ids = SPECIAL_IDS
    )

def build_index_transformer(_=None):
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src    = config["vocab_size_src"],
        vocab_size_tgt    = config["vocab_size_tgt"],
        max_seq_len       = SEQ_LEN_AI,
        d_model           = config["d_model"],
        d_ff              = config["d_ff"],
        n_layers          = config["N"],
        n_heads           = config["num_heads"],
        dropout           = config["dropout"],
        kernel_type       = config["kernel_type"],
        feature_tensor    = feature_tensor,
        special_token_ids = SPECIAL_IDS
    )

BUILDERS = {
    "featurebased": build_feature_transformer,
    "indexbased" : build_index_transformer,
    "gru"        : build_gru,
    "lstm"       : build_lstm,
}

# -----------------  Inference & harvest  -------------------------
def harvest(model, loader, tag):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            X   = batch["aggregate_input"].to(device)
            lab = batch["label"].to(device)
            logits = model(X)
            probs  = torch.softmax(logits, dim=-1)[:,:,1]
            scores.append(probs.cpu())
            labels.append(lab.cpu())
    y_score = torch.cat(scores).view(-1).numpy()
    y_true  = torch.cat(labels).view(-1).numpy()
    np.save(f"{tag}_val_scores.npy", y_score)
    np.save(f"{tag}_val_labels.npy", y_true)
    print(f"{tag}: AUPRC={average_precision_score(y_true, y_score):.4f}")

# -----------------  Main loop  ------------------------------------
for pt in LOCALDIR.glob("*.pt"):
    key = next((k for k in BUILDERS if k in pt.name.lower()), None)
    if key is None:
        print(f"⚠️ skipping {pt.name}; no builder match")
        continue

    print(f"\n▶ loading {pt.name} as {key}")
    net = BUILDERS[key](pt.name).to(device)

    # Robust load & filter state dict
    chk = torch.load(pt, map_location=device, weights_only=False)
    sd  = chk.get("model_state_dict", chk)
    if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
        sd = sd["module"]

    base_params = net.state_dict()
    filtered_sd = {}
    for k, v in sd.items():
        k0 = k[len("module."):]
        if k.startswith("module.") else k
        if k0 in base_params:
            filtered_sd[k0] = v

    net.load_state_dict(filtered_sd, strict=False)
    harvest(net, val_loader, key)

# -----------------  Upload results back to S3 --------------------
for npy in Path(".").glob("*_val_*.npy"):
    key_out = f"{PREFIX}{npy.name}"
    print(f"↑ uploading {npy} → s3://{BUCKET}/{key_out}")
    s3.upload_file(str(npy), BUCKET, key_out)
