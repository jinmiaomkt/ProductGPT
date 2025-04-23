#!/usr/bin/env python
# harvest_s3_models.py  – run on EC2

import os
import re
import boto3
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from multiprocessing import freeze_support

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import random_split, DataLoader

from dataset4_decoderonly import TransformerDataset, load_json_dataset
from tokenizers import Tokenizer, models, pre_tokenizers

# ── CONSTANTS ─────────────────────────────────────────────────────
BUCKET      = "productgptbucket"
PREFIX      = "winningmodel/"
LOCALDIR    = Path("/home/ec2-user/ProductGPT/checkpoints")
RAW_JSON    = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
EXCEL_PATH  = "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx"

SEQ_LEN_AI  = 15
SEQ_LEN_TGT = SEQ_LEN_AI   # must cover every decision position
NUM_HEADS   = 4
AI_RATE     = 15
BATCH_SIZE  = 256
SEED        = 33

# ── HELPERS ───────────────────────────────────────────────────────
def build_tokenizer_fixed():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {str(i): i for i in range(60)}
    vocab.update({"[PAD]": 0, "[SOS]": 10, "[EOS]": 11, "[UNK]": 12})
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

def _parse_hidden_size(name, default=128):
    m = re.search(r"h(\d+)", name)
    return int(m.group(1)) if m else default

def load_feature_tensor():
    df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    feature_cols = [
        "Rarity","MaxLife","MaxOffense","MaxDefense",
        "WeaponTypeOneHandSword","WeaponTypeTwoHandSword",
        "WeaponTypeArrow","WeaponTypeMagic","WeaponTypePolearm",
        "EthnicityIce","EthnicityRock","EthnicityWater","EthnicityFire",
        "EthnicityThunder","EthnicityWind","GenderFemale","GenderMale",
        "CountryRuiYue","CountryDaoQi","CountryZhiDong","CountryMengDe",
        "type_figure","MinimumAttack","MaximumAttack",
        "MinSpecialEffect","MaxSpecialEffect","SpecialEffectEfficiency",
        "SpecialEffectExpertise","SpecialEffectAttack","SpecialEffectSuper",
        "SpecialEffectRatio","SpecialEffectPhysical","SpecialEffectLife","LTO"
    ]
    FIRST_PROD_ID, LAST_PROD_ID = 13, 56
    arr = np.zeros((60, len(feature_cols)), dtype=np.float32)
    for _, row in df.iterrows():
        tid = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= tid <= LAST_PROD_ID:
            arr[tid] = row[feature_cols].values.astype(np.float32)
    return torch.from_numpy(arr)

def build_feature_transformer(_=None):
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src=60, vocab_size_tgt=60,
        max_seq_len=SEQ_LEN_AI, d_model=64, d_ff=64,
        n_layers=4, n_heads=4, dropout=0.1, kernel_type="relu",
        feature_tensor=feature_tensor, special_token_ids=[]
    )

def build_index_transformer(_=None):
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src=60, vocab_size_tgt=60,
        max_seq_len=SEQ_LEN_AI, d_model=64, d_ff=64,
        n_layers=6, n_heads=8, dropout=0.1, kernel_type="relu",
        feature_tensor=feature_tensor, special_token_ids=[]
    )

def build_gru(ckpt_name):
    from hP_tuning_GRU import GRUClassifier
    return GRUClassifier(_parse_hidden_size(ckpt_name))

def build_lstm(_):
    from LSTM import LSTMClassifier
    return LSTMClassifier()

BUILDERS = {
    "featurebased": build_feature_transformer,
    "indexbased":   build_index_transformer,
    "gru":          build_gru,
    "lstm":         build_lstm,
}

def download_checkpoints():
    LOCALDIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    resp  = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    ckpts = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".pt")]
    if not ckpts:
        raise RuntimeError("No .pt files found under that prefix!")
    for key in ckpts:
        dest = LOCALDIR / Path(key).name
        if not dest.exists():
            print(f"▶ downloading s3://{BUCKET}/{key} → {dest}")
            s3.download_file(BUCKET, key, str(dest))
        else:
            print(f"✓ already have {dest}")
    return s3

def prepare_dataloader():
    # seed → deterministic split
    data = load_json_dataset(RAW_JSON)
    n = len(data)
    t, v = int(0.8*n), int(0.1*n)
    torch.manual_seed(SEED)
    _, val, _ = random_split(
        data, [t, v, n-t-v],
        generator=torch.Generator().manual_seed(SEED)
    )
    tok_ai  = build_tokenizer_fixed()
    tok_tgt = build_tokenizer_fixed()
    ds = TransformerDataset(val, tok_ai, tok_tgt, SEQ_LEN_AI, SEQ_LEN_TGT, NUM_HEADS, AI_RATE)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def harvest(model, loader, tag):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X   = batch["aggregate_input"].to(device)
            lab = batch["label"].to(device)
            logits = model(X)  # (B, seq_len, V)
            dp = torch.arange(AI_RATE-1, logits.size(1), AI_RATE, device=device)
            dl = logits[:, dp, :]     # (B, D, V)
            ll = lab[:, dp]           # (B, D)
            all_logits.append(dl.cpu())
            all_labels.append(ll.cpu())

    logits_arr = torch.cat(all_logits).view(-1, dl.size(-1)).numpy()  # (N, V)
    labels_arr = torch.cat(all_labels).view(-1).numpy()               # (N,)
    classes    = np.arange(1,10)
    y_true_bin = label_binarize(labels_arr, classes=classes)         # (N,9)
    # softmax over logits for classes 1..9
    exp_logits = np.exp(logits_arr)
    probs      = exp_logits[:, classes] / exp_logits.sum(axis=1, keepdims=True)

    # save
    np.save(f"{tag}_val_labels.npy", labels_arr)
    np.save(f"{tag}_val_scores.npy", probs)

    auprc = average_precision_score(y_true_bin, probs, average="macro")
    print(f"{tag}: N={labels_arr.size:,}  class‐pos‐rates={y_true_bin.mean(axis=0)}")
    print(f"{tag}: Macro‐AUPRC = {auprc:.4f}")

def main():
    global feature_tensor, device
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load feature tensor once
    feature_tensor = load_feature_tensor()

    # download checkpoints & get s3 client
    s3 = download_checkpoints()

    # build validation DataLoader
    val_loader = prepare_dataloader()

    # loop
    for pt in LOCALDIR.glob("*.pt"):
        key = next((k for k in BUILDERS if k in pt.name.lower()), None)
        if key is None:
            print(f"⚠️ skipping {pt.name}")
            continue

        print(f"\n▶ loading {pt.name} as {key}")
        net = BUILDERS[key](pt.name).to(device)

        chk = torch.load(pt, map_location=device, weights_only=False)
        sd  = chk.get("model_state_dict", chk)
        if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
            sd = sd["module"]

        net_dict = net.state_dict()
        filtered = {}
        for k,v in sd.items():
            name0 = k.removeprefix("module.")
            if name0 in net_dict and v.shape == net_dict[name0].shape:
                filtered[name0] = v

        net.load_state_dict(filtered, strict=False)
        harvest(net, val_loader, key)

    # upload
    for f in Path(".").glob("*_val_*.npy"):
        dest = f"{PREFIX}{f.name}"
        print(f"↑ uploading {f} → s3://{BUCKET}/{dest}")
        s3.upload_file(str(f), BUCKET, dest)

if __name__ == "__main__":
    freeze_support()
    main()
