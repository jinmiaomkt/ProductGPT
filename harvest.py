#!/usr/bin/env python
import os
import re
import boto3
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from multiprocessing import freeze_support
from tqdm import tqdm

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from torch.utils.data import random_split, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers

# Transformer dataset + loader
from dataset4_decoderonly import TransformerDataset, load_json_dataset

# ── CONSTANTS ─────────────────────────────────────────────────────
BUCKET      = "productgptbucket"
PREFIX      = "winningmodel/"
CKPT_DIR    = Path("/home/ec2-user/ProductGPT/checkpoints")
RAW_JSON    = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
EXCEL_PATH  = "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx"

SEQ_LEN_AI  = 15
SEQ_LEN_TGT = SEQ_LEN_AI
NUM_HEADS   = 4
AI_RATE     = 15
BATCH_SIZE  = 256
SEED        = 33

# ── COMMON HELPERS ────────────────────────────────────────────────
def build_tokenizer_fixed():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {str(i): i for i in range(60)}
    vocab.update({"[PAD]": 0, "[SOS]": 10, "[EOS]": 11, "[UNK]": 12})
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

def download_checkpoints():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".pt")]
    if not keys:
        raise RuntimeError("No .pt files under that prefix!")
    for key in keys:
        dest = CKPT_DIR/Path(key).name
        if not dest.exists():
            print(f"▶ downloading s3://{BUCKET}/{key} → {dest}")
            s3.download_file(BUCKET, key, str(dest))
        else:
            print(f"✓ have {dest}")
    return s3

# ── TRANSFORMER PIPELINE ──────────────────────────────────────────
# load your 34-dim feature tensor as before
def load_feature_tensor():
    df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    cols = [  # same 34 feature columns
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
    arr = np.zeros((60, len(cols)), dtype=np.float32)
    for _, row in df.iterrows():
        tid = int(row["NewProductIndex6"])
        if 13 <= tid <= 56:
            arr[tid] = row[cols].values.astype(np.float32)
    return torch.from_numpy(arr)

# generic harvest for transformer
def harvest_transformer(model, loader, tag, device, feature_tensor):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X   = batch["aggregate_input"].to(device)  # (B, seq_len)
            lab = batch["label"].to(device)
            logits = model(X)  # (B, seq_len, V)
            dp = torch.arange(AI_RATE-1, logits.size(1), AI_RATE, device=device)
            dl = logits[:, dp, :]     # (B, D, V)
            ll = lab[:, dp]           # (B, D)
            all_logits.append(dl.cpu())
            all_labels.append(ll.cpu())

    logits_arr = torch.cat(all_logits).view(-1, dl.size(-1)).numpy()
    labels_arr = torch.cat(all_labels).view(-1).numpy()
    classes    = np.arange(1,10)
    y_true     = label_binarize(labels_arr, classes=classes)
    exp_logits = np.exp(logits_arr)
    probs      = exp_logits[:, classes] / exp_logits.sum(axis=1, keepdims=True)

    np.save(f"{tag}_val_scores.npy", probs)
    auprc = average_precision_score(y_true, probs, average="macro")
    print(f"{tag}: Macro-AUPRC = {auprc:.4f}")

# ── RNN PIPELINE ──────────────────────────────────────────────────
# for GRU
from hP_tuning_GRU import SequenceDataset as GRU_DS, collate_fn as gru_collate, evaluate as gru_evaluate, NUM_CLASSES as GRU_NUM_CLASSES, CLASS_9_WEIGHT as GRU_WEIGHT  # :contentReference[oaicite:1]{index=1}&#8203;:contentReference[oaicite:2]{index=2}
# for LSTM
from LSTM       import SequenceDataset as LSTM_DS, collate_fn as lstm_collate, evaluate as lstm_evaluate, NUM_CLASSES as LSTM_NUM_CLASSES, CLASS_9_WEIGHT as LSTM_WEIGHT  # :contentReference[oaicite:3]{index=3}&#8203;:contentReference[oaicite:4]{index=4}

def harvest_rnn(model, loader, tag, device, key):
    # pick correct evaluate + weights
    if key == "gru":
        eval_fn, num_cls, cw = gru_evaluate, GRU_NUM_CLASSES, GRU_WEIGHT
    else:
        eval_fn, num_cls, cw = lstm_evaluate, LSTM_NUM_CLASSES, LSTM_WEIGHT

    # build loss_fn identical to training
    weights = torch.ones(num_cls, device=device)
    weights[-1] = cw
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
    # run their evaluate to reproduce f1 & auprc
    _, _, _, _, _, auprc = eval_fn(loader, model, device, loss_fn)
    print(f"{tag}: Macro-AUPRC = {auprc:.4f}")

# ── MODEL BUILDERS ────────────────────────────────────────────────
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
    # hidden size parsed from filename
    m = re.search(r"h(\d+)", ckpt_name)
    hs = int(m.group(1)) if m else 128
    return GRUClassifier(hs)

def build_lstm(_):
    from LSTM import LSTMClassifier
    return LSTMClassifier()

BUILDERS = {
    "featurebased": build_feature_transformer,
    "indexbased":   build_index_transformer,
    "gru":          build_gru,
    "lstm":         build_lstm,
}

def prepare_transformer_loader():
    data = load_json_dataset(RAW_JSON)
    n = len(data)
    t, v = int(0.8*n), int(0.1*n)
    torch.manual_seed(SEED)
    _, val, _ = random_split(
        data, [t, v, n-t-v],
        generator=torch.Generator().manual_seed(SEED)
    )
    tok_ai, tok_tgt = build_tokenizer_fixed(), build_tokenizer_fixed()
    ds = TransformerDataset(val, tok_ai, tok_tgt, SEQ_LEN_AI, SEQ_LEN_TGT, NUM_HEADS, AI_RATE)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def prepare_rnn_loader(key):
    if key == "gru":
        ds, coll = GRU_DS(RAW_JSON), gru_collate
    else:
        ds, coll = LSTM_DS(RAW_JSON), lstm_collate
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=coll, num_workers=4, pin_memory=True)

def main():
    global feature_tensor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = load_feature_tensor()

    s3 = download_checkpoints()

    for pt in CKPT_DIR.glob("*.pt"):
        key = next((k for k in BUILDERS if k in pt.name.lower()), None)
        if key is None:
            print(f"⚠️ skipping {pt.name}")
            continue

        print(f"\n▶ {pt.name} → builder `{key}`")
        net = BUILDERS[key](pt.name).to(device)
        chk = torch.load(pt, map_location=device, weights_only=False)
        sd  = chk.get("model_state_dict", chk)
        if isinstance(sd, dict) and "module" in sd:
            sd = sd["module"]
        # filter & load only matching shapes
        base = net.state_dict()
        filt = {k.removeprefix("module."):v for k,v in sd.items()
                if k.removeprefix("module.") in base and v.shape == base[k.removeprefix("module.")].shape}
        net.load_state_dict(filt, strict=False)

        if key in ("featurebased","indexbased"):
            loader = prepare_transformer_loader()
            harvest_transformer(net, loader, key, device, feature_tensor)
        else:
            loader = prepare_rnn_loader(key)
            harvest_rnn(net, loader, key, device, key)

    # (optionally) re-upload any *_val_scores.npy to S3
    for f in Path(".").glob("*_val_scores.npy"):
        dest = f"{PREFIX}{f.name}"
        print(f"↑ uploading {f} → s3://{BUCKET}/{dest}")
        s3.upload_file(str(f), BUCKET, dest)

if __name__ == "__main__":
    freeze_support()
    main()
