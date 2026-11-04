#!/usr/bin/env python
# harvest_s3_models.py  – run on EC2
# ===============================================================

import os, re, boto3, numpy as np, pandas as pd, torch
from pathlib import Path
from multiprocessing import freeze_support
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from dataset4_decoderonly import TransformerDataset, load_json_dataset
from tokenizers import Tokenizer, models, pre_tokenizers

# ----------------------------------------------------------------
#  CONFIG – edit the two paths below if your files live elsewhere
# ----------------------------------------------------------------
BUCKET   = "productgptbucket"
PREFIX   = "winningmodel/"

TRAIN_JSON = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
VAL_JSON   = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedVal.json"   # ← if absent, script falls back to 10 % split
EXCEL_PATH = "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx"

LOCALDIR   = Path("/home/ec2-user/ProductGPT/checkpoints")

SEQ_LEN_AI  = 15
SEQ_LEN_TGT = SEQ_LEN_AI         # cover every decision position
AI_RATE     = 15                 # predict one decision every 15 tokens
NUM_HEADS   = 4
BATCH_SIZE  = 256
SEED        = 33

# ----------------------------------------------------------------
#  TOKENISER (fixed 0-59 vocab)
# ----------------------------------------------------------------
def build_tokenizer_fixed():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {str(i): i for i in range(60)}
    vocab.update({"[PAD]": 0, "[SOS]": 10, "[EOS]": 11, "[UNK]": 12})
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ----------------------------------------------------------------
#  FEATURE TENSOR  (60 × 34)
# ----------------------------------------------------------------
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
    arr = np.zeros((60, len(feature_cols)), dtype=np.float32)
    for _, row in df.iterrows():
        tid = int(row["NewProductIndex6"])
        if 13 <= tid <= 56:
            arr[tid] = row[feature_cols].values.astype(np.float32)
    return torch.from_numpy(arr)

feature_tensor = load_feature_tensor()

# ----------------------------------------------------------------
#  MODEL BUILDERS
# ----------------------------------------------------------------
def _parse_hidden_size(name, default=128):
    m = re.search(r"h(\d+)", name)
    return int(m.group(1)) if m else default

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

# ----------------------------------------------------------------
#  ONE-HOT helper (for GRU/LSTM)
# ----------------------------------------------------------------
def one_hot(x, vocab_size=60):
    # x : (B,T) int64 → (B,T,vocab_size) float32
    return torch.nn.functional.one_hot(x, vocab_size).float()

# ----------------------------------------------------------------
#  CHECKPOINT DOWNLOADER
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
#  VALIDATION DATALOADER  (no leakage)
# ----------------------------------------------------------------
def build_val_loader():
    tok_ai  = build_tokenizer_fixed()
    tok_tgt = build_tokenizer_fixed()

    if Path(VAL_JSON).exists():          # dedicated validation file
        val_data = load_json_dataset(VAL_JSON)
    else:                                # fall back to 10 % split
        all_data = load_json_dataset(TRAIN_JSON)
        n        = len(all_data)
        train_n  = int(0.8*n)
        val_n    = int(0.1*n)
        torch.manual_seed(SEED)
        _, val_data, _ = random_split(
            all_data, [train_n, val_n, n-train_n-val_n],
            generator=torch.Generator().manual_seed(SEED)
        )

    ds = TransformerDataset(
        val_data,
        tok_ai, tok_tgt,
        SEQ_LEN_AI, SEQ_LEN_TGT,
        NUM_HEADS, AI_RATE
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)

# ----------------------------------------------------------------
#  HARVEST ONE MODEL
# ----------------------------------------------------------------
def harvest(model, loader, tag):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            X   = batch["aggregate_input"].to(device)  # (B,T) int64
            lab = batch["label"].to(device)

            # convert for RNN checkpoints
            if tag in {"gru", "lstm"}:
                X_in = one_hot(X)                      # (B,T,60) float32
            else:
                X_in = X                               # (B,T) int64

            out = model(X_in)                          # (B,T,V)
            dp  = torch.arange(AI_RATE-1, out.size(1), AI_RATE, device=device)
            dec_logits = out[:, dp, :]                 # (B,D,V)
            dec_labels = lab[:, dp]                    # (B,D)

            logits_all.append(dec_logits.cpu())
            labels_all.append(dec_labels.cpu())

    logits = torch.cat(logits_all).view(-1, out.size(-1)).numpy()
    labels = torch.cat(labels_all).view(-1).numpy()

    classes = np.arange(1,10)
    y_true  = label_binarize(labels, classes=classes)          # (N,9)
    softmax = np.exp(logits)
    softmax = softmax/softmax.sum(axis=1, keepdims=True)
    y_prob  = softmax[:, classes]                              # (N,9)

    # save raw arrays
    np.save(f"{tag}_val_labels.npy", labels)
    np.save(f"{tag}_val_scores.npy", y_prob)

    auprc = average_precision_score(y_true, y_prob, average="macro")
    print(f"{tag}: N_decisions={labels.size:,}  Macro-AUPRC={auprc:.4f}")

# ----------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------
def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s3          = download_checkpoints()
    val_loader  = build_val_loader()

    for pt in LOCALDIR.glob("*.pt"):
        tag = next((k for k in BUILDERS if k in pt.name.lower()), None)
        if tag is None:
            print(f"⚠️ skipping {pt.name}")
            continue

        print(f"\n▶ loading {pt.name} as {tag}")
        net = BUILDERS[tag](pt.name).to(device)

        ckpt = torch.load(pt, map_location=device, weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
            sd = sd["module"]

        net_dict  = net.state_dict()
        clean_sd  = {k.removeprefix("module."):v for k,v in sd.items()
                     if k.removeprefix("module.") in net_dict
                     and v.shape == net_dict[k.removeprefix("module.")].shape}

        net.load_state_dict(clean_sd, strict=False)
        harvest(net, val_loader, tag)

    # ── push .npy files back to S3
    for f in Path(".").glob("*_val_*.npy"):
        dst = f"{PREFIX}{f.name}"
        print(f"↑ uploading {f} → s3://{BUCKET}/{dst}")
        s3.upload_file(str(f), BUCKET, dst)

# ----------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()    # needed on spawn platforms
    main()
