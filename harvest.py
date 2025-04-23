#!/usr/bin/env python
# harvest_s3_models.py  – run on EC2
# --------------------------------------------------------------
import os, re, json, boto3, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# -----------------  S3 parameters  -------------------------------
BUCKET   = "productgptbucket"
PREFIX   = "winningmodel/"
LOCALDIR = Path("/home/ec2-user/ProductGPT/checkpoints")
LOCALDIR.mkdir(parents=True, exist_ok=True)
s3 = boto3.client("s3")

# -----------------  download .pt files  --------------------------
resp  = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
ckpts = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".pt")]
if not ckpts:
    raise RuntimeError("No .pt files found under that prefix!")
for key in ckpts:
    dest = LOCALDIR / Path(key).name
    if not dest.exists():
        print(f"▶ downloading s3://{BUCKET}/{key} → {dest}")
        s3.download_file(BUCKET, key, str(dest))
    else:
        print(f"✓ already have {dest}")

# -----------------  dataloader setup  ----------------------------
from torch.utils.data import random_split, DataLoader
from dataset4_decoderonly import TransformerDataset, load_json_dataset
from tokenizers import Tokenizer, models, pre_tokenizers

RAW_JSON    = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
SEQ_LEN_AI  = 15
SEQ_LEN_TGT = 1
NUM_HEADS   = 4
AI_RATE     = 15
BATCH_SIZE  = 256
SEED        = 33

def build_tokenizer_fixed():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {str(i): i for i in range(60)}
    vocab.update({"[PAD]":0, "[SOS]":10, "[EOS]":11, "[UNK]":12})
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

tokenizer_ai  = build_tokenizer_fixed()
tokenizer_tgt = build_tokenizer_fixed()

full_data  = load_json_dataset(RAW_JSON)
train_sz   = int(0.8 * len(full_data))
val_sz     = int(0.1 * len(full_data))
test_sz    = len(full_data) - train_sz - val_sz

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
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------  load feature_tensor  -------------------------
df = pd.read_excel("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx", sheet_name=0)
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
feature_array = np.zeros((60, len(feature_cols)), dtype=np.float32)
for _, row in df.iterrows():
    tid = int(row["NewProductIndex6"])
    if FIRST_PROD_ID <= tid <= LAST_PROD_ID:
        feature_array[tid] = row[feature_cols].values.astype(np.float32)
feature_tensor = torch.from_numpy(feature_array)

# -----------------  model builders  ------------------------------
def _parse_hidden_size(name, default=128):
    m = re.search(r"h(\d+)", name)
    return int(m.group(1)) if m else default

def build_gru(ckpt_name):
    from hP_tuning_GRU import GRUClassifier
    return GRUClassifier(_parse_hidden_size(ckpt_name))

def build_lstm(_):
    from LSTM import LSTMClassifier
    return LSTMClassifier()

def build_feature_transformer(_=None):
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src=60, vocab_size_tgt=60,
        max_seq_len=SEQ_LEN_AI, d_model=64, d_ff=64,
        n_layers=4, n_heads=4, dropout=0.1, kernel_type="relu",
        feature_tensor=feature_tensor,
        special_token_ids=[]
    )

def build_index_transformer(_=None):
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src=60, vocab_size_tgt=60,
        max_seq_len=SEQ_LEN_AI, d_model=64, d_ff=64,
        n_layers=6, n_heads=8, dropout=0.1, kernel_type="relu",
        feature_tensor=feature_tensor,
        special_token_ids=[]
    )

BUILDERS = {
    "featurebased": build_feature_transformer,
    "indexbased"  : build_index_transformer,
    "gru"         : build_gru,
    "lstm"        : build_lstm,
}

# -----------------  inference helper  ----------------------------
def harvest(model, loader, tag):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            X   = batch["aggregate_input"].to(device)
            lab = batch["label"].to(device)
            logits = model(X)
            probs  = torch.softmax(logits, dim=-1)[:,:,1]
            scores.append(probs.cpu()); labels.append(lab.cpu())
    y_score = torch.cat(scores).view(-1).numpy()
    y_true  = torch.cat(labels).view(-1).numpy()
    np.save(f"{tag}_val_scores.npy", y_score)
    np.save(f"{tag}_val_labels.npy", y_true)
    auprc = average_precision_score(y_true, y_score)
    print(f"{tag}: AUPRC={auprc:.4f}")

# -----------------  main loop  -----------------------
for pt in LOCALDIR.glob("*.pt"):
    key = next((k for k in BUILDERS if k in pt.name.lower()), None)
    if key is None:
        print(f"⚠️ skipping {pt.name}; no builder match")
        continue

    print(f"\n▶ loading {pt.name} as {key}")
    # load full pickle (weights_only=False) so we can extract our state_dict
    chk = torch.load(pt, map_location=device, weights_only=False)
    # extract the dict we saved in training
    sd  = chk.get("model_state_dict", chk)
    # if DeepSpeed wrapped it under 'module', unwrap
    if isinstance(sd, dict) and "module" in sd:
        sd = sd["module"]
    # rebuild and load
    net = BUILDERS[key](pt.name).to(device)
    net.load_state_dict(sd)
    harvest(net, val_loader, key)

# -----------------  upload results  ----------------------
for npy in Path(".").glob("*_val_*.npy"):
    key_out = f"{PREFIX}{npy.name}"
    print(f"↑ uploading {npy} → s3://{BUCKET}/{key_out}")
    s3.upload_file(str(npy), BUCKET, key_out)
