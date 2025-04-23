#!/usr/bin/env python
# harvest_s3_models.py  –  run on your EC2 box
# --------------------------------------------------------------
import os, json, boto3, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

# -----------------  S3 parameters  --------------------------------
BUCKET   = "productgptbucket"            # ← edit
PREFIX   = "winningmodel/"               # key prefix inside the bucket
LOCALDIR = Path("/home/ec2-user/ProductGPT") # checkpoints land here

LOCALDIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

# -----------------  download .pt files  ---------------------------
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
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

# -----------------  dataset / dataloader  -------------------------
# import your existing helpers exactly as you used in training
from dataset4_decoderonly import TransformerDataset, load_json_dataset
from tokenizers import Tokenizer, models, pre_tokenizers     # for tokenisers

def build_tokenizer_src():
    # same fixed-vocab builder you pasted earlier
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {str(i): i for i in range(60)}
    vocab["[PAD]"] = 0;  vocab["[SOS]"] = 10;  vocab["[EOS]"] = 11;  vocab["[UNK]"] = 12
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

tokenizer_ai  = build_tokenizer_src()
tokenizer_tgt = build_tokenizer_src()

CFG = dict(               # **keep in sync with training**
    filepath      = "/home/ec2-user/data/your_val_split.json",  # ← edit
    seq_len_ai    = 15,      # -- as in training
    seq_len_tgt   = 1,
    num_heads     = 4,
    ai_rate       = 15,
    batch_size    = 256,
)

val_data = load_json_dataset(CFG["filepath"])
val_ds   = TransformerDataset(
    val_data, tokenizer_ai, tokenizer_tgt,
    CFG["seq_len_ai"], CFG["seq_len_tgt"],
    CFG["num_heads"], CFG["ai_rate"]
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=CFG["batch_size"], shuffle=False,
    num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------  model builders  -------------------------------
def build_feature_transformer():
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src = 60, vocab_size_tgt = 60,
        max_seq_len    = CFG["seq_len_ai"],
        d_model=64, d_ff=64, n_layers=4, n_heads=4,
        dropout=0.1, kernel_type="relu",
        feature_tensor=None, special_token_ids=[]
    )

def build_index_transformer():
    from model4_decoderonly_feature_git import build_transformer
    return build_transformer(
        vocab_size_src = 60, vocab_size_tgt = 60,
        max_seq_len    = CFG["seq_len_ai"],
        d_model=64, d_ff=64, n_layers=6, n_heads=8,
        dropout=0.1, kernel_type="relu",
        feature_tensor=None, special_token_ids=[]
    )

def build_gru():
    from model_gru import GRUClassifier        # ← your GRU class
    return GRUClassifier(input_dim=60, hidden_size=128, num_classes=60)

def build_lstm():
    from model_lstm import LSTMClassifier      # ← your LSTM class
    return LSTMClassifier(input_dim=60, hidden_size=64, num_classes=60)

# map filename hints → builder
BUILDERS = {
    "FeatureBased": build_feature_transformer,
    "IndexBased"  : build_index_transformer,
    "GRU"         : build_gru,
    "LSTM"        : build_lstm,
}

# -----------------  inference helper  -----------------------------
def harvest(model, loader, tag):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X   = batch["aggregate_input"].to(device)
            lab = batch["label"].to(device)
            logits = model(X)
            probs  = torch.softmax(logits, dim=-1)[:, :, 1]  # positive class at idx=1
            all_scores.append(probs.cpu())
            all_labels.append(lab.cpu())
    y_score = torch.cat(all_scores).view(-1).numpy()
    y_true  = torch.cat(all_labels).view(-1).numpy()
    np.save(f"{tag}_val_scores.npy", y_score)
    np.save(f"{tag}_val_labels.npy", y_true)
    auprc = average_precision_score(y_true, y_score)
    print(f"{tag}: AUPRC={auprc:.4f}")
    return auprc

# -----------------  loop over checkpoints  ------------------------
for pt in LOCALDIR.glob("*.pt"):
    tag = None
    for k in BUILDERS:
        if k.lower() in pt.name.lower():
            tag = k
            break
    if tag is None:
        print(f"⚠️  cannot map {pt.name} → builder; skipping")
        continue

    print(f"\n▶ loading {pt.name} as {tag}")
    net = BUILDERS[tag]().to(device)
    net.load_state_dict(torch.load(pt, map_location=device))
    harvest(net, val_loader, tag)

# optional – push the .npy results back to S3
for npy in Path(".").glob("*_val_*.npy"):
    key_out = f"{PREFIX}{npy.name}"
    print(f"↑ uploading {npy} → s3://{BUCKET}/{key_out}")
    s3.upload_file(str(npy), BUCKET, key_out)