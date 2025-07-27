import torch, json, boto3, argparse, random 
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset4_productgpt import load_json_dataset, TransformerDataset   # reuse tokeniser
from predict_decision_probs_gru import GRUClassifier                    # already in repo
from predict_decision_probs_lstm import LSTMClassifier                                   # assume you have this
from tokenizers import Tokenizer, models, pre_tokenizers
from typing import Any, Dict, List, Tuple

# ══════════════════════════════ 1. Constants ═══════════════════════════
PAD_ID = 0
DECISION_IDS = list(range(1, 10))  # 1‑9
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
SPECIAL_IDS = [
    PAD_ID,
    SOS_DEC_ID,
    EOS_DEC_ID,
    UNK_DEC_ID,
    EOS_PROD_ID,
    SOS_PROD_ID,
]
MAX_TOKEN_ID = UNK_PROD_ID  # 59

def _base_tokeniser(extra_vocab: Dict[str, int] | None = None) -> Tokenizer:
    """Word‑level tokeniser with a fixed numeric vocabulary."""
    vocab: Dict[str, int] = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},  # decisions
        "[SOS]": SOS_DEC_ID,
        "[EOS]": EOS_DEC_ID,
        "[UNK]": UNK_DEC_ID,
    }
    if extra_vocab:
        vocab.update(extra_vocab)
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok


def build_tokenizer_src() -> Tokenizer:  # with product IDs
    prod_vocab = {str(i): i for i in range(FIRST_PROD_ID, UNK_PROD_ID + 1)}
    return _base_tokeniser(prod_vocab)


def build_tokenizer_tgt() -> Tokenizer:  # decisions only
    return _base_tokeniser()


# ---------- tiny util ---------------------------------------------------
def save_and_upload(obj, local_path: Path, bucket: str, key: str): 
    torch.save(obj, local_path)
    boto3.client("s3").upload_file(str(local_path), bucket, key)

# ---------- cli ---------------------------------------------------------
p = argparse.ArgumentParser() 
p.add_argument("--fold", type=int, required=True)
p.add_argument("--model", choices=["gru","lstm"], required=True)
p.add_argument("--bucket", default="productgptbucket")
p.add_argument("--train_file", default="/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json")
p.add_argument("--test_file",  default="/home/ec2-user/data/clean_list_int_wide4_simple6.json")
args = p.parse_args()
cfg = {"input_dim":15, "batch":4, "hidden":128 if args.model=="gru" else 64,
       "epochs":8, "lr":1e-3 if args.model=="lstm" else 1e-4,
       "fold":args.fold, "model":args.model}

# ---------- load fold mapping ------------------------------------------
spec = json.loads(boto3.client("s3")
                  .get_object(Bucket=args.bucket, Key="CV/folds.json")
                  ["Body"].read())
uids_test = [u for u,f in spec["assignment"].items() if f==cfg["fold"]]
uids_trva = [u for u in spec["assignment"] if u not in uids_test]

# ---------- build Dataset & loaders ------------------------------------
raw = load_json_dataset(args.train_file, keep_uids=set(uids_trva))
n = len(raw); tr,va = int(.8*n), int(.1*n)
tr_ds, va_ds, _ = random_split(raw,[tr,va,n-tr-va],
                               generator=torch.Generator().manual_seed(33))
tok_src, tok_tgt = build_tokenizer_src(), build_tokenizer_tgt()
wrap = lambda ds: TransformerDataset(ds,tok_src,tok_tgt,
                                     seq_len_ai=225, seq_len_tgt=15,
                                     num_heads=1, ai_rate=15)
tr_loader = DataLoader(wrap(tr_ds), batch_size=cfg["batch"], shuffle=True)
va_loader = DataLoader(wrap(va_ds), batch_size=cfg["batch"])

# ---------- model -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if cfg["model"] == "gru":
    net = GRUClassifier(cfg["input_dim"], cfg["hidden"]).to(device)
else:
    net = LSTMClassifier(cfg["input_dim"], cfg["hidden"]).to(device)

opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
best_va = 1e9; ckpt = {}
for ep in range(cfg["epochs"]):
    net.train()
    for b in tr_loader:
        x = b["aggregate_input"].float().to(device)   # (B,T,D)
        y = b["label"].to(device)
        logits = net(x)[:, -1, :]                     # predict last step
        loss = torch.nn.functional.cross_entropy(logits, y[:,0])
        opt.zero_grad(); loss.backward(); opt.step()
    # --- simple val ---
    net.eval(); tot=0; correct=0
    with torch.no_grad():
        for b in va_loader:
            x=b["aggregate_input"].float().to(device)
            y=b["label"][:,0].to(device)
            pred = net(x)[:, -1, :].argmax(1)
            tot += y.size(0); correct += (pred==y).sum().item()
    va_loss = 1-correct/tot
    if va_loss < best_va:
        best_va = va_loss
        ckpt = {"model_state_dict": net.state_dict()}
        torch.save(ckpt, "best.pt")

# ---------- upload checkpoint ------------------------------------------
name = f"{cfg['model']}_h{cfg['hidden']}_lr{cfg['lr']}_bs{cfg['batch']}_fold{cfg['fold']}.pt"
save_and_upload(ckpt,"best.pt", args.bucket, f"CV_{cfg['model'].upper()}/checkpoints/{name}")

# ---------- inference on FULL 30 campaigns -----------------------------
from predict_decision_probs_gru import PredictDataset, collate
net.load_state_dict(ckpt["model_state_dict"]); net.eval()
full_ds  = PredictDataset(Path(args.test_file), cfg["input_dim"])
full_ld  = DataLoader(full_ds, batch_size=128, collate_fn=collate)
pred_path = f"{name.replace('.pt','')}_predictions.jsonl"
with open(pred_path,"w") as fp, torch.no_grad():
    for batch in full_ld:
        x = batch["x"].to(device)
        probs = torch.softmax(net(x),-1).cpu().tolist()
        for uid,p in zip(batch["uid"], probs):
            fp.write(json.dumps({"uid":uid,"probs":p})+"\n")
boto3.client("s3").upload_file(pred_path, args.bucket,
        f"CV_{cfg['model'].upper()}/predictions/{pred_path}")

# ---------- store simple metric (acc on val set) -----------------------
metric = {"fold":cfg["fold"], "val_error":best_va, "ckpt":name,
          "pred_file":pred_path}
json_name = name.replace(".pt",".json")
Path("m.json").write_text(json.dumps(metric,indent=2))
boto3.client("s3").upload_file("m.json", args.bucket,
        f"CV_{cfg['model'].upper()}/metrics/{json_name}")
print(f"[✓] fold{cfg['fold']} for {cfg['model']} done.")
