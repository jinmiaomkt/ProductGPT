"""
Decision-Only trainer
────────────────────────────────────────────────────────────────────────────
• revenue–gap loss
• DeepSpeed stage-1 (FusedLamb)
• Quiet (all DS chatter silenced)
• On every better val-loss:
    └─ DecisionOnly_<uid>.pt    → s3://<bucket>/DecisionOnly/checkpoints/
    └─ DecisionOnly_<uid>.json  → s3://<bucket>/DecisionOnly/metrics/
• Local copies are deleted after a successful upload
"""

# ────────────────────────── silence noisy libs ──────────────────────────
import os, warnings, logging, json, numpy as np, boto3, botocore
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL="3",
                  DEEPSPEED_LOG_LEVEL="error",
                  DS_DISABLE_LOGS="1")
warnings.filterwarnings("ignore")
for n in ("deepspeed", "torch_checkpoint_engine", "engine"):
    logging.getLogger(n).setLevel(logging.ERROR)
    logging.getLogger(n).propagate = False

# ────────────────────────── std / 3rd-party imports ─────────────────────
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb                                    # optimiser

from model4_decoderonly     import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers
from config4_decision_only_git import get_config

# ────────────────────────── helpers ─────────────────────────────────────
def json_safe(obj):
    """Recursively turn tensors / numpy into vanilla Python."""
    import numpy as _np, torch as _th
    if isinstance(obj, (_th.Tensor, _th.nn.Parameter)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, _np.ndarray):  return obj.tolist()
    if isinstance(obj, (_np.generic,)):  return obj.item()
    if isinstance(obj, dict):  return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return obj

def get_s3():
    try: return boto3.client("s3")
    except botocore.exceptions.BotoCoreError: return None

def upload_s3(local:Path, bucket:str, key:str, client) -> bool:
    if client is None: return False
    try:
        client.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name} → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-ERR] {e}")
        return False

def build_tok():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(
        {"[PAD]":0, **{str(i):i for i in range(1,10)}, "[SOS]":10, "[UNK]":12},
        unk_token="[UNK]")
    return tok

def transition_mask(lbl):                       # (B,T)
    prev = F.pad(lbl,(1,0),value=-1)[:,:-1]
    return lbl != prev

def perplexity(logits, tgt, pad):
    lp = F.log_softmax(logits, dim=-1)
    lp2d, t = lp.view(-1,lp.size(-1)), tgt.view(-1)
    m = t != pad
    return float("nan") if m.sum()==0 else \
        torch.exp(F.nll_loss(lp2d[m],t[m],reduction="mean")).item()

# ────────────────────────── loss ────────────────────────────────────────
class RevenueLoss(nn.Module):
    def __init__(self, rev, V, ignore=0):
        super().__init__()
        rev = rev + [0.]*(V-len(rev)) if len(rev)<V else rev
        rev = torch.tensor(rev)
        self.register_buffer("pen", -torch.abs(rev[:,None]-rev[None,:]))
        self.ignore = ignore
    def forward(self, logits, tgt):
        B,T,V = logits.shape
        probs = F.softmax(logits.view(-1,V),dim=-1)
        tgt   = tgt.view(-1)
        keep  = tgt != self.ignore
        if keep.sum()==0: return logits.new_tensor(0.0)
        gap = (probs[keep]*self.pen.to(probs)[tgt[keep]]).sum(1)
        return (-gap).mean()

# ────────────────────────── data / model builders ───────────────────────
def dataloaders(cfg):
    data = load_json_dataset(cfg["filepath"])
    n=len(data); tr,va = int(.8*n), int(.1*n)
    tr_ds,va_ds,te_ds = random_split(data,[tr,va,n-tr-va],
        generator=torch.Generator().manual_seed(33))
    tok_ai = tok_tgt = build_tok()
    Path(cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
    tok_ai.save (str(Path(cfg["model_folder"])/"tokenizer_ai.json"))
    tok_tgt.save(str(Path(cfg["model_folder"])/"tokenizer_tgt.json"))

    def mk(split):
        return TransformerDataset(split,tok_ai,tok_tgt,
            cfg["seq_len_ai"],cfg["seq_len_tgt"],
            cfg["num_heads"],cfg["ai_rate"],pad_token=0)
    mk_loader = lambda d,sh: DataLoader(d,batch_size=cfg["batch_size"],shuffle=sh)
    return mk_loader(mk(tr_ds),True), mk_loader(mk(va_ds),False), mk_loader(mk(te_ds),False), tok_tgt

def model(cfg):
    return build_transformer(cfg["vocab_size_tgt"], cfg["seq_len_ai"],
                             cfg["d_model"], cfg["N"], cfg["num_heads"],
                             cfg["d_ff"], cfg["dropout"])

# ────────────────────────── validation ──────────────────────────────────
def metrics(loader, eng, device, loss_fn, step, pad, tok):
    if not loader: nan=float("nan"); return nan,nan,{}, {}, {}, {}
    sp={pad,tok.token_to_id("[SOS]"),tok.token_to_id("[UNK]")}
    tl,tp=0.,0.; P=L=PR=[]; ms=mp=mt=[]
    P=[];L=[];PR=[];ms=[];mp_=[];mt=[]
    eng.eval()
    with torch.no_grad():
        for b in loader:
            x=b["aggregate_input"].to(device); y=b["label"].to(device)
            g=eng(x); g=g[:,torch.arange(step-1,g.size(1),step,device=device),:]
            y_e=y.clone(); y_e[transition_mask(y)]=pad
            tl+=loss_fn(g,y_e).item(); tp+=perplexity(g,y_e,pad)
            pr=F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pd=pr.argmax(1); lb=y.view(-1).cpu().numpy()
            v=~np.isin(lb,list(sp))
            P.append(pd[v]);L.append(lb[v]);PR.append(pr[v])
            ms.append((y==9).view(-1).cpu().numpy()[v])
            mp_.append((F.pad(y,(1,0),value=-1)[:,:-1]==9).view(-1).cpu().numpy()[v])
            mt.append(transition_mask(y).view(-1).cpu().numpy()[v])
    P,L,PR=np.concatenate(P),np.concatenate(L),np.concatenate(PR)
    ms,mp_,mt = map(np.concatenate,(ms,mp_,mt))
    m_main = ~mt
    sub=lambda m: _subset(P,L,PR,m)
    return tl/len(loader), tp/len(loader), sub(m_main), sub(ms), sub(mp_), sub(mt)

def _subset(P,L,PR,m, cls=np.arange(1,10)):
    if m.sum()==0: return {"hit":np.nan,"f1":np.nan,"auprc":np.nan,"conf":None}
    p,l,pr = P[m],L[m],PR[m]
    hit=accuracy_score(l,p); f1=f1_score(l,p,average="macro")
    try:
        au=average_precision_score(label_binarize(l,classes=cls),pr[:,1:10],average="macro")
    except ValueError: au=np.nan
    return {"hit":hit,"f1":f1,"auprc":au,"conf":confusion_matrix(l,p)}

def pprint(tag,d): print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

# ────────────────────────── trainer ─────────────────────────────────────
def train_model(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slots = cfg["seq_len_ai"]//cfg["ai_rate"]
    uid = f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}"
    ck_local = Path(cfg["model_folder"])/f"DecisionOnly_{uid}.pt"
    js_local = ck_local.with_suffix(".json")

    s3c = get_s3(); bkt=cfg["s3_bucket"]
    ck_key=f"DecisionOnly/checkpoints/{ck_local.name}"
    js_key=f"DecisionOnly/metrics/{js_local.name}"
    print(f"[INFO] artefacts will be saved to\n  • s3://{bkt}/{ck_key}\n  • s3://{bkt}/{js_key}\n")

    tr,va,te,tok = dataloaders(cfg); pad=tok.token_to_id("[PAD]")
    eng,_ ,_,_ = deepspeed.initialize(
        model=model(cfg), model_parameters=model(cfg).parameters(),
        config={"train_micro_batch_size_per_gpu":cfg["batch_size"],
                "zero_allow_untested_optimizer":True,
                "optimizer":{"type":"Lamb","params":{"lr":cfg["lr"],"eps":cfg["eps"],
                                                     "weight_decay":cfg["weight_decay"]}},
                "zero_optimization":{"stage":1},"fp16":{"enabled":False}})
    loss_fn=RevenueLoss([0,1,10,1,10,1,10,1,10,0],cfg["vocab_size_tgt"],pad)

    best=None; patience=0
    for ep in range(cfg["num_epochs"]):
        eng.train(); s=0.
        for b in tqdm(tr,desc=f"Ep {ep:02d}",leave=False):
            x=b["aggregate_input"].to(dev); y=b["label"].to(dev)
            pos=torch.arange(cfg["ai_rate"]-1,x.size(1),cfg["ai_rate"],device=dev)
            g=eng(x)[:,pos,:]; ytr=y.clone(); ytr[transition_mask(y)]=pad
            loss=loss_fn(g,ytr); eng.zero_grad(); eng.backward(loss); eng.step(); s+=loss.item()
        print(f"\nTrain loss {s/len(tr):.4f}")

        v_loss,v_ppl,m1,m2,m3,m4 = metrics(va,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for t,d in (("main",m1),("STOP cur",m2),("afterSTOP",m3),("transition",m4)): pprint(t,d)

        if best is None or v_loss<best:
            best=v_loss; patience=0
            torch.save({"epoch":ep,"model_state_dict":eng.module.state_dict()},str(ck_local))
            js_local.write_text(json.dumps(json_safe({
                "best_checkpoint_path":ck_local.name,
                "val_loss":best,"val_ppl":v_ppl,
                "val_main":m1,"val_stop_cur":m2,
                "val_after_stop":m3,"val_transition":m4}),indent=2))
            upload_s3(ck_local,bkt,ck_key,s3c); ck_local.unlink(missing_ok=True)
            upload_s3(js_local,bkt,js_key,s3c); js_local.unlink(missing_ok=True)
        else:
            patience+=1
            if patience>=cfg["patience"]:
                print("Early stopping."); break

    return {"uid":uid,"val_loss":best}

if __name__=="__main__":
    train_model(get_config())
