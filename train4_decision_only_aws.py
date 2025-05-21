"""
Decision-Only trainer – silent + S3 upload
"""
# ────────────────── mute everything from DeepSpeed / root logging ────────
import os, warnings, logging, json, numpy as np, boto3, botocore
from pathlib import Path
for k, v in {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "DEEPSPEED_LOG_LEVEL" : "error",
    "DS_DISABLE_LOGS"     : "1",
}.items():
    os.environ[k] = v

warnings.filterwarnings("ignore")
# root logger to ERROR keeps our own print() intact
logging.getLogger().setLevel(logging.ERROR)
for name in (
    "deepspeed", "deepspeed.utils", "deepspeed.runtime.utils",
    "deepspeed.runtime.config", "deepspeed.runtime.zero.stage_1_and_2",
):
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

# ────────────────── std / 3rd-party ──────────────────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

from model4_decoderonly     import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers
from config4_decision_only_git import get_config

# ────────────────── tiny helpers ─────────────────────────────────────────
def json_safe(o):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)):
        return o.detach().cpu().tolist()
    if isinstance(o, _np.ndarray):      return o.tolist()
    if isinstance(o, (_np.generic,)):   return o.item()
    if isinstance(o, dict):             return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):    return [json_safe(v) for v in o]
    return o

def get_s3():
    try: return boto3.client("s3")
    except botocore.exceptions.BotoCoreError: return None

def upload(local:Path, bucket:str, key:str, s3) -> bool:
    if s3 is None: return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name} → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-ERR] {e}")
        return False

def tok():
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.model = models.WordLevel(
        {"[PAD]":0, **{str(i):i for i in range(1,10)}, "[SOS]":10, "[UNK]":12},
        unk_token="[UNK]")
    return t

def transition_mask(lbl):               # (B,T)
    return lbl != F.pad(lbl,(1,0),value=-1)[:,:-1]

def ppl(logits, tgt, pad):
    lp = F.log_softmax(logits,dim=-1)
    lp2d, t = lp.view(-1,lp.size(-1)), tgt.view(-1)
    m = t != pad
    return float("nan") if m.sum()==0 else \
        torch.exp(F.nll_loss(lp2d[m],t[m],reduction="mean")).item()

# ────────────────── loss ────────────────────────────────────────────────
class RevenueLoss(nn.Module):
    def __init__(self, rev, V, ignore=0):
        super().__init__()
        rev = rev+[0.]*(V-len(rev)) if len(rev)<V else rev
        r = torch.tensor(rev); self.register_buffer(
            "pen", -torch.abs(r[:,None]-r[None,:]))
        self.ignore = ignore
    def forward(self, logit, tgt):
        B,T,V = logit.shape
        p = F.softmax(logit.view(-1,V),dim=-1); t = tgt.view(-1)
        keep = t!=self.ignore
        if keep.sum()==0: return logit.new_tensor(0.0)
        return -(p[keep]*self.pen.to(p)[t[keep]]).sum(1).mean()

# ────────────────── data / model ────────────────────────────────────────
def loaders(cfg):
    raw=load_json_dataset(cfg["filepath"]); n=len(raw); tr,va=int(.8*n),int(.1*n)
    tr_ds,va_ds,te_ds = random_split(raw,[tr,va,n-tr-va],
        generator=torch.Generator().manual_seed(33))
    tok_ai = tok_tgt = tok()
    out=Path(cfg["model_folder"]); out.mkdir(parents=True,exist_ok=True)
    tok_ai.save (str(out/"tokenizer_ai.json"));  tok_tgt.save(str(out/"tokenizer_tgt.json"))
    mk=lambda s:TransformerDataset(s,tok_ai,tok_tgt,cfg["seq_len_ai"],
            cfg["seq_len_tgt"],cfg["num_heads"],cfg["ai_rate"],pad_token=0)
    data=lambda d,sh:DataLoader(d,batch_size=cfg["batch_size"],shuffle=sh)
    return data(mk(tr_ds),True),data(mk(va_ds),False),data(mk(te_ds),False),tok_tgt

def net(cfg):
    return build_transformer(cfg["vocab_size_tgt"], cfg["seq_len_ai"],
                             cfg["d_model"],cfg["N"],cfg["num_heads"],
                             cfg["d_ff"],cfg["dropout"])

# ────────────────── validation ──────────────────────────────────────────
def subset(P,L,PR,mask, cls=np.arange(1,10)):
    if mask.sum()==0: return {"hit":np.nan,"f1":np.nan,"auprc":np.nan,"conf":None}
    p,l,pr=P[mask],L[mask],PR[mask]
    hit=accuracy_score(l,p); f1=f1_score(l,p,average="macro")
    try: au=average_precision_score(label_binarize(l,classes=cls), pr[:,1:10], average="macro")
    except ValueError: au=np.nan
    return {"hit":hit,"f1":f1,"auprc":au,"conf":confusion_matrix(l,p)}

def val(loader, eng, dev, loss_fn, step, pad, tok):
    if not loader: nan=float("nan"); return nan,nan,{}, {}, {}, {}
    sp={pad,tok.token_to_id("[SOS]"),tok.token_to_id("[UNK]")}
    tl,tp=0.,0.; P=[];L=[];PR=[]; ms=[];mp=[];mt=[]
    eng.eval()
    with torch.no_grad():
        for b in loader:
            x=b["aggregate_input"].to(dev); y=b["label"].to(dev)
            g=eng(x)[:,torch.arange(step-1,g.size(1),step,device=dev),:]
            y_e=y.clone(); y_e[transition_mask(y)]=pad
            tl+=loss_fn(g,y_e).item(); tp+=ppl(g,y_e,pad)
            pr=F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pd=pr.argmax(1); lb=y.view(-1).cpu().numpy(); v=~np.isin(lb,list(sp))
            P.append(pd[v]);L.append(lb[v]);PR.append(pr[v])
            ms.append((y==9).view(-1).cpu().numpy()[v])
            mp.append((F.pad(y,(1,0),value=-1)[:,:-1]==9).view(-1).cpu().numpy()[v])
            mt.append(transition_mask(y).view(-1).cpu().numpy()[v])
    P,L,PR=np.concatenate(P),np.concatenate(L),np.concatenate(PR)
    ms,mp,mt = map(np.concatenate, (ms,mp,mt))
    return tl/len(loader), tp/len(loader), \
           subset(P,L,PR,~mt), subset(P,L,PR,ms), subset(P,L,PR,mp), subset(P,L,PR,mt)

# ────────────────── main trainer ─────────────────────────────────────────
def train_model(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slots=cfg["seq_len_ai"]//cfg["ai_rate"]
    uid=f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}"
    ck=Path(cfg["model_folder"])/f"DecisionOnly_{uid}.pt"
    js=ck.with_suffix(".json")
    s3c=get_s3(); bucket=cfg["s3_bucket"]
    ck_key=f"DecisionOnly/checkpoints/{ck.name}"
    js_key=f"DecisionOnly/metrics/{js.name}"
    print(f"[INFO] artefacts will be saved to\n  • s3://{bucket}/{ck_key}\n  • s3://{bucket}/{js_key}\n")

    tr,va,te,tok = loaders(cfg); pad=tok.token_to_id("[PAD]")
    eng,_,_,_=deepspeed.initialize(
        model=net(cfg), model_parameters=net(cfg).parameters(),
        config={"train_micro_batch_size_per_gpu":cfg["batch_size"],
                "zero_allow_untested_optimizer":True,
                "optimizer":{"type":"Lamb","params":{"lr":cfg["lr"],
                    "eps":cfg["eps"],"weight_decay":cfg["weight_decay"]}},
                "zero_optimization":{"stage":1}, "fp16":{"enabled":False}})
    loss_fn=RevenueLoss([0,1,10,1,10,1,10,1,10,0],cfg["vocab_size_tgt"],pad)

    best=None; patience=0
    for ep in range(cfg["num_epochs"]):
        eng.train(); run=0.
        for b in tqdm(tr,desc=f"Ep {ep:02d}",leave=False):
            x=b["aggregate_input"].to(dev); y=b["label"].to(dev)
            pos=torch.arange(cfg["ai_rate"]-1,x.size(1),cfg["ai_rate"],device=dev)
            g=eng(x)[:,pos,:]; y_t=y.clone(); y_t[transition_mask(y)]=pad
            ls=loss_fn(g,y_t); eng.zero_grad(); eng.backward(ls); eng.step(); run+=ls.item()
        print(f"\nTrain loss {run/len(tr):.4f}")

        v_loss,v_ppl,m1,m2,m3,m4 = val(va,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag,d in (("main",m1),("STOP cur",m2),("afterSTOP",m3),("transition",m4)):
            print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

        if best is None or v_loss<best:
            best=v_loss; patience=0
            torch.save({"epoch":ep,"model_state_dict":eng.module.state_dict()}, str(ck))
            js.write_text(json.dumps(json_safe({
                "best_checkpoint_path":ck.name,"val_loss":best,"val_ppl":v_ppl,
                "val_main":m1,"val_stop_cur":m2,"val_after_stop":m3,"val_transition":m4}),indent=2))
            upload(ck,bucket,ck_key,s3c); ck.unlink(missing_ok=True)
            upload(js,bucket,js_key,s3c); js.unlink(missing_ok=True)
        else:
            patience+=1
            if patience>=cfg["patience"]: print("Early stopping."); break

    # ---------- optional: test set (only if we kept a local ckpt) -------
    if ck.exists():
        state=torch.load(str(ck),map_location=dev); eng.module.load_state_dict(state["model_state_dict"])
        t_loss,t_ppl,m1,m2,m3,m4 = val(te,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
        print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")

    return {"uid":uid,"val_loss":best}

if __name__=="__main__":
    train_model(get_config())
