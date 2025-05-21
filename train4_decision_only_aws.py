# train4_decision_only_aws.py
# =======================================================================
# Decision-Only trainer – quiet – PairwiseRevenueLoss
# • Validation metrics on FOUR groups
#     ① all         – every decision position
#     ② main        – decision positions EXCEPT transitions
#     ③ STOP        – positions whose *current* token is 9
#     ④ transition  – token ≠ previous token
# • Best checkpoint & JSON pushed to S3 (then wiped locally)
# =======================================================================

import os, warnings, logging, json, numpy as np, boto3, botocore
from pathlib import Path
os.environ.update({"TF_CPP_MIN_LOG_LEVEL":"3",
                   "DEEPSPEED_LOG_LEVEL":"error",
                   "DS_DISABLE_LOGS":"1"})
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for n in ("deepspeed","torch_checkpoint_engine","engine"):
    logging.getLogger(n).setLevel(logging.ERROR)
    logging.getLogger(n).propagate=False

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

from model4_decoderonly     import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers
from config4_decision_only_git import get_config

# ───────────────────────── tokeniser ─────────────────────────────────────
def _build_tok():
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.model = models.WordLevel(
        {**{str(i): i for i in range(1,10)}, "[PAD]":0, "[SOS]":10, "[UNK]":12},
        unk_token="[UNK]")
    return t

# ───────────────────────── loss ──────────────────────────────────────────
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue += [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("pen", -torch.abs(rev[:,None]-rev[None,:]))
        self.ignore = ignore_index
    def forward(self, logit, tgt):
        V = logit.size(-1)
        p = F.softmax(logit.view(-1,V), dim=-1)
        tgt = tgt.view(-1); keep = tgt!=self.ignore
        if keep.sum()==0: return logit.new_tensor(0.0)
        return -(p[keep]*self.pen.to(p)[tgt[keep]]).sum(1).mean()

# ───────────────────────── misc helpers ──────────────────────────────────
def _transition_mask(y):      return y != F.pad(y,(1,0),value=-1)[:,:-1]
def _ppl(logit,tgt,pad):
    lp = F.log_softmax(logit, dim=-1)
    lp2,t = lp.view(-1,lp.size(-1)), tgt.view(-1)
    m = t!=pad
    return float("nan") if m.sum()==0 else torch.exp(F.nll_loss(lp2[m],t[m])).item()

def _subset(P,L,PR,mask, cls=np.arange(1,10)):
    if mask.sum()==0:
        return {"hit":np.nan,"f1":np.nan,"auprc":np.nan}
    p,l,pr = P[mask],L[mask],PR[mask]
    hit = accuracy_score(l,p); f1=f1_score(l,p,average="macro")
    try:
        au = average_precision_score(label_binarize(l,classes=cls),
                                     pr[:,1:10],average="macro")
    except ValueError: au=np.nan
    return {"hit":hit,"f1":f1,"auprc":au}

def _json_safe(o):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)): return o.cpu().tolist()
    if isinstance(o, _np.ndarray): return o.tolist()
    if isinstance(o, (_np.generic,)): return o.item()
    if isinstance(o, dict):  return {k:_json_safe(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [_json_safe(v) for v in o]
    return o

# ───────────────────────── S3 helpers ────────────────────────────────────
def _s3():  # singleton
    if not hasattr(_s3,"c"):
        try: _s3.c = boto3.client("s3")
        except botocore.exceptions.BotoCoreError: _s3.c=None
    return _s3.c
def _up(local,bucket,key):
    c=_s3(); 
    if c and local.exists():
        try: c.upload_file(str(local), bucket, key)
        except botocore.exceptions.BotoCoreError as e: print("[S3] err",e)
        else: print(f"[S3] {local.name} → s3://{bucket}/{key}")

# ───────────────────────── data & model ──────────────────────────────────
def _loaders(cfg):
    data = load_json_dataset(cfg["filepath"]); n=len(data)
    tr,va = int(.8*n), int(.1*n)
    tr_ds,va_ds,te_ds = random_split(data,[tr,va,n-tr-va],
                                     generator=torch.Generator().manual_seed(33))
    tok_ai = tok_tgt = _build_tok()
    out=Path(cfg["model_folder"]); out.mkdir(parents=True, exist_ok=True)
    tok_ai.save (str(out/"tokenizer_ai.json"))
    tok_tgt.save(str(out/"tokenizer_tgt.json"))
    mk=lambda s:TransformerDataset(s,tok_ai,tok_tgt,
              cfg["seq_len_ai"],cfg["seq_len_tgt"],
              cfg["num_heads"],cfg["ai_rate"],pad_token=0)
    ld=lambda d,sh:DataLoader(d,batch_size=cfg["batch_size"],shuffle=sh)
    return ld(mk(tr_ds),True),ld(mk(va_ds),False),ld(mk(te_ds),False),tok_tgt

def _model(cfg):
    return build_transformer(cfg["vocab_size_tgt"],cfg["seq_len_ai"],
                             cfg["d_model"],cfg["N"],cfg["num_heads"],
                             cfg["d_ff"],cfg["dropout"])

# ───────────────────────── evaluation (now +“all”) ───────────────────────
def _eval(loader, eng, dev, loss_fn, step, pad, tok):
    if not loader: nan=float("nan"); return nan,nan,{},{},{},{}
    sp={pad,tok.token_to_id("[SOS]"),tok.token_to_id("[UNK]")}
    tl,tp=0.,0.; P=[];L=[];PR=[]; m_stop=[];m_tr=[]
    eng.eval()
    with torch.no_grad():
        for b in loader:
            x,y = b["aggregate_input"].to(dev), b["label"].to(dev)
            pos = torch.arange(step-1,x.size(1),step,device=dev)
            g   = eng(x)[:,pos,:]
            tgt = y[:,pos].clone(); tgt[_transition_mask(y)[:,pos]] = pad
            tl += loss_fn(g,tgt).item(); tp += _ppl(g,tgt,pad)

            prob = F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pred = prob.argmax(1); lbl = tgt.view(-1).cpu().numpy()
            keep = ~np.isin(lbl,list(sp))

            P.append(pred[keep]); L.append(lbl[keep]); PR.append(prob[keep])
            m_stop.append((tgt==9).view(-1).cpu().numpy()[keep])
            m_tr  .append(_transition_mask(y)[:,pos].view(-1).cpu().numpy()[keep])

    P,L,PR = map(np.concatenate,(P,L,PR))
    m_stop,m_tr = map(np.concatenate,(m_stop,m_tr))
    all_mask   = np.ones_like(P,dtype=bool)
    main_mask  = ~m_tr
    return (tl/len(loader), tp/len(loader),
            _subset(P,L,PR, all_mask),
            _subset(P,L,PR, main_mask),
            _subset(P,L,PR, m_stop),
            _subset(P,L,PR, m_tr))

# ───────────────────────── train ─────────────────────────────────────────
def train_model(cfg):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slots=cfg["seq_len_ai"]//cfg["ai_rate"]
    uid=f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}"
    ck = Path(cfg["model_folder"])/f"DecisionOnly_{uid}.pt"
    js = ck.with_suffix(".json")
    bucket=cfg["s3_bucket"]; ck_key=f"DecisionOnly/checkpoints/{ck.name}"
    js_key=f"DecisionOnly/metrics/{js.name}"
    print(f"[INFO] artefacts → s3://{bucket}/{ck_key}\n"
          f"                     s3://{bucket}/{js_key}\n")

    tr,va,te,tok = _loaders(cfg); pad=tok.token_to_id("[PAD]")
    loss_fn=PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],
                                cfg["vocab_size_tgt"],pad)
    eng,_,_,_=deepspeed.initialize(model=_model(cfg),
        model_parameters=_model(cfg).parameters(),
        config={"train_micro_batch_size_per_gpu":cfg["batch_size"],
                "zero_allow_untested_optimizer":True,
                "optimizer":{"type":"Lamb","params":{"lr":cfg["lr"],
                    "eps":cfg["eps"],"weight_decay":cfg["weight_decay"]}},
                "zero_optimization":{"stage":1},"fp16":{"enabled":False}})

    best=None; patience=0
    for ep in range(cfg["num_epochs"]):
        eng.train(); run=0.
        for b in tqdm(tr,desc=f"Ep {ep:02d}",leave=False):
            x,y = b["aggregate_input"].to(dev), b["label"].to(dev)
            pos = torch.arange(cfg["ai_rate"]-1,x.size(1),cfg["ai_rate"],device=dev)
            g   = eng(x)[:,pos,:]
            tgt = y[:,pos].clone(); tgt[_transition_mask(y)[:,pos]]=pad
            ls  = loss_fn(g,tgt)
            eng.zero_grad(); eng.backward(ls); eng.step(); run+=ls.item()
        print(f"\nTrain loss {run/len(tr):.4f}")

        v_loss,v_ppl,v_all,v_main,v_stop,v_tr = _eval(
            va,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag,d in (("all",v_all),("main",v_main),
                      ("STOP",v_stop),("transition",v_tr)):
            print(f"  {tag:<10} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

        if best is None or v_loss<best:
            best,patience = v_loss,0
            torch.save({"epoch":ep,"model_state_dict":eng.module.state_dict()}, ck)
            meta=_json_safe({"best_checkpoint_path":ck.name,
                             "val_loss":best,"val_ppl":v_ppl,
                             "val_all":v_all,"val_main":v_main,
                             "val_stop":v_stop,"val_transition":v_tr})
            js.write_text(json.dumps(meta,indent=2))
            _up(ck,bucket,ck_key); ck.unlink(missing_ok=True)
            _up(js,bucket,js_key); js.unlink(missing_ok=True)
        else:
            patience+=1
            if patience>=cfg["patience"]:
                print("Early stopping."); break

    t_loss,t_ppl,t_all,t_main,t_stop,t_tr = _eval(
        te,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
    print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
    for tag,d in (("all",t_all),("main",t_main),
                  ("STOP",t_stop),("transition",t_tr)):
        print(f"  {tag:<10} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
              f"AUPRC={d['auprc']:.4f}")

    return {"uid":uid,"val_loss":best}

# ───────────────────────── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    train_model(get_config())
