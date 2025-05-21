# train4_decoderonly_git.py  – silent & JSON-safe
# ===============================================================
import os, logging, warnings, json
from pathlib import Path
import numpy as np
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

# ────────────────────────── suppress noisy logs ──────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPSPEED_LOG_LEVEL"]  = "error"   # ds internal logger
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("torch_checkpoint_engine").setLevel(logging.ERROR)
logging.getLogger("engine").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ────────────────────────── tokenizer (fixed) ────────────────────────────
def _build_tok():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {**{str(i): i for i in range(1, 10)},
             "[PAD]": 0, "[SOS]": 10, "[UNK]": 12}
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ────────────────────────── revenue loss ─────────────────────────────────
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore=0):
        super().__init__()
        rev = torch.as_tensor(revenue + [0.]*(vocab_size-len(revenue)))
        self.register_buffer("pen", -torch.abs(rev[:, None] - rev[None, :]))
        self.ignore = ignore

    def forward(self, logits, tgt):
        B,T,V = logits.shape
        probs = torch.softmax(logits.view(-1, V), dim=-1)
        tgt   = tgt.view(-1)
        keep  = tgt != self.ignore
        if keep.sum() == 0:
            return logits.new_tensor(0.0)
        p = self.pen.to(probs.device)
        return -(probs[keep] * p[tgt[keep]]).sum(dim=-1).mean()

# ────────────────────────── helpers ──────────────────────────────────────
def _transition_mask(y):
    return y != F.pad(y, (1,0), value=-1)[:, :-1]

def _calc_ppl(logits, lbl, pad=0):
    lp = F.log_softmax(logits, dim=-1)
    lp2d, t = lp.view(-1, lp.size(-1)), lbl.view(-1)
    keep = t != pad
    return float("nan") if keep.sum()==0 else \
           torch.exp(F.nll_loss(lp2d[keep], t[keep], reduction="mean")).item()

def _subset(pred,lbl,probs,mask,cls=np.arange(1,10)):
    if mask.sum()==0:
        return {"hit":np.nan,"f1":np.nan,"auprc":np.nan}
    p,l,pr = pred[mask], lbl[mask], probs[mask]
    return {
        "hit": accuracy_score(l,p),
        "f1" : f1_score(l,p,average="macro"),
        "auprc": (average_precision_score(
                    label_binarize(l,classes=cls), pr[:,1:10], average="macro")
                  if len(np.unique(l))>1 else np.nan)
    }

# ────────────────────────── dataloaders ──────────────────────────────────
def _get_dl(cfg):
    data = load_json_dataset(cfg["filepath"])
    n = len(data); tr,va = int(.8*n), int(.1*n)
    s = torch.Generator().manual_seed(33)
    tr_ds,va_ds,te_ds = random_split(data,[tr,va,n-tr-va],generator=s)

    tok_tgt = _build_tok(); tok_ai=_build_tok()
    Path(cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
    tok_tgt.save(str(Path(cfg["model_folder"])/"tok_tgt.json"))
    tok_ai .save(str(Path(cfg["model_folder"])/"tok_ai.json"))

    def mk(split):
        return TransformerDataset(split, tok_ai, tok_tgt,
                                  cfg["seq_len_ai"], cfg["seq_len_tgt"],
                                  cfg["num_heads"], cfg["ai_rate"],0)
    loader = lambda ds,sh: DataLoader(ds,batch_size=cfg["batch_size"],shuffle=sh)
    return loader(mk(tr_ds),True), loader(mk(va_ds),False), \
           loader(mk(te_ds),False), tok_tgt

# ────────────────────────── build model ──────────────────────────────────
def _get_model(cfg):
    return build_transformer(cfg["vocab_size_tgt"], cfg["seq_len_ai"],
                             cfg["d_model"], cfg["N"], cfg["num_heads"],
                             cfg["d_ff"], cfg["dropout"])

# ────────────────────────── evaluation ───────────────────────────────────
def _eval(loader, eng, dev, loss_fn, step, pad, tok):
    if not len(loader): return float("nan"),float("nan"),{}
    sp = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}
    tl,tp,Ps, Ls, PRs, Tmask = 0.,0.,[],[],[],[]
    eng.eval()
    with torch.no_grad():
        for b in loader:
            x,y = b["aggregate_input"].to(dev), b["label"].to(dev)
            g   = eng(x)[:, torch.arange(step-1,g.size(1),step,device=dev), :]
            y_e = y.clone(); y_e[_transition_mask(y)] = pad
            tl += loss_fn(g,y_e).item(); tp += _calc_ppl(g,y_e,pad)
            pr  = F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pd  = pr.argmax(1); lb = y.view(-1).cpu().numpy()
            keep= ~np.isin(lb,list(sp))
            Ps.append(pd[keep]); Ls.append(lb[keep]); PRs.append(pr[keep])
            Tmask.append(_transition_mask(y).view(-1).cpu().numpy()[keep])
    P,L,PR = map(np.concatenate,(Ps,Ls,PRs)); Tm = np.concatenate(Tmask)
    res = _subset(P,L,PR,~Tm)
    return tl/len(loader), tp/len(loader), res

# ────────────────────────── train ────────────────────────────────────────
def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slots = cfg["seq_len_ai"] // cfg["ai_rate"]
    uid   = (f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")
    ckpt  = Path(cfg["model_folder"]) / f"ProductGPT_{uid}.pt"

    tr,va,te,tok = _get_dl(cfg)
    pad = tok.token_to_id("[PAD]")

    model   = _get_model(cfg)
    loss_fn = PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],
                                  cfg["vocab_size_tgt"], pad)

    eng,_,_,_ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={"train_micro_batch_size_per_gpu": cfg["batch_size"],
                "zero_allow_untested_optimizer": True,
                "optimizer":{"type":"Lamb","params":{
                    "lr":cfg["lr"],"eps":cfg["eps"],"weight_decay":cfg["weight_decay"]}},
                "fp16":{"enabled":False},"zero_optimization":{"stage":1}})

    best = None; patience = 0
    for ep in range(cfg["num_epochs"]):
        eng.train(); run=0.
        for b in tqdm(tr,desc=f"Ep {ep:02d}",miniters=10):
            x,y = b["aggregate_input"].to(dev), b["label"].to(dev)
            g = eng(x)[:, torch.arange(cfg["ai_rate"]-1,x.size(1),cfg["ai_rate"],device=dev), :]
            ytr = y.clone(); ytr[_transition_mask(y)] = pad
            loss = loss_fn(g,ytr)
            eng.zero_grad(); eng.backward(loss); eng.step()
            run += loss.item()
        print(f"\nTrain loss {run/len(tr):.4f}")

        v_loss,v_ppl,v_main = _eval(va,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
        print(f"Epoch {ep:02d} ValLoss={v_loss:.4f} PPL={v_ppl:.4f} "
              f"Hit={v_main['hit']:.4f} F1={v_main['f1']:.4f}")

        if best is None or v_loss < best:
            best, patience = v_loss, 0
            torch.save(eng.module.state_dict(), ckpt)
        else:
            patience +=1
            if patience>=cfg["patience"]: break

    eng.module.load_state_dict(torch.load(ckpt,map_location=dev))
    t_loss,t_ppl,t_main = _eval(te,eng,dev,loss_fn,cfg["ai_rate"],pad,tok)
    print(f"\n** TEST ** Loss={t_loss:.4f} PPL={t_ppl:.4f} "
          f"Hit={t_main['hit']:.4f} F1={t_main['f1']:.4f}")

    meta = {"best_checkpoint_path": str(ckpt),
            "val": {k: (v.tolist() if isinstance(v,np.ndarray) else v)
                    for k,v in {"loss":best,"ppl":v_ppl,**v_main}.items()},
            "test":{k: (v.tolist() if isinstance(v,np.ndarray) else v)
                    for k,v in {"loss":t_loss,"ppl":t_ppl,**t_main}.items()}}
    with ckpt.with_suffix(".json").open("w") as f:
        json.dump(meta,f,indent=2)
    return meta

if __name__ == "__main__":
    cfg = get_config()
    train_model(cfg)
