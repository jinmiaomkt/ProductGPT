# train4_decision_only_aws.py
# =======================================================================
# Decision-Only training (quiet) + automatic S3 upload
# =======================================================================

# ─── env / logging ──────────────────────────────────────────────────────
import os, warnings, logging, json, numpy as np
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPSPEED_LOG_LEVEL"]  = "error"
os.environ["DS_DISABLE_LOGS"]      = "1"
warnings.filterwarnings("ignore")
for n in ["deepspeed", "torch_checkpoint_engine", "engine"]:
    logging.getLogger(n).setLevel(logging.ERROR)
    logging.getLogger(n).propagate = False

# ─── std / 3rd-party ────────────────────────────────────────────────────
import boto3, botocore
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb                        # optimiser

from model4_decoderonly     import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers
from config4_decision_only_git import get_config
# ======================================================================


# ══════════════════════════════════════════════════════════════════════
#                              utilities
# ══════════════════════════════════════════════════════════════════════
def s3_client():
    try:
        return boto3.client("s3")
    except botocore.exceptions.BotoCoreError:
        return None

def s3_upload(local: Path, bucket: str, key: str, s3) -> bool:
    if s3 is None:
        return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] uploaded → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-ERROR] {e}")
        return False

def json_safe(x):
    """recursively convert numpy / torch objects → JSON-serialisable types"""
    import torch, numpy as np
    if isinstance(x, (torch.Tensor, torch.nn.Parameter)):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    return x
# ----------------------------------------------------------------------


# ══════════════════════════════════════════════════════════════════════
#                       tokeniser & loss
# ══════════════════════════════════════════════════════════════════════
def build_tokenizer_tgt():
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.model = models.WordLevel(
        vocab={ "[PAD]":0, "1":1, "2":2, "3":3, "4":4, "5":5,
                "6":6, "7":7, "8":8, "9":9, "[SOS]":10, "[UNK]":12},
        unk_token="[UNK]")
    return tk

class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue += [0.] * (vocab_size-len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty", -torch.abs(rev[:,None]-rev[None,:]))
        self.ignore_index = ignore_index

    def forward(self, logits, tgt):
        B,T,V = logits.shape
        probs = torch.softmax(logits.view(-1,V), dim=-1)
        tgt   = tgt.view(-1)
        keep  = tgt != self.ignore_index
        if keep.sum() == 0:
            return logits.new_tensor(0.0)
        gap = (probs[keep] * self.penalty.to(probs.device)[tgt[keep]]).sum(1)
        return (-gap).mean()
# ----------------------------------------------------------------------


# ══════════════════════════════════════════════════════════════════════
#                        data / model helpers
# ══════════════════════════════════════════════════════════════════════
def transition_mask(lbl):   # (B,T) Bool where decision changes
    return lbl != F.pad(lbl,(1,0),value=-1)[:,:-1]

def perplexity(logits, tgt, pad):
    lp = F.log_softmax(logits, dim=-1)
    lp2,t = lp.view(-1,lp.size(-1)), tgt.view(-1)
    keep = t!=pad
    if keep.sum()==0: return float("nan")
    return torch.exp(F.nll_loss(lp2[keep], t[keep])).item()

def subset_metrics(pred,lbl,probs,mask, cls=np.arange(1,10)):
    if mask.sum()==0:
        return dict(hit=float("nan"),f1=float("nan"),auprc=float("nan"),conf=None)
    p,l,pr = pred[mask], lbl[mask], probs[mask]
    return dict(
        hit   = accuracy_score(l,p),
        f1    = f1_score(l,p,average="macro"),
        auprc = average_precision_score(
                    label_binarize(l, classes=cls), pr[:,1:10], average="macro"
                ) if len(np.unique(l))>1 else float("nan"),
        conf  = confusion_matrix(l,p,labels=np.unique(l))
    )

def get_dataloaders(cfg):
    raw = load_json_dataset(cfg["filepath"])
    n   = len(raw); tr,va = int(.8*n), int(.1*n)
    torch.manual_seed(33)
    tr_ds,va_ds,te_ds = random_split(raw,[tr,va,n-tr-va],
        generator=torch.Generator().manual_seed(33))

    tok_ai  = build_tokenizer_tgt()
    tok_tgt = build_tokenizer_tgt()
    Path(cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
    tok_ai .save(str(Path(cfg["model_folder"])/"tokenizer_ai.json"))
    tok_tgt.save(str(Path(cfg["model_folder"])/"tokenizer_tgt.json"))

    mkds = lambda ds: TransformerDataset(ds,tok_ai,tok_tgt,
                                         cfg["seq_len_ai"],cfg["seq_len_tgt"],
                                         cfg["num_heads"],cfg["ai_rate"],0)
    mkdl = lambda ds,sh: DataLoader(mkds(ds),batch_size=cfg["batch_size"],shuffle=sh)
    return mkdl(tr_ds,True), mkdl(va_ds,False), mkdl(te_ds,False), tok_tgt

def build_model(cfg):
    return build_transformer(cfg["vocab_size_tgt"],
                             cfg["seq_len_ai"],
                             cfg["d_model"], cfg["N"],
                             cfg["num_heads"], cfg["d_ff"],
                             cfg["dropout"])

def evaluate(dl, engine, dev, loss_fn, step, pad, tok):
    if not dl: n=float("nan"); return n,n,{}, {}, {}, {}
    special={pad,tok.token_to_id("[SOS]"),tok.token_to_id("[UNK]")}
    tl,tp=0.0,0.0; P,L,PR=[],[],[]; mS,mP,mT=[],[],[]
    engine.eval()
    with torch.no_grad():
        for b in dl:
            x,y=b["aggregate_input"].to(dev), b["label"].to(dev)
            g=engine(x)[:,torch.arange(step-1,g:=engine(x).size(1),step,device=dev),:]
            tl+=loss_fn(g,y).item(); tp+=perplexity(g,y,pad)
            pr = F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pd = pr.argmax(1); lb = y.view(-1).cpu().numpy()
            v  = ~np.isin(lb,list(special))
            P.append(pd[v]); L.append(lb[v]); PR.append(pr[v])
            mS.append((y==9).view(-1).cpu().numpy()[v])
            mP.append((F.pad(y,(1,0),value=-1)[:,:-1]==9).view(-1).cpu().numpy()[v])
            mT.append(transition_mask(y).view(-1).cpu().numpy()[v])
    P,L,PR=map(np.concatenate,(P,L,PR)); mS,mP,mT=map(np.concatenate,(mS,mP,mT))
    return (tl/len(dl), tp/len(dl),
            subset_metrics(P,L,PR,~mT),
            subset_metrics(P,L,PR,mS),
            subset_metrics(P,L,PR,mP),
            subset_metrics(P,L,PR,mT))
# ----------------------------------------------------------------------


# ══════════════════════════════════════════════════════════════════════
#                         main training loop
# ══════════════════════════════════════════════════════════════════════
def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slots = cfg["seq_len_ai"] // cfg["ai_rate"]
    uid   = (f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")

    ckpt_local = Path(cfg["model_folder"])/f"DecisionOnly_{uid}.pt"
    json_local = ckpt_local.with_suffix(".json")

    s3       = s3_client()
    bucket   = cfg["s3_bucket"]
    prefix   = (cfg.get("s3_prefix") or "").rstrip("/")
    if prefix: prefix+="/"
    ckpt_key = f"{prefix}DecisionOnly/checkpoints/{ckpt_local.name}"
    json_key = f"{prefix}DecisionOnly/metrics/{json_local.name}"

    print(f"[INFO] artefacts will be saved to\n"
          f"  • s3://{bucket}/{ckpt_key}\n"
          f"  • s3://{bucket}/{json_key}\n")

    tr_dl,va_dl,te_dl,tok=get_dataloaders(cfg); pad=tok.token_to_id("[PAD]")
    loss_fn=PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],cfg["vocab_size_tgt"],pad)
    engine,*_=deepspeed.initialize(
        model=build_model(cfg),
        model_parameters=build_model(cfg).parameters(),
        config={"train_micro_batch_size_per_gpu":cfg["batch_size"],
                "zero_allow_untested_optimizer":True,
                "optimizer":{"type":"Lamb","params":{"lr":cfg["lr"],
                                                     "eps":cfg["eps"],
                                                     "weight_decay":cfg["weight_decay"]}},
                "fp16":{"enabled":False},
                "zero_optimization":{"stage":1}}
    )

    best=None; patience=0
    for ep in range(cfg["num_epochs"]):
        # ◼ train --------------------------------------------------------
        engine.train(); running=0.0
        for b in tqdm(tr_dl,desc=f"Ep {ep:02d}"):
            x,y=b["aggregate_input"].to(dev), b["label"].to(dev)
            pos=torch.arange(cfg["ai_rate"]-1,x.size(1),cfg["ai_rate"],device=dev)
            g=engine(x)[:,pos,:]
            y_tr=y.clone(); y_tr[transition_mask(y)]=pad
            loss=loss_fn(g,y_tr)
            engine.zero_grad(); engine.backward(loss); engine.step()
            running+=loss.item()
        print(f"\nTrain loss {running/len(tr_dl):.4f}")

        # ◼ validate -----------------------------------------------------
        v_loss,v_ppl,*metrics = evaluate(va_dl,engine,dev,loss_fn,cfg["ai_rate"],pad,tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag,d in zip(["main","STOP cur","afterSTOP","transition"],metrics):
            print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

        # ◼ checkpoint if better ----------------------------------------
        if best is None or v_loss<best:
            best,patience=v_loss,0
            torch.save({"epoch":ep,"model_state_dict":engine.module.state_dict()},
                       str(ckpt_local))                 # <-- str!

            meta=dict(best_checkpoint_path=str(ckpt_local.name),
                      val_loss=best,val_ppl=v_ppl,
                      val_main=json_safe(metrics[0]),
                      val_stop_cur=json_safe(metrics[1]),
                      val_after_stop=json_safe(metrics[2]),
                      val_transition=json_safe(metrics[3]))
            with json_local.open("w") as fp:
                json.dump(meta,fp,indent=2)

            if s3_upload(ckpt_local,bucket,ckpt_key,s3):
                ckpt_local.unlink(missing_ok=True)
            if s3_upload(json_local,bucket,json_key,s3):
                json_local.unlink(missing_ok=True)
        else:
            patience+=1
            if patience>=cfg["patience"]:
                print("Early stopping."); break

    # ◼ test -------------------------------------------------------------
    if ckpt_local.exists(): state=torch.load(str(ckpt_local),map_location=dev)
    elif s3 is not None:
        tmp=Path("/tmp")/ckpt_local.name
        try: s3.download_file(bucket,ckpt_key,str(tmp)); state=torch.load(str(tmp),map_location=dev)
        except botocore.exceptions.BotoCoreError: state=None
    else: state=None
    if state and "model_state_dict" in state:
        engine.module.load_state_dict(state["model_state_dict"])
    t_loss,t_ppl,*_=evaluate(te_dl,engine,dev,loss_fn,cfg["ai_rate"],pad,tok)
    print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
    return best            # safe scalar for the tuning driver

# ══════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    train_model(get_config())
