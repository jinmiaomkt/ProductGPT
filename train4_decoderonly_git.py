import os, json, logging, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from config4_decision_only_git import get_config, get_weights_file_path

# ─────────────────────────── env / logging ────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# ─────────────────────────── 1 ─ tokenizer ────────────────────────────────
def build_tokenizer_tgt():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {"[PAD]": 0,
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
             "6": 6, "7": 7, "8": 8, "9": 9,
             "[SOS]": 10, "[UNK]": 12}
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ─────────────────────────── 2 ─ revenue loss ─────────────────────────────
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue = revenue + [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty",
                             -torch.abs(rev[:, None] - rev[None, :]))  # V×V
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        tgt   = targets.reshape(-1)
        m     = tgt != self.ignore_index
        if m.sum() == 0:
            return logits.new_tensor(0.0)
        exp_gap = (probs[m] * self.penalty[tgt[m]]).sum(dim=-1)
        return (-exp_gap).mean()

# ─────────────────────────── 3 ─ helpers ──────────────────────────────────
def transition_mask(lbl: torch.Tensor):
    prev = F.pad(lbl, (1,0), value=-1)[:, :-1]
    return lbl != prev

def calc_perplexity(logits, tgt, pad=0):
    logp = F.log_softmax(logits, dim=-1)
    lp2d, t = logp.reshape(-1, logp.size(-1)), tgt.reshape(-1)
    m = t != pad
    if m.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[m], t[m], reduction="mean")).item()

def _pp_subset(tag, d):
    print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

# ─────────────────────────── 4 ─ dataloaders ──────────────────────────────
def get_loaders(cfg):
    data = load_json_dataset(cfg["filepath"])
    n = len(data); tr, va = int(.8*n), int(.1*n)
    torch.manual_seed(33)
    tr_ds, va_ds, te_ds = random_split(
        data, [tr, va, n-tr-va],
        generator=torch.Generator().manual_seed(33))

    tok_tgt = build_tokenizer_tgt(); tok_ai = build_tokenizer_tgt()
    Path(cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
    tok_tgt.save(str(Path(cfg["model_folder"])/"tokenizer_tgt.json"))
    tok_ai .save(str(Path(cfg["model_folder"])/"tokenizer_ai.json"))

    def mk(split):
        return TransformerDataset(split, tok_ai, tok_tgt,
                                  cfg['seq_len_ai'], cfg['seq_len_tgt'],
                                  cfg['num_heads'], cfg['ai_rate'], pad_token=0)
    mk_loader = lambda d, sh: DataLoader(d, batch_size=cfg['batch_size'], shuffle=sh)
    return (mk_loader(mk(tr_ds), True),
            mk_loader(mk(va_ds), False),
            mk_loader(mk(te_ds), False),
            tok_tgt)

# ─────────────────────────── 5 ─ model ────────────────────────────────────
def get_model(cfg):
    return build_transformer(cfg['vocab_size_tgt'], 
                             cfg['seq_len_ai'], 
                             cfg['d_model'],
                             cfg['N'], 
                             cfg['num_heads'],
                             cfg['d_ff'], 
                             cfg['dropout'])

# ─────────────────────────── 6 ─ evaluation ───────────────────────────────
def _subset_metrics(pred, lbl, probs, mask, cls=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": float("nan"), "f1": float("nan"),
                "auprc": float("nan"), "conf": None}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    conf = confusion_matrix(l, p, labels=np.unique(l))
    hit  = accuracy_score(l, p)
    f1   = f1_score(l, p, average='macro')
    try:
        y_true = label_binarize(l, classes=cls)
        auprc  = average_precision_score(y_true, pr[:,1:10], average='macro')
    except ValueError:
        auprc = float("nan")
    return {"hit": hit, "f1": f1, "auprc": auprc, "conf": conf}

def evaluate(loader, engine, device, loss_fn, step, pad_id, tok):
    if len(loader) == 0:
        nan=float('nan'); return nan,nan,{}, {}, {}, {}
    special = {pad_id, tok.token_to_id('[SOS]'), tok.token_to_id('[UNK]')}
    tloss = tppl = 0.0
    P,L,PR = [],[],[]
    m_stop,m_prev,m_tr = [],[],[]

    engine.eval()
    with torch.no_grad():
        for b in loader:
            x,y = b['aggregate_input'].to(device), b['label'].to(device)
            g   = engine(x)
            pos = torch.arange(step-1, g.size(1), step, device=device)
            g   = g[:,pos,:]
            y_eval = y.clone(); y_eval[transition_mask(y)] = pad_id
            tloss += loss_fn(g, y_eval).item()
            tppl  += calc_perplexity(g, y_eval, pad_id)

            probs = F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pred  = probs.argmax(1)
            lbl   = y.view(-1).cpu().numpy()
            valid = ~np.isin(lbl, list(special))
            P.append(pred[valid]); L.append(lbl[valid]); PR.append(probs[valid])
            m_stop.append((y==9).view(-1).cpu().numpy()[valid])
            m_prev.append((F.pad(y,(1,0),value=-1)[:,:-1]==9).view(-1).cpu().numpy()[valid])
            m_tr  .append(transition_mask(y).view(-1).cpu().numpy()[valid])

    P,L,PR = np.concatenate(P),np.concatenate(L),np.concatenate(PR)
    m_stop,m_prev,m_tr = map(np.concatenate,(m_stop,m_prev,m_tr))
    main_mask = ~m_tr
    return (tloss/len(loader), tppl/len(loader),
            _subset_metrics(P,L,PR,main_mask),
            _subset_metrics(P,L,PR,m_stop),
            _subset_metrics(P,L,PR,m_prev),
            _subset_metrics(P,L,PR,m_tr))

# ─────────────────────────── 7 ─ train ────────────────────────────────────
def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_id = (f"ctx_window{cfg['ctx_window']/cfg['ai_rate']}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
                 f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")
    
    tr_dl, va_dl, te_dl, tok = get_loaders(cfg)
    pad = tok.token_to_id("[PAD]")

    model = get_model(cfg).to(device)
    loss_fn = PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],
                                  cfg['vocab_size_tgt'], pad).to(device)

    steps_total = len(tr_dl)*cfg['num_epochs']
    ds_cfg = {"train_micro_batch_size_per_gpu": cfg['batch_size'],
              "zero_allow_untested_optimizer": True,
              "optimizer": {"type": "Lamb",
                            "params": {"lr": cfg['lr'],
                                       "eps": cfg['eps'],
                                       "weight_decay": cfg['weight_decay']}},
              "lr_scheduler": {"type":"WarmupDecayLR",
                               "params":{"warmup_min_lr":cfg['min_lr'],
                                         "warmup_max_lr":cfg['lr'],
                                         "warmup_num_steps":cfg['warmup_steps'],
                                         "total_num_steps":steps_total,
                                         "decay_style":"cosine"}},
              "fp16":{"enabled":True},
              "zero_optimization":{"stage":1}}
    engine,_,_,_ = deepspeed.initialize(model=model,
                                        model_parameters=model.parameters(),
                                        config=ds_cfg)

    best,pat,ckpt=None,0,None
    for ep in range(cfg['num_epochs']):
        # --- train --------------------------------------------------------
        engine.train(); run=0.0
        for b in tqdm(tr_dl, desc=f"Ep {ep:02d}"):
            x,y=b['aggregate_input'].to(device),b['label'].to(device)
            pos = torch.arange(cfg['ai_rate']-1,x.size(1),cfg['ai_rate'],device=device)
            g   = engine(x)[:,pos,:]
            y_tr=y.clone(); y_tr[transition_mask(y)]=pad
            loss=loss_fn(g,y_tr)
            engine.zero_grad(); engine.backward(loss); engine.step()
            run+=loss.item()
        print(f"\nTrain loss {run/len(tr_dl):.4f}")

        # --- validate -----------------------------------------------------
        v_loss,v_ppl,v_main,v_stop,v_after,v_tr = evaluate(
            va_dl,engine,device,loss_fn,cfg['ai_rate'],pad,tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        _pp_subset("main",v_main); _pp_subset("STOP cur",v_stop)
        _pp_subset("afterSTOP",v_after); _pp_subset("transition",v_tr)

        if best is None or v_loss<best:
            best=v_loss; pat=0
            # ckpt=get_weights_file_path(cfg,'best')
            ckpt = str(Path(cfg["model_folder"]) / f"ProductGPT_{unique_id}.pt")
            engine.save_checkpoint(str(Path(ckpt).parent),tag='best')
            print("  [*] checkpoint saved")
        else:
            pat+=1
            if pat>=cfg['patience']:
                print("Early stop."); break

    # --- test -------------------------------------------------------------
    engine.load_checkpoint(str(Path(ckpt).parent),tag='best')

    # after engine.save_checkpoint(...)
    best_dir   = Path(cfg["model_folder"]) / "best"
    main_pt    = best_dir / "mp_rank_00_model_states.pt"
    pretty_pt = Path(cfg["model_folder"]) / f"ProductGPT_{unique_id}.pt"

    if main_pt.exists():          # keep only the model-state file
        main_pt.replace(pretty_pt)      # moves & renames
        for f in best_dir.glob("*optim_states.pt"):
            f.unlink()                  # optional: remove giant optimizer file
        best_dir.rmdir()                # delete the now-empty 'best/' folder

    best_checkpoint_path = str(pretty_pt)
    print(f"  [*] checkpoint saved → {best_checkpoint_path}")

    t_loss,t_ppl,t_main,t_stop,t_after,t_tr = evaluate(
        te_dl,engine,device,loss_fn,cfg['ai_rate'],pad,tok)
    print("\n** TEST ** Loss={:.4f}  PPL={:.4f}".format(t_loss,t_ppl))
    _pp_subset("main",t_main); _pp_subset("STOP cur",t_stop)
    _pp_subset("afterSTOP",t_after); _pp_subset("transition",t_tr)

    meta = {
        "best_checkpoint_path": ckpt,
        # val
        "val_loss": best,
        "val_ppl":  v_ppl,
        "val_main": v_main,
        "val_stop_cur": v_stop,
        "val_after_stop": v_after,
        "val_transition": v_tr,
        # test
        "test_loss": t_loss,
        "test_ppl":  t_ppl,
        "test_main": t_main,
        "test_stop_cur": t_stop,
        "test_after_stop": t_after,
        "test_transition": t_tr
    }
    with open(Path(ckpt).with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta

# ─────────────────────────── 8 ─ main ─────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    res = train_model(cfg)
    print("\nSaved →", res["best_checkpoint_path"])
