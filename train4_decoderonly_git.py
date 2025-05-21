# train4_decoderonly.py  ──────────────────────────────────────────────────
# Decision-Only trainer
#   • PairwiseRevenueLoss
#   • Four evaluation subsets  →  all / STOP-cur / after-STOP / transition
#   • Keeps original logging, checkpoint & JSON-metadata style
# ------------------------------------------------------------------------

import os, json, warnings, logging, numpy as np
from pathlib import Path
from typing import Dict, Any

# ─── silence noisy libraries ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# ─── 3-party deps ───────────────────────────────────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

# ─── project imports ────────────────────────────────────────────────────
from config4              import get_config
from model4_decoderonly   import build_transformer
from dataset4_decoderonly import TransformerDataset, load_json_dataset
from tokenizers           import Tokenizer, models, pre_tokenizers


# ═════════════════════════ 1. tokenisers ════════════════════════════════
def _tok_base(extra: Dict[str, int]) -> Tokenizer:
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {
        "[PAD]": 0,
        **{str(i): i for i in range(1, 10)},   # 1‥9
        "[SOS]": 10,
        "[EOS]": 11,
        "[UNK]": 12,
        **extra
    }
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

build_tokenizer_src = lambda: _tok_base({str(i): i for i in range(13, 61)})
build_tokenizer_tgt = lambda: _tok_base({})                     # only 1‥9


# ═════════════════════════ 2. revenue-gap loss ══════════════════════════
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue += [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty", -torch.abs(rev[:, None] - rev[None, :]))
        self.ignore = ignore_index

    def forward(self, logits, targets):
        V   = logits.size(-1)
        prob = F.softmax(logits.reshape(-1, V), dim=-1)
        tgt  = targets.reshape(-1)
        keep = tgt != self.ignore
        if keep.sum() == 0:
            return logits.new_tensor(0.0)
        pen = self.penalty.to(prob)
        return -(prob[keep] * pen[tgt[keep]]).sum(dim=1).mean()


# ═════════════════════════ 3. helpers ═══════════════════════════════════
def transition_mask(seq: torch.Tensor) -> torch.Tensor:
    """True where seq[t] ≠ seq[t-1].  Shape (B,T)"""
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev

def _perplexity(logits, targets, pad=0):
    logp = F.log_softmax(logits, dim=-1)
    lp2d, tgt = logp.view(-1, logp.size(-1)), targets.view(-1)
    m = tgt != pad
    if m.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[m], tgt[m], reduction="mean")).item()

def _subset_metrics(pred, lbl, probs, mask, classes=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    hit = accuracy_score(l, p)
    f1  = f1_score(l, p, average="macro")
    try:
        auprc = average_precision_score(
            label_binarize(l, classes=classes), pr[:, 1:10], average="macro")
    except ValueError:
        auprc = np.nan
    return {"hit": hit, "f1": f1, "auprc": auprc}

def _json_safe(o: Any):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)): return o.cpu().tolist()
    if isinstance(o, _np.ndarray):  return o.tolist()
    if isinstance(o, (_np.floating, _np.integer)):   return o.item()
    if isinstance(o, dict):   return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    return o


# ═════════════════════════ 4. data loaders ══════════════════════════════
def _make_loaders(cfg):
    raw = load_json_dataset(cfg["filepath"])
    n = len(raw); tr, va = int(.8*n), int(.1*n)
    gen = torch.Generator().manual_seed(33)
    tr_ds, va_ds, te_ds = random_split(raw, [tr, va, n-tr-va], generator=gen)

    tok_ai  = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()

    out = Path(cfg["model_folder"]); out.mkdir(parents=True, exist_ok=True)
    tok_ai .save(str(out/"tokenizer_ai.json"))
    tok_tgt.save(str(out/"tokenizer_tgt.json"))

    def mk(split):
        return TransformerDataset(
            split, tok_ai, tok_tgt,
            cfg["seq_len_ai"], cfg["seq_len_tgt"],
            cfg["num_heads"], cfg["ai_rate"], pad_token=0)

    LD = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
    return LD(mk(tr_ds), True), LD(mk(va_ds), False), LD(mk(te_ds), False), tok_tgt


# ═════════════════════════ 5. model ═════════════════════════════════════
def _build_model(cfg):
    return build_transformer(
        vocab_size  = cfg["vocab_size_src"],
        d_model     = cfg["d_model"],
        n_layers    = cfg["N"],
        n_heads     = cfg["num_heads"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
        max_seq_len = cfg["seq_len_ai"])


# ═════════════════════════ 6. evaluation ════════════════════════════════
def _evaluate(loader, eng, dev, loss_fn, step, pad, tok):
    if not len(loader):
        nan=float("nan"); emp={"hit":nan,"f1":nan,"auprc":nan}
        return nan,nan,emp,emp,emp,emp

    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tot_loss = tot_ppl = 0.0
    P,L,PR = [], [], []
    m_stop_cur, m_after_stop, m_tr = [], [], []

    eng.eval()
    with torch.no_grad():
        for b in loader:
            x = b["aggregate_input"].to(dev)
            y = b["label"].to(dev)

            pos = torch.arange(step-1, x.size(1), step, device=dev)

            logits = eng(x)[:, pos, :]        # (B,N,V)
            tgt    = y[:, pos].clone()        # (B,N)

            mask_tr = transition_mask(y)[:, pos]
            tgt[mask_tr] = pad                # mask transitions for loss

            tot_loss += loss_fn(logits, tgt).item()
            tot_ppl  += _perplexity(logits, tgt, pad)

            prob = F.softmax(logits,dim=-1).view(-1,logits.size(-1)).cpu().numpy()
            pred = prob.argmax(1)
            lbl  = tgt.view(-1).cpu().numpy()
            keep = ~np.isin(lbl, list(special))

            P.append(pred[keep]); L.append(lbl[keep]); PR.append(prob[keep])

            flat = lambda m: m.view(-1).cpu().numpy()[keep]
            m_stop_cur  .append(flat(tgt == 9))
            prev_tok     = F.pad(tgt,(1,0),value=-1)[:, :-1]
            m_after_stop.append(flat(prev_tok == 9))
            m_tr        .append(flat(mask_tr))

    P,L,PR = map(np.concatenate,(P,L,PR))
    stop_cur   = np.concatenate(m_stop_cur)
    after_stop = np.concatenate(m_after_stop)
    tr_mask    = np.concatenate(m_tr)
    all_mask   = np.ones_like(P, dtype=bool)

    return (tot_loss/len(loader), tot_ppl/len(loader),
            _subset_metrics(P,L,PR, all_mask),
            _subset_metrics(P,L,PR, stop_cur),
            _subset_metrics(P,L,PR, after_stop),
            _subset_metrics(P,L,PR, tr_mask))


# ═════════════════════════ 7. training loop ═════════════════════════════
def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slots = cfg["seq_len_ai"] // cfg["ai_rate"]
    uid   = (f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")

    ckpt = Path(cfg["model_folder"]) / f"DecisionOnly_{uid}.pt"
    meta = ckpt.with_suffix(".json")

    tr,va,te,tok = _make_loaders(cfg)
    pad_id = tok.token_to_id("[PAD]")

    model   = _build_model(cfg)
    loss_fn = PairwiseRevenueLoss(
        revenue=[0,1,10,1,10,1,10,1,10,0],
        vocab_size=tok.get_vocab_size(),
        ignore_index=pad_id)

    eng,_,_,_ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "zero_allow_untested_optimizer": True,
            "optimizer":{"type":"Lamb",
                         "params":{"lr":cfg["lr"],
                                   "eps":cfg["eps"],
                                   "weight_decay":cfg["weight_decay"]}},
            "zero_optimization":{"stage":1},
            "fp16":{"enabled":False}})

    best=None; patience=0
    for ep in range(cfg["num_epochs"]):
        eng.train(); run=0.0
        for b in tqdm(tr, desc=f"Ep {ep:02d}", leave=False):
            x = b["aggregate_input"].to(dev)
            y = b["label"].to(dev)

            pos    = torch.arange(cfg["ai_rate"]-1, x.size(1),
                                  cfg["ai_rate"], device=dev)
            logits = eng(x)[:, pos, :]            # (B,N,V)

            tgt = y[:, pos].clone()
            tgt[transition_mask(y)[:, pos]] = pad_id

            loss = loss_fn(logits, tgt)
            eng.zero_grad(); eng.backward(loss); eng.step()
            run += loss.item()
        print(f"\nTrain loss {run/len(tr):.4f}")

        v_loss,v_ppl,v_all,v_stop,v_after,v_tr = _evaluate(
            va, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)

        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag,d in (("all",v_all),("STOP_cur",v_stop),
                      ("after_STOP",v_after),("transition",v_tr)):
            print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

        if best is None or v_loss < best:
            best= v_loss; patience=0
            torch.save({"epoch":ep,"model_state_dict":eng.module.state_dict()}, ckpt)
            meta.write_text(json.dumps(_json_safe({
                "best_checkpoint_path": ckpt.name,
                "val_loss": best, "val_ppl": v_ppl,
                "val_all": v_all, "val_stop_cur": v_stop,
                "val_after_stop": v_after, "val_transition": v_tr
            }), indent=2))
            print(f"  [*] new best saved → {ckpt.name}")
        else:
            patience+=1
            if patience>=cfg["patience"]:
                print("Early stopping."); break

    # —— test on best weights
    if ckpt.exists():
        state=torch.load(ckpt,map_location=dev)
        eng.module.load_state_dict(state["model_state_dict"])
        t_loss,t_ppl,t_all,t_stop,t_after,t_tr = _evaluate(
            te, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)
        print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
        for tag,d in (("all",t_all),("STOP_cur",t_stop),
                      ("after_STOP",t_after),("transition",t_tr)):
            print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

    return {"uid":uid,"val_loss":best}


# ═════════════════════════ 8. CLI entry ═════════════════════════════════
if __name__ == "__main__":
    train_model(get_config())
