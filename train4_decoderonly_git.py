# train4_decoderonly.py  ────────────────────────────────────────────────────
# Decision-Only trainer
#   • PairwiseRevenueLoss (revenue-gap)
#   • Four evaluation subsets at each decision position:
#       all / STOP-cur / after-STOP / transition
#   • Keeps original logging, checkpoint & JSON style
# ───────────────────────────────────────────────────────────────────────────

import os, json, warnings, logging, numpy as np
from pathlib import Path
from typing import Dict, Any

# ─── mute noisy frameworks ────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
logging.getLogger("deepspeed").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ─── 3rd-party ------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

# ─── project code ---------------------------------------------------------
from config4 import get_config, get_weights_file_path, latest_weights_file_path
from model4_decoderonly     import build_transformer
from dataset4_decoderonly   import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers

# --- tokeniser helpers ----------------------------------------------------
def _build_tok_base(vocab_extra: Dict[str, int]) -> Tokenizer:
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    # ➊  — add 11 back as [EOS] —
    vocab = {
        "[PAD]": 0,
        **{str(i): i for i in range(1, 10)},   # 1..9
        "[SOS]": 10,
        "[EOS]": 11,                           # ← restore
        "[UNK]": 12,
    }
    vocab.update(vocab_extra)
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

build_tokenizer_src = lambda: _build_tok_base({str(i): i for i in range(13,61)})
build_tokenizer_tgt = lambda: _build_tok_base({})

# ═════════════════════════════════ loss ══════════════════════════════════
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue += [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty",
                             -torch.abs(rev[:, None] - rev[None, :]))  # V×V
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        V = logits.size(-1)
        probs = F.softmax(logits.view(-1, V), dim=-1)
        tgt   = targets.view(-1)
        keep  = tgt != self.ignore_index
        if keep.sum() == 0:
            return logits.new_tensor(0.0)
        pen = self.penalty.to(probs)            # correct device
        gap = (probs[keep] * pen[tgt[keep]]).sum(dim=-1)
        return (-gap).mean()

# ═════════════════════════════════ helpers ═══════════════════════════════
def transition_mask(y: torch.Tensor) -> torch.Tensor:          # (B,T)
    prev = F.pad(y, (1, 0), value=-1)[:, :-1]
    return y != prev

def safe_json(obj: Any):
    """convert ndarray / tensors so json.dump works"""
    import numpy as _np, torch as _th
    if isinstance(obj, (_th.Tensor, _th.nn.Parameter)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    return obj

def calculate_perplexity(logits, targets, pad_token=0):
    logp = F.log_softmax(logits, dim=-1)
    lp2d, tgt = logp.view(-1, logp.size(-1)), targets.view(-1)
    mask = tgt != pad_token
    if mask.sum() == 0:
        return float("nan")
    nll = F.nll_loss(lp2d[mask], tgt[mask], reduction='mean')
    return torch.exp(nll).item()

def subset_metrics(pred, lbl, probs, mask, classes=np.arange(1,10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan, "conf": None}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    hit = accuracy_score(l, p)
    f1  = f1_score(l, p, average='macro')
    try:
        y_true = label_binarize(l, classes=classes)
        auprc  = average_precision_score(y_true, pr[:,1:10], average='macro')
    except ValueError:
        auprc = np.nan
    conf = confusion_matrix(l, p, labels=np.unique(l))
    return {"hit": hit, "f1": f1, "auprc": auprc, "conf": conf}

# ═════════════════════════════════ data ══════════════════════════════════
def get_dataloaders(cfg):
    data = load_json_dataset(cfg["filepath"])
    n     = len(data)
    tr,va = int(0.8*n), int(0.1*n)
    tr_ds, va_ds, te_ds = random_split(
        data, [tr, va, n-tr-va], generator=torch.Generator().manual_seed(33))

    tok_ai  = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()

    out = Path(cfg["model_folder"]); out.mkdir(parents=True, exist_ok=True)
    tok_ai.save (str(out / "tokenizer_ai.json"))
    tok_tgt.save(str(out / "tokenizer_tgt.json"))

    mk_ds = lambda split: TransformerDataset(
        split, tok_ai, tok_tgt,
        cfg["seq_len_ai"], cfg["seq_len_tgt"],
        cfg["num_heads"], cfg["ai_rate"], pad_token=0)

    loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
    return loader(mk_ds(tr_ds), True), loader(mk_ds(va_ds), False), \
           loader(mk_ds(te_ds), False), tok_tgt

# ═════════════════════════════════ model ═════════════════════════════════
def get_model(cfg):
    return build_transformer(
        vocab_size  = cfg["vocab_size_src"],
        d_model     = cfg["d_model"],
        n_layers    = cfg["N"],
        n_heads     = cfg["num_heads"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
        max_seq_len = cfg["seq_len_ai"])

# ═════════════════════════════════ evaluation ════════════════════════════
def evaluate(loader, engine, device, loss_fn, step, pad, tok):
    """
    Returns:
        loss, ppl,
        metrics_all, metrics_stop_cur, metrics_after_stop, metrics_transition
    """
    if len(loader) == 0:
        nan=float("nan"); empty={"hit":nan,"f1":nan,"auprc":nan}
        return nan,nan,empty,empty,empty,empty

    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tot_loss = tot_ppl = 0.0
    P,L,PR   = [], [], []
    m_stop_cur  = []          # current token == 9
    m_after_stop= []          # previous decision token == 9
    m_tr        = []          # transition positions

    engine.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['aggregate_input'].to(device)
            y = batch['label'].to(device)

            logits = engine(x)
            pos    = torch.arange(step-1, logits.size(1), step, device=device)
            logits = logits[:, pos, :]                 # decision positions only
            tgt    = y[:, pos]                         # labels at decisions

            # mask transitions for loss
            tgt_masked = tgt.clone()
            tgt_masked[transition_mask(y)[:, pos]] = pad
            tot_loss += loss_fn(logits, tgt_masked).item()
            tot_ppl  += calculate_perplexity(logits, tgt_masked, pad)

            probs = F.softmax(logits, dim=-1).view(-1, logits.size(-1)).cpu().numpy()
            pred  = probs.argmax(1)
            lbl   = tgt.view(-1).cpu().numpy()
            valid = ~np.isin(lbl, list(special))

            P.append(pred[valid]); L.append(lbl[valid]); PR.append(probs[valid])
            flat = lambda m: m.view(-1).cpu().numpy()[valid]
            m_stop_cur  .append(flat(tgt == 9))
            prev_tok     = F.pad(tgt, (1,0), value=-1)[:, :-1]
            m_after_stop.append(flat(prev_tok == 9))
            m_tr.append(flat(transition_mask(y)[:, pos]))

    P,L,PR = map(np.concatenate,(P,L,PR))
    m_stop_cur   = np.concatenate(m_stop_cur)
    m_after_stop = np.concatenate(m_after_stop)
    m_tr         = np.concatenate(m_tr)
    all_mask     = np.ones_like(P, dtype=bool)

    return (tot_loss/len(loader), tot_ppl/len(loader),
            subset_metrics(P,L,PR, all_mask),
            subset_metrics(P,L,PR, m_stop_cur),
            subset_metrics(P,L,PR, m_after_stop),
            subset_metrics(P,L,PR, m_tr))

# ═════════════════════════════════ train ═════════════════════════════════
def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slots = cfg["seq_len_ai"] // cfg["ai_rate"]
    uid   = (f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")
    ckpt_path = Path(cfg["model_folder"]) / f"DecisionOnly_{uid}.pt"
    json_path = ckpt_path.with_suffix(".json")

    # ── data, model, loss ───────────────────────────────────────────────
    tr_dl, va_dl, te_dl, tok = get_dataloaders(cfg)
    pad_id = tok.token_to_id("[PAD]")

    model = get_model(cfg)
    loss_fn = PairwiseRevenueLoss(
        revenue=[0,1,10,1,10,1,10,1,10,0],
        vocab_size=cfg["vocab_size_src"],
        ignore_index=pad_id)

    # ── DeepSpeed initialise ────────────────────────────────────────────
    ds_cfg = {
        "train_micro_batch_size_per_gpu": cfg["batch_size"],
        "zero_allow_untested_optimizer": True,
        "optimizer": {"type": "Lamb",
                      "params": {"lr": cfg["lr"],
                                 "eps": cfg["eps"],
                                 "weight_decay": cfg["weight_decay"]}},
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 1}
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_cfg)

    best_val = None; patience = 0
    for ep in range(cfg["num_epochs"]):
        # ── TRAIN ───────────────────────────────────────────────────────
        engine.train(); running = 0.0
        for batch in tqdm(tr_dl, desc=f"Ep {ep:02d}", leave=False):
            x = batch['aggregate_input'].to(device)
            y = batch['label'].to(device)

            pos = torch.arange(cfg["ai_rate"]-1, x.size(1),
                               cfg["ai_rate"], device=device)
            logits = engine(x)[:, pos, :]

            y_tr = y.clone(); y_tr[transition_mask(y)] = pad_id
            loss = loss_fn(logits, y_tr)

            engine.zero_grad()
            engine.backward(loss)
            engine.step()
            running += loss.item()
        print(f"\nTrain loss {running/len(tr_dl):.4f}")

        # ── VALIDATE ────────────────────────────────────────────────────
        v_loss,v_ppl,v_all,v_stop_cur,v_after_stop,v_tr = evaluate(
            va_dl, engine, device, loss_fn,
            cfg["ai_rate"], pad_id, tok)

        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag,d in (("all",v_all),("STOP-cur",v_stop_cur),
                      ("after-STOP",v_after_stop),("transition",v_tr)):
            print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

        # ── Checkpoint on improvement ───────────────────────────────────
        if best_val is None or v_loss < best_val:
            best_val = v_loss; patience = 0
            torch.save({"epoch": ep,
                        "model_state_dict": engine.module.state_dict()},
                       ckpt_path)

            meta = safe_json({
                "best_checkpoint_path": ckpt_path.name,
                "val_loss": best_val,
                "val_ppl":  v_ppl,
                "val_all":          v_all,
                "val_stop_cur":     v_stop_cur,
                "val_after_stop":   v_after_stop,
                "val_transition":   v_tr
            })
            json_path.write_text(json.dumps(meta, indent=2))
            print(f"  [*] new best saved → {ckpt_path.name}")
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping."); break

    # ── TEST (using the best weights we kept locally) ───────────────────
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        engine.module.load_state_dict(state["model_state_dict"])

        t_loss,t_ppl,t_all,t_stop_cur,t_after_stop,t_tr = evaluate(
            te_dl, engine, device, loss_fn,
            cfg["ai_rate"], pad_id, tok)

        print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
        for tag,d in (("all",t_all),("STOP-cur",t_stop_cur),
                      ("after-STOP",t_after_stop),("transition",t_tr)):
            print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

    # return stats if another script (e.g. sweep) imports this
    return {"uid": uid, "val_loss": best_val}

# ═════════════════════════════════ CLI entry ═════════════════════════════
if __name__ == "__main__":
    train_model(get_config())
