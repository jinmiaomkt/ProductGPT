# train4_decision_only_aws.py  ── silent-checkpoint version
# =========================================================
# * Decision-Only model with revenue-gap loss
# * Quiet: no DeepSpeed "Saving model checkpoint..." lines
# * Still produces <DecisionOnly_*.pt> and its companion JSON
# ---------------------------------------------------------

# ─── ENV & LOGGING ─────────────────────────────────────────────────────────
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # silence TensorFlow
os.environ["DEEPSPEED_LOG_LEVEL"]  = "error"  # silence DS
os.environ["DS_DISABLE_LOGS"]      = "1"      # ⬆ completely mute

warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("deepspeed").propagate = False

# ─── STD LIB / 3RD-PARTY IMPORTS ───────────────────────────────────────────
import json, math
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
from pytorch_lamb import Lamb                                # optimiser

from model4_decoderonly     import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers
from config4_decision_only_git import get_config

# ─── TOKENISER ─────────────────────────────────────────────────────────────
def build_tokenizer_tgt():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {"[PAD]": 0,
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
             "6": 6, "7": 7, "8": 8, "9": 9,
             "[SOS]": 10, "[UNK]": 12}
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ─── LOSS ──────────────────────────────────────────────────────────────────
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue = revenue + [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty",
                             -torch.abs(rev[:, None] - rev[None, :]))  # (V,V)
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits:  (B, T, V)   raw scores
        targets: (B, T)      int labels
        """
        # -----------------------------------------------------------------
        # 1. flatten
        B, T, V = logits.shape
        probs = torch.softmax(logits.reshape(-1, V), dim=-1)
        tgt   = targets.reshape(-1)

        # -----------------------------------------------------------------
        # 2. mask "ignore_index"
        keep = (tgt != self.ignore_index)
        if keep.sum() == 0:
            # nothing to optimise for this mini-batch
            return logits.new_tensor(0.0)

        # -----------------------------------------------------------------
        # 3. bring buffer to the same device as logits (robust!)
        pen = self.penalty.to(probs.device)   # <── magic line

        # gather & expectation
        exp_gap = (probs[keep] * pen[tgt[keep]]).sum(dim=-1)
        return (-exp_gap).mean()
    
# ─── HELPERS ───────────────────────────────────────────────────────────────
def transition_mask(lbl: torch.Tensor):
    prev = F.pad(lbl, (1, 0), value=-1)[:, :-1]
    return lbl != prev

def calc_perplexity(logits, tgt, pad=0):
    lp = F.log_softmax(logits, dim=-1)
    lp2d, t = lp.view(-1, lp.size(-1)), tgt.view(-1)
    m = t != pad
    if m.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[m], t[m], reduction="mean")).item()

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
        auprc  = average_precision_score(y_true, pr[:, 1:10], average='macro')
    except ValueError:
        auprc = float("nan")
    return {"hit": hit, "f1": f1, "auprc": auprc, "conf": conf}

def _pretty(tag, d):
    print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

# ─── DATALOADERS ───────────────────────────────────────────────────────────
def get_dataloaders(cfg):
    data = load_json_dataset(cfg["filepath"])
    n   = len(data)
    trn = int(0.8 * n)
    val = int(0.1 * n)
    tst = n - trn - val

    torch.manual_seed(33)
    train, val, test = random_split(
        data, [trn, val, tst], generator=torch.Generator().manual_seed(33))

    tok_tgt = build_tokenizer_tgt()
    tok_ai  = build_tokenizer_tgt()

    Path(cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
    tok_tgt.save(str(Path(cfg["model_folder"]) / "tokenizer_tgt.json"))
    tok_ai .save(str(Path(cfg["model_folder"]) / "tokenizer_ai.json"))

    def mk(split):
        return TransformerDataset(split, tok_ai, tok_tgt,
                                  cfg["seq_len_ai"], cfg["seq_len_tgt"],
                                  cfg["num_heads"], cfg["ai_rate"],
                                  pad_token=0)

    dl = lambda d, s: DataLoader(d, batch_size=cfg["batch_size"], shuffle=s)
    return dl(mk(train), True), dl(mk(val), False), dl(mk(test), False), tok_tgt

# ─── MODEL ────────────────────────────────────────────────────────────────
def get_model(cfg):
    return build_transformer(cfg["vocab_size_tgt"],
                             cfg["seq_len_ai"],
                             cfg["d_model"],
                             cfg["N"],
                             cfg["num_heads"],
                             cfg["d_ff"],
                             cfg["dropout"])

# ─── EVALUATION ───────────────────────────────────────────────────────────
def evaluate(loader, engine, device, loss_fn, step, pad, tok):
    if len(loader) == 0:
        nan = float("nan"); return nan, nan, {}, {}, {}, {}
    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tloss = tppl = 0.0
    P, L, PR = [], [], []
    m_stop, m_prev, m_tr = [], [], []

    engine.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["aggregate_input"].to(device)
            y = batch["label"].to(device)

            g   = engine(x)
            pos = torch.arange(step - 1, g.size(1), step, device=device)
            g   = g[:, pos, :]

            y_eval = y.clone()
            y_eval[transition_mask(y)] = pad

            tloss += loss_fn(g, y_eval).item()
            tppl  += calc_perplexity(g, y_eval, pad)

            probs = F.softmax(g, dim=-1).view(-1, g.size(-1)).cpu().numpy()
            pred  = probs.argmax(1)
            lbl   = y.view(-1).cpu().numpy()
            valid = ~np.isin(lbl, list(special))

            P.append(pred[valid]); L.append(lbl[valid]); PR.append(probs[valid])
            m_stop .append((y == 9).view(-1).cpu().numpy()[valid])
            m_prev .append((F.pad(y, (1, 0), value=-1)[:, :-1] == 9).view(-1).cpu().numpy()[valid])
            m_tr   .append(transition_mask(y).view(-1).cpu().numpy()[valid])

    P, L, PR = map(np.concatenate, (P, L, PR))
    m_stop, m_prev, m_tr = map(np.concatenate, (m_stop, m_prev, m_tr))
    main_mask = ~m_tr

    return (tloss / len(loader), tppl / len(loader),
            _subset_metrics(P, L, PR, main_mask),
            _subset_metrics(P, L, PR, m_stop),
            _subset_metrics(P, L, PR, m_prev),
            _subset_metrics(P, L, PR, m_tr))

# ─── TRAIN LOOP ───────────────────────────────────────────────────────────
def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slots = cfg["seq_len_ai"] // cfg["ai_rate"]
    uid   = (f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")
    checkpoint_path = Path(cfg["model_folder"]) / f"DecisionOnly_{uid}.pt"

    tr_dl, va_dl, te_dl, tok = get_dataloaders(cfg)
    pad = tok.token_to_id("[PAD]")

    model   = get_model(cfg)
    loss_fn = PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],
                                  cfg["vocab_size_tgt"], pad)

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "zero_allow_untested_optimizer": True,
            "optimizer": {
                "type": "Lamb",
                "params": {"lr": cfg["lr"], "eps": cfg["eps"],
                           "weight_decay": cfg["weight_decay"]}
            },
            "fp16": {"enabled": False},
            "zero_optimization": {"stage": 1},
        }
    )

    best_val, patience = None, 0
    for ep in range(cfg["num_epochs"]):
        # --------------- train -----------------------------------------
        engine.train()
        tot = 0.0
        for batch in tqdm(tr_dl, desc=f"Ep {ep:02d}"):
            x = batch["aggregate_input"].to(device)
            y = batch["label"].to(device)

            pos = torch.arange(cfg["ai_rate"] - 1, x.size(1),
                               cfg["ai_rate"], device=device)
            logits = engine(x)[:, pos, :]

            y_tr = y.clone()
            y_tr[transition_mask(y)] = pad

            loss = loss_fn(logits, y_tr)
            engine.zero_grad()
            engine.backward(loss)
            engine.step()
            tot += loss.item()

        print(f"\nTrain loss {tot/len(tr_dl):.4f}")

        # --------------- validation -----------------------------------
        v_loss, v_ppl, v_main, v_stop, v_after, v_tr = evaluate(
            va_dl, engine, device, loss_fn, cfg["ai_rate"], pad, tok)

        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        _pretty("main",       v_main)
        _pretty("STOP cur",   v_stop)
        _pretty("afterSTOP",  v_after)
        _pretty("transition", v_tr)

        if best_val is None or v_loss < best_val:
            best_val = v_loss
            patience = 0
            # ---- QUIET checkpoint ------------------------------------
            torch.save({
                "epoch": ep,
                "model_state_dict": engine.module.state_dict(),
                "optimizer_state_dict": engine.optimizer.state_dict()
            }, checkpoint_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping.")
                break

    # --------------- test ------------------------------------------------
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    engine.module.load_state_dict(state["model_state_dict"])

    t_loss, t_ppl, t_main, t_stop, t_after, t_tr = evaluate(
        te_dl, engine, device, loss_fn, cfg["ai_rate"], pad, tok)

    print("\n** TEST ** Loss={:.4f}  PPL={:.4f}".format(t_loss, t_ppl))
    _pretty("main",       t_main)
    _pretty("STOP cur",   t_stop)
    _pretty("afterSTOP",  t_after)
    _pretty("transition", t_tr)

    # --------------- persist JSON ---------------------------------------
    meta = {
        "best_checkpoint_path": str(checkpoint_path),
        "val_loss": best_val,
        "val_ppl":  v_ppl,
        "val_main": v_main,
        "val_stop_cur": v_stop,
        "val_after_stop": v_after,
        "val_transition": v_tr,
        "test_loss": t_loss,
        "test_ppl":  t_ppl,
        "test_main": t_main,
        "test_stop_cur": t_stop,
        "test_after_stop": t_after,
        "test_transition": t_tr
    }
    with open(checkpoint_path.with_suffix(".json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return meta

# ─── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    res = train_model(cfg)
    print("\nSaved →", res["best_checkpoint_path"])
