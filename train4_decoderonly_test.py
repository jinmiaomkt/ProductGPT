# -*- coding: utf-8 -*-
"""
train_decision_model_mask_transition.py  – FULL SCRIPT (2025-05-20)

* Classic printout for val/test plus subset metrics.
* Writes rich JSON alongside the best checkpoint.
"""
import os, json, logging, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             f1_score, average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import deepspeed
from pytorch_lamb import Lamb            # fused LAMB

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
    vocab = {"[PAD]": 0, "1": 1, "2": 2, "3": 3, "4": 4,
             "5": 5,  "6": 6, "7": 7, "8": 8, "9": 9,
             "[SOS]": 10, "[UNK]": 12}
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ─────────────────────────── 2 ─ loss ─────────────────────────────────────
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
        probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
        tgt   = targets.view(-1)
        mask  = tgt != self.ignore_index
        if mask.sum() == 0:
            return logits.new_tensor(0.0)
        exp_gap = (probs[mask] * self.penalty[tgt[mask]]).sum(dim=-1)
        return (-exp_gap).mean()

# ─────────────────────────── 3 ─ helpers ──────────────────────────────────
def transition_mask(labels: torch.Tensor):
    prev = F.pad(labels, (1, 0), value=-1)[:, :-1]
    return labels != prev

def calculate_perplexity(logits, targets, pad_token=0):
    logp = F.log_softmax(logits, dim=-1)
    lp2d, tgt1d = logp.view(-1, logp.size(-1)), targets.view(-1)
    mask = tgt1d != pad_token
    if mask.sum() == 0:
        return float("nan")
    nll = F.nll_loss(lp2d[mask], tgt1d[mask], reduction="mean")
    return torch.exp(nll).item()

# pretty-print one subset line
def _pp_subset(tag, d):
    print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

# ─────────────────────────── 4 ─ dataloaders ──────────────────────────────
def get_dataloaders(cfg):
    raw = load_json_dataset(cfg["filepath"])
    n = len(raw)
    tr, va = int(0.8 * n), int(0.1 * n)
    torch.manual_seed(33)
    train, val, test = random_split(
        raw, [tr, va, n - tr - va], generator=torch.Generator().manual_seed(33))

    tok_tgt = build_tokenizer_tgt()
    tok_ai  = build_tokenizer_tgt()
    Path(cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
    tok_tgt.save(str(Path(cfg["model_folder"]) / "tokenizer_tgt.json"))
    tok_ai .save(str(Path(cfg["model_folder"]) / "tokenizer_ai.json"))

    def make_ds(split):
        return TransformerDataset(split, tok_ai, tok_tgt,
                                  cfg["seq_len_ai"], cfg["seq_len_tgt"],
                                  cfg["num_heads"], cfg["ai_rate"], pad_token=0)

    mk_loader = lambda d, shuf: DataLoader(
        d, batch_size=cfg["batch_size"], shuffle=shuf)

    return (mk_loader(make_ds(train), True),
            mk_loader(make_ds(val  ), False),
            mk_loader(make_ds(test ), False),
            tok_tgt)

# ─────────────────────────── 5 ─ model ────────────────────────────────────
def get_model(cfg):
    return build_transformer(
        vocab_size   = cfg["vocab_size_tgt"],
        d_model      = cfg["d_model"],
        n_layers     = cfg["N"],
        n_heads      = cfg["num_heads"],
        d_ff         = cfg["d_ff"],
        max_seq_len  = cfg["seq_len_ai"],
        dropout      = cfg["dropout"])

# ─────────────────────────── 6 ─ evaluation ───────────────────────────────
def _subset_metrics(pred, lbl, probs, mask, classes=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": float("nan"), "f1": float("nan"),
                "auprc": float("nan"), "conf": None}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    conf = confusion_matrix(l, p, labels=np.unique(l))
    hit  = accuracy_score(l, p)
    f1   = f1_score(l, p, average="macro")
    try:
        y_true = label_binarize(l, classes=classes)
        auprc  = average_precision_score(
            y_true, pr[:, 1:10], average="macro")
    except ValueError:
        auprc = float("nan")
    return {"hit": hit, "f1": f1, "auprc": auprc, "conf": conf}

def evaluate(loader, engine, device, loss_fn, stepsize, pad_id, tok):
    if len(loader) == 0:
        nan = float("nan")
        return nan, nan, {}, {}, {}, {}

    special = {pad_id, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}
    tot_loss = tot_ppl = 0.0
    P, L, PR = [], [], []
    m_stop, m_prevstop, m_trans = [], [], []

    engine.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch["aggregate_input"].to(device), batch["label"].to(device)
            logits = engine(x)
            pos = torch.arange(stepsize - 1, logits.size(1), stepsize,
                               device=device)
            log_dec = logits[:, pos, :]

            y_eval = y.clone()
            y_eval[transition_mask(y)] = pad_id
            loss = loss_fn(log_dec, y_eval)
            tot_loss += loss.item()
            tot_ppl  += calculate_perplexity(log_dec, y_eval, pad_token=pad_id)

            probs = F.softmax(log_dec, dim=-1).view(-1,
                                                    log_dec.size(-1)).cpu().numpy()
            pred  = probs.argmax(axis=1)
            lbl   = y.view(-1).cpu().numpy()
            valid = ~np.isin(lbl, list(special))

            P.append(pred[valid]); L.append(lbl[valid]); PR.append(probs[valid])
            m_stop    .append((y == 9).view(-1).cpu().numpy()[valid])
            m_prevstop.append((F.pad(y, (1, 0), value=-1)[:, :-1] == 9)
                              .view(-1).cpu().numpy()[valid])
            m_trans   .append(transition_mask(y).view(-1).cpu().numpy()[valid])

    P, L, PR = np.concatenate(P), np.concatenate(L), np.concatenate(PR)
    m_stop, m_prevstop, m_trans = map(np.concatenate,
                                      (m_stop, m_prevstop, m_trans))
    main_mask = ~m_trans

    return (tot_loss / len(loader),
            tot_ppl  / len(loader),
            _subset_metrics(P, L, PR, main_mask),
            _subset_metrics(P, L, PR, m_stop),
            _subset_metrics(P, L, PR, m_prevstop),
            _subset_metrics(P, L, PR, m_trans))

# ─────────────────────────── 7 ─ training loop ────────────────────────────
def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, va_dl, te_dl, tok = get_dataloaders(cfg)
    pad_id = tok.token_to_id("[PAD]")

    model = get_model(cfg).to(device)
    loss_fn = PairwiseRevenueLoss(
        revenue=[0, 1, 10, 1, 10, 1, 10, 1, 10, 0],
        vocab_size=cfg["vocab_size_tgt"],
        ignore_index=pad_id).to(device)

    tot_steps = cfg["num_epochs"] * len(tr_dl)

    ds_cfg = {
        "train_micro_batch_size_per_gpu": cfg["batch_size"],
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "Lamb",
            "params": {"lr": cfg["lr"], "eps": cfg["eps"],
                       "weight_decay": cfg["weight_decay"]}
        },
        "lr_scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": cfg["min_lr"],
                "warmup_max_lr": cfg["lr"],
                "warmup_num_steps": cfg["warmup_steps"],
                "total_num_steps": tot_steps,
                "decay_style": "cosine"
            }
        },
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 1}
    }

    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_cfg)

    best_val, best_ckpt, patience = None, None, 0

    for epoch in range(cfg["num_epochs"]):
        # ---- training -----------------------------------------------------
        engine.train()
        running = 0.0
        pbar = tqdm(tr_dl, desc=f"Ep {epoch:02d}")
        for batch in pbar:
            x, y = batch["aggregate_input"].to(device), batch["label"].to(device)
            pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"],
                               device=device)
            logits = engine(x)[:, pos, :]
            y_tr = y.clone()
            y_tr[transition_mask(y)] = pad_id
            loss = loss_fn(logits, y_tr)

            engine.zero_grad()
            engine.backward(loss)
            engine.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"\nTrain loss {running / len(tr_dl):.4f}")

        # ---- validation ---------------------------------------------------
        v_loss, v_ppl, v_main, v_stop, v_after, v_trans = evaluate(
            va_dl, engine, device, loss_fn, cfg["ai_rate"], pad_id, tok)

        print(f"Epoch {epoch:02d} "
              f"ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        _pp_subset("main",       v_main)
        _pp_subset("STOP cur",   v_stop)
        _pp_subset("afterSTOP",  v_after)
        _pp_subset("transition", v_trans)

        # ---- early stop / checkpoint -------------------------------------
        if best_val is None or v_loss < best_val:
            best_val = v_loss
            patience = 0
            best_ckpt = get_weights_file_path(cfg, "best")
            engine.save_checkpoint(str(Path(best_ckpt).parent), tag="best")
            print(f"  [*] new best saved → {best_ckpt}")
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stop triggered.")
                break

    # ─────── test on best checkpoint ──────────────────────────────────────
    engine.load_checkpoint(str(Path(best_ckpt).parent), tag="best")
    t_loss, t_ppl, t_main, t_stop, t_after, t_trans = evaluate(
        te_dl, engine, device, loss_fn, cfg["ai_rate"], pad_id, tok)

    print("\n** TEST **")
    print(f"Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
    _pp_subset("main",       t_main)
    _pp_subset("STOP cur",   t_stop)
    _pp_subset("afterSTOP",  t_after)
    _pp_subset("transition", t_trans)

    # ─────── write JSON ───────────────────────────────────────────────────
    meta = {
        "best_checkpoint_path": best_ckpt,
        "val_loss": v_loss,
        "val_ppl":  v_ppl,
        "val_main": v_main,
        "val_stop_cur": v_stop,
        "val_after_stop": v_after,
        "val_transition": v_trans,
        "test_loss": t_loss,
        "test_ppl":  t_ppl,
        "test_main": t_main,
        "test_stop_cur": t_stop,
        "test_after_stop": t_after,
        "test_transition": t_trans
    }
    json_path = Path(best_ckpt).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

# ─────────────────────────── 8 ─ main ─────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    res = train_model(cfg)
    print("\nSaved →", res["best_checkpoint_path"])
