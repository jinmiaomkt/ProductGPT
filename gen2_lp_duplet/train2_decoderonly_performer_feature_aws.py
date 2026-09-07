#!/usr/bin/env python3
"""
train2_decoderonly_performer_feature_aws.py
- Decoder-only Performer with feature embeddings (LP/Duplet variant)
- Robust resume: skip training if checkpoint exists on S3
- Keep local ckpt during training; upload but don't unlink
- Proper early stopping with patience
- Safe test/infer with download-if-missing
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import gzip

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- runtime knobs (before import deepspeed) ---
os.environ.setdefault("DS_BUILD_OPS", "0")                    # no fused kernels
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True                  # safer perf on Ampere+
torch.set_float32_matmul_precision("high")

import deepspeed
from pytorch_lamb import Lamb  # noqa: F401  (used by DeepSpeed JSON)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer, models, pre_tokenizers

# ────────────────────────────── project local
from config2 import get_config, get_weights_file_path, latest_weights_file_path
from dataset2_productgpt import TransformerDataset, load_json_dataset
from model2_decoderonly_feature_performer import build_transformer

# S3
import boto3
import botocore

# ────────────────────────────── global config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# ══════════════════════════════ 1. Constants ═══════════════════════════
PAD_ID = 0
DECISION_IDS = list(range(1, 10))  # 1-9
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
SPECIAL_IDS = [
    PAD_ID,
    SOS_DEC_ID,
    EOS_DEC_ID,
    UNK_DEC_ID,
    EOS_PROD_ID,
    SOS_PROD_ID,
]
MAX_TOKEN_ID = UNK_PROD_ID  # 59

# ══════════════════════════════ 2. Data helpers ════════════════════════
FEAT_FILE = Path("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")
FEATURE_COLS: List[str] = [
    # stats
    "Rarity",
    "MaxLife",
    "MaxOffense",
    "MaxDefense",
    # categorical one-hots
    "WeaponTypeOneHandSword",
    "WeaponTypeTwoHandSword",
    "WeaponTypeArrow",
    "WeaponTypeMagic",
    "WeaponTypePolearm",
    "EthnicityIce",
    "EthnicityRock",
    "EthnicityWater",
    "EthnicityFire",
    "EthnicityThunder",
    "EthnicityWind",
    "GenderFemale",
    "GenderMale",
    "CountryRuiYue",
    "CountryDaoQi",
    "CountryZhiDong",
    "CountryMengDe",
    # misc
    "type_figure",
    "MinimumAttack",
    "MaximumAttack",
    "MinSpecialEffect",
    "MaxSpecialEffect",
    "SpecialEffectEfficiency",
    "SpecialEffectExpertise",
    "SpecialEffectAttack",
    "SpecialEffectSuper",
    "SpecialEffectRatio",
    "SpecialEffectPhysical",
    "SpecialEffectLife",
    "LTO",
]


def load_feature_tensor(xls_path: Path) -> torch.Tensor:
    """Load product-level feature embeddings – (V, D) FloatTensor."""
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


# ══════════════════════════════ 3. Tokenisers ═════════════════════════=

def _base_tokeniser(extra_vocab: Dict[str, int] | None = None) -> Tokenizer:
    """Word-level tokeniser with a fixed numeric vocabulary."""
    vocab: Dict[str, int] = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},  # decisions
        "[SOS]": SOS_DEC_ID,
        "[EOS]": EOS_DEC_ID,
        "[UNK]": UNK_DEC_ID,
    }
    if extra_vocab:
        vocab.update(extra_vocab)
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok


def build_tokenizer_src() -> Tokenizer:  # with product IDs
    prod_vocab = {str(i): i for i in range(FIRST_PROD_ID, UNK_PROD_ID + 1)}
    return _base_tokeniser(prod_vocab)


def build_tokenizer_tgt() -> Tokenizer:  # decisions only
    return _base_tokeniser()


# ══════════════════════════════ 4. Losses ═════════════════════════════
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 0.0,
        ignore_index: int = 0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        logits = logits.view(-1, V)
        targets = targets.view(-1)

        # ─── make sure the `weight` tensor matches logits’ dtype/device ──
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(
                dtype=logits.dtype, device=logits.device
            )

        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=weight,
            ignore_index=self.ignore_index,
        )

        mask = targets != self.ignore_index
        ce = ce[mask]
        if ce.numel() == 0:
            return logits.new_tensor(0.0)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ══════════════════════════════ 5. Utility fns ════════════════════════
def transition_mask(seq: torch.Tensor) -> torch.Tensor:  # (B, T)
    """Mask where *decision* changes wrt previous step."""
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev


def perplexity(logits: torch.Tensor, targets: torch.Tensor, pad: int = PAD_ID) -> float:
    logp = F.log_softmax(logits, dim=-1)
    lp2d, tgt = logp.view(-1, logp.size(-1)), targets.view(-1)
    mask = tgt != pad
    if mask.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[mask], tgt[mask], reduction="mean")).item()


# ══════════════════════════════ 6. DataLoaders ═══════════════════════=
def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    mode = cfg.get("mode", "train")

    if mode == "infer":
        raw = load_json_dataset(cfg["test_filepath"], keep_uids=None)
    else:
        keep = None
        if mode == "test":
            keep = set(cfg["uids_test"])
        elif "uids_trainval" in cfg:
            keep = set(cfg["uids_trainval"])
        raw = load_json_dataset(cfg["filepath"], keep_uids=keep)

    # ------------- train / val / test splits -------------
    if mode == "infer":
        tr_ds = raw                                   # will be wrapped once
        va_ds = te_ds = []
    else:
        n = len(raw)
        tr, va = int(0.8 * n), int(0.1 * n)
        g = torch.Generator().manual_seed(33)
        tr_ds, va_ds, te_ds = random_split(raw, [tr, va, n - tr - va], generator=g)

    tok_src = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()

    out_dir = Path(cfg["model_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_src.save(str(out_dir / "tokenizer_lp.json"))
    tok_tgt.save(str(out_dir / "tokenizer_tgt.json"))

    def _wrap(split):
        return TransformerDataset(
            split,
            tok_src,
            tok_tgt,
            cfg["seq_len_lp"],
            cfg["seq_len_tgt"],
            cfg["num_heads"],
            cfg["lp_rate"],
            pad_token=PAD_ID,
        )

    make_loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)

    if mode == "infer":
        return make_loader(_wrap(tr_ds), False), None, None, tok_tgt

    return (
        make_loader(_wrap(tr_ds), True),
        make_loader(_wrap(va_ds), False),
        make_loader(_wrap(te_ds), False),
        tok_tgt,
    )


# ══════════════════════════════ 7. S3 helpers ═════════════════════════
def _json_safe(o: Any):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)): return o.cpu().tolist()
    if isinstance(o, _np.ndarray):  return o.tolist()
    if isinstance(o, (_np.floating, _np.integer)):   return o.item()
    if isinstance(o, dict):   return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    return o


def _s3_client():
    try:
        return boto3.client("s3")
    except botocore.exceptions.BotoCoreError:
        return None


def _s3_exists(s3, bucket: str, key: str) -> bool:
    if s3 is None:
        return False
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _upload_file(s3, local: Path, bucket: str, key: str, *, content_type: str | None = None):
    if s3 is None or not local.exists():
        return
    extra = {}
    if content_type:
        extra["ExtraArgs"] = {"ContentType": content_type}
    s3.upload_file(str(local), bucket, key, **extra)
    print(f"[S3] {local} → s3://{bucket}/{key}")


# ─────────────────── pretty-printer for metric blocks ───────────────────
def _show(tag: str, metrics: Tuple[float, float, dict, dict, dict, dict]) -> None:
    loss, ppl, m_all, m_st, m_af, m_tr = metrics
    print(f"{tag}  Loss={loss:.4f}  PPL={ppl:.4f}")
    for name, d in (
        ("all",          m_all),
        ("cur-STOP",     m_st),
        ("after-STOP",   m_af),
        ("transition",   m_tr),
    ):
        print(f"  {name:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
              f"AUPRC={d['auprc']:.4f}")


# ══════════════════════════════ 8. Model builder ═════════════════════=
def build_model(cfg: Dict[str, Any], feat_tensor: torch.Tensor) -> nn.Module:
    return build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_lp"],
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        nb_features=cfg["nb_features"],
        kernel_type=cfg["kernel_type"],
        d_ff=cfg["d_ff"],
        feature_tensor=feat_tensor,
        special_token_ids=SPECIAL_IDS,
    )


# ══════════════════════════════ 9. Evaluation ════════════════════════
def _subset(pred, lbl, probs, rev_err, mask, classes=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": float("nan"), "f1": float("nan"), "auprc": float("nan"), "rev_mae": float("nan")}
    p, l, pr, re = pred[mask], lbl[mask], probs[mask], rev_err[mask]
    return {
        "hit": accuracy_score(l, p),
        "f1": f1_score(l, p, average="macro"),
        "auprc": average_precision_score(label_binarize(l, classes=classes), pr[:, 1:10], average="macro"),
        "rev_mae": re.mean(),
    }


def evaluate(loader: DataLoader, model: nn.Module, dev: torch.device, loss_fn, pad: int, tok: Tokenizer, lp_rate: int):
    if not loader:
        nan = float("nan")
        emp = {"hit": nan, "f1": nan, "auprc": nan, "rev_mae": nan}
        return nan, nan, emp, emp, emp, emp

    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tot_loss = tot_ppl = 0.0
    P: List[np.ndarray] = []
    L: List[np.ndarray] = []
    PR: List[np.ndarray] = []
    RE: List[np.ndarray] = []
    m_stop: List[np.ndarray] = []
    m_after_stop: List[np.ndarray] = []
    m_tr: List[np.ndarray] = []

    REV_VEC = torch.tensor([1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=torch.float32)  # shape (9,)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["LTO_PreviousDecision"].to(dev)
            tgt = batch["label"].to(dev)
            pos = torch.arange(lp_rate - 1, x.size(1), lp_rate, device=dev)
            logits = model(x)[:, pos, :]

            tgt_ = tgt.clone()
            tot_loss += loss_fn(logits, tgt_).item()
            tot_ppl += perplexity(logits, tgt_, pad)

            prob_t = F.softmax(logits, dim=-1)
            rev_vec = REV_VEC.to(device=prob_t.device, dtype=prob_t.dtype)

            exp_rev  = (prob_t[..., 1:10] * rev_vec).sum(-1)              # (B, n_slots)
            true_rev = rev_vec[(tgt - 1).clamp(min=0, max=8)]             # same shape
            rev_err  = torch.abs(exp_rev - true_rev).view(-1).cpu().numpy()

            prob = prob_t.view(-1, prob_t.size(-1)).cpu().numpy()         # NumPy copy
            pred = prob.argmax(1)
            lbl = tgt.view(-1).cpu().numpy()
            keep = ~np.isin(lbl, list(special))

            P.append(pred[keep])
            L.append(lbl[keep])
            PR.append(prob[keep])
            RE.append(rev_err[keep])

            flat = lambda m: m.view(-1).cpu().numpy()[keep]
            m_stop.append(flat(tgt == 9))
            prev = F.pad(tgt, (1, 0), value=-1)[:, :-1]
            m_after_stop.append(flat(prev == 9))
            m_tr.append(flat(transition_mask(tgt)))

    P_, L_, PR_, RE_ = map(np.concatenate, (P, L, PR, RE))
    return (
        tot_loss / len(loader),
        tot_ppl / len(loader),
        _subset(P_, L_, PR_, RE_, np.ones_like(P_, dtype=bool)),
        _subset(P_, L_, PR_, RE_, np.concatenate(m_stop)),
        _subset(P_, L_, PR_, RE_, np.concatenate(m_after_stop)),
        _subset(P_, L_, PR_, RE_, np.concatenate(m_tr)),
    )


# ══════════════════════════════ 10. Training loop ════════════════════
def train_model(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- artefact paths ------------------------------------------------
    uid = (
        f"featurebased_performerfeatures{cfg['nb_features']}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_"
        f"N{cfg['N']}_heads{cfg['num_heads']}_lr{cfg['lr']}_w{cfg['weight']}"
    )

    # --- artefact folders ---------------------------------------------
    ckpt_dir    = Path(cfg["model_folder"]) / "checkpoints"
    metrics_dir = Path(cfg["model_folder"]) / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir   / f"LP_ProductGPT_{uid}.pt"
    json_path = metrics_dir / f"LP_ProductGPT_{uid}.json"

    s3 = _s3_client()
    bucket = cfg["s3_bucket"]
    ck_key = f"LP_ProductGPT/performer/FeatureBased/checkpoints/{ckpt_path.name}"
    js_key = f"LP_ProductGPT/performer/FeatureBased/metrics/{json_path.name}"
    print(f"[INFO] artefacts → s3://{bucket}/{ck_key} and s3://{bucket}/{js_key}")

    # --- data ----------------------------------------------------------
    train_dl, val_dl, test_dl, tok_tgt = build_dataloaders(cfg)
    pad_id = tok_tgt.token_to_id("[PAD]")

    # --- model ---------------------------------------------------------
    feat_tensor = load_feature_tensor(FEAT_FILE)
    model = build_model(cfg, feat_tensor).to(device)

    # ---- criterion ----------------------------------------------------
    weights = torch.ones(cfg["vocab_size_tgt"], device=device)
    weights[9] = cfg["weight"]
    loss_fn = FocalLoss(cfg["gamma"], pad_id, weights)

    # ---- DeepSpeed ----------------------------------------------------
    ds_cfg = {
        "train_micro_batch_size_per_gpu": cfg["batch_size"],
        "zero_allow_untested_optimizer": True,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "Lamb",
            "params": {"lr": cfg["lr"], "eps": cfg["eps"], "weight_decay": cfg["weight_decay"]},
        },
        "zero_optimization": {"stage": 1},
        "fp16": {"enabled": False},
        "lr_scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": cfg["min_lr"],
                "warmup_max_lr": cfg["lr"],
                "warmup_num_steps": cfg["warmup_steps"],
                "total_num_steps": max(1, cfg["num_epochs"] * max(1, len(train_dl))),
                "decay_style": "cosine",
            },
        },
    }

    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)

    # ---- (optional) preload ------------------------------------------
    preload = cfg.get("preload", None)
    if preload:
        file = latest_weights_file_path(cfg) if preload == "latest" else get_weights_file_path(cfg, preload)
        if file and Path(file).exists():
            print("Preloading", file)
            state = torch.load(file, map_location="cpu")
            engine.module.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])

    # ---------- resume: if a checkpoint is already on S3, download & skip training
    resume_only = False
    if _s3_exists(s3, bucket, ck_key):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, ck_key, str(ckpt_path))
            print(f"[RESUME] Found existing best ckpt → s3://{bucket}/{ck_key}. Skipping training.")
            resume_only = True
        except Exception as e:
            print(f"[RESUME] Failed to download existing ckpt, will train: {e}")

    # Safe defaults so metadata/test never crash
    nan = float("nan")
    best_val_loss = nan
    best_val_metrics: Dict[str, Any] = {}
    t_loss = t_ppl = nan
    t_all = t_stop = t_after = t_tr = {"hit": nan, "f1": nan, "auprc": nan, "rev_mae": nan}

    # ---- training -----------------------------------------------------
    if not resume_only:
        best_val_loss = None
        patience = 0

        for ep in range(cfg["num_epochs"]):
            engine.train()
            running = 0.0
            for batch in deepspeed.comm.tqdm(train_dl, desc=f"Ep {ep:02d}"):
                x = batch["LTO_PreviousDecision"].to(device)
                tgt = batch["label"].to(device)
                pos = torch.arange(cfg["lp_rate"] - 1, cfg["seq_len_lp"], cfg["lp_rate"], device=device)
                logits = engine(x)[:, pos, :]
                tgt_ = tgt.clone()

                # Skip batches with no labels (optional; FocalLoss is already safe)
                if not (tgt_ != pad_id).any():
                    continue

                loss = loss_fn(logits, tgt_)
                engine.zero_grad()
                engine.backward(loss)
                engine.step()
                running += loss.item()

            # ---- validation (INSIDE epoch loop) -----------------------
            v_loss, v_ppl, v_all, v_stop, v_after, v_tr = evaluate(
                val_dl, engine, device, loss_fn, pad_id, tok_tgt, cfg["lp_rate"]
            )

            print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
            for tag, d in (("all", v_all), ("STOP_cur", v_stop), ("after_STOP", v_after), ("transition", v_tr)):
                print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                      f"AUPRC={d['auprc']:.4f}  RevMAE={d.get('rev_mae', float('nan')):.4f}")

            print(f"[INFO] artefacts → s3://{bucket}/{ck_key} and s3://{bucket}/{js_key}")

            if (best_val_loss is None) or (v_loss < best_val_loss):
                best_val_loss, patience = v_loss, 0

                best_val_metrics = {
                    "val_loss": v_loss,
                    "val_ppl": v_ppl,
                    "val_all_hit_rate": v_all["hit"],
                    "val_all_f1_score": v_all["f1"],
                    "val_all_auprc": v_all["auprc"],
                    "val_all_rev_mae": v_all.get("rev_mae", nan),
                    "val_stop_hit_rate": v_stop["hit"],
                    "val_stop_f1_score": v_stop["f1"],
                    "val_stop_auprc": v_stop["auprc"],
                    "val_stop_rev_mae": v_stop.get("rev_mae", nan),
                    "val_after_hit_rate": v_after["hit"],
                    "val_after_f1_score": v_after["f1"],
                    "val_after_auprc": v_after["auprc"],
                    "val_after_rev_mae": v_after.get("rev_mae", nan),
                    "val_transition_hit_rate": v_tr["hit"],
                    "val_transition_f1_score": v_tr["f1"],
                    "val_transition_auprc": v_tr["auprc"],
                    "val_transition_rev_mae": v_tr.get("rev_mae", nan),
                }

                ckpt = {
                    "epoch": ep,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": engine.module.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                json_path.write_text(json.dumps(_json_safe(best_val_metrics), indent=2))

                # upload (do NOT unlink during training)
                _upload_file(s3, json_path, bucket, js_key, content_type="application/json")
                _upload_file(s3, ckpt_path, bucket, ck_key)

            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print("Early stopping")
                    break

    # Ensure local checkpoint is present for evaluation
    if not ckpt_path.exists() and s3 is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, ck_key, str(ckpt_path))
            print(f"[S3] downloaded best ckpt for test → s3://{bucket}/{ck_key}")
        except Exception as e:
            print(f"[WARN] Could not download ckpt for test: {e}")

    # ---- test ---------------------------------------------------------
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        engine.module.load_state_dict(state["model_state_dict"])

        t_loss, t_ppl, t_all, t_stop, t_after, t_tr = evaluate(
            test_dl, engine, device, loss_fn, pad_id, tok_tgt, cfg["lp_rate"]
        )

        print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
        for tag, d in (("all", t_all), ("STOP_cur", t_stop), ("after_STOP", t_after), ("transition", t_tr)):
            print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

    # ------------------ inference on full 30 campaigns ------------------
    inf_dl, _, _, _ = build_dataloaders({**cfg, "mode": "infer"})

    tmp_pred = Path("/tmp") / f"{uid}_predictions.jsonl.gz"
    uid_counter = 0
    with gzip.open(tmp_pred, "wt", encoding="utf-8") as fp, torch.no_grad():
        for batch in deepspeed.comm.tqdm(inf_dl, desc="Infer 30-campaign set"):
            x = batch["LTO_PreviousDecision"].to(device)

            # Use real UIDs if provided; otherwise synthesize placeholders
            if "uid" in batch and batch["uid"] is not None:
                uids = batch["uid"]
            else:
                B = x.size(0)
                uids = [f"ex_{i:08d}" for i in range(uid_counter, uid_counter + B)]
                uid_counter += B

            logits = engine(x)[:, cfg["lp_rate"]-1::cfg["lp_rate"], :]
            probs  = torch.softmax(logits, -1).cpu().numpy()   # (B, N, 60)
            for u, p in zip(uids, probs):
                fp.write(json.dumps({"uid": u, "probs": p.tolist()}) + "\n")

    pred_s3_key = f"CV/predictions/{tmp_pred.name}"
    if s3 is not None:
        try:
            s3.upload_file(str(tmp_pred), bucket, pred_s3_key,
                           ExtraArgs={"ContentType": "application/json", "ContentEncoding": "gzip"})
            print(f"[S3] {tmp_pred} → s3://{bucket}/{pred_s3_key}")
            try:
                tmp_pred.unlink()
            except Exception:
                pass
        except Exception as e:
            print(f"[S3-ERR] {e}")

    # Build metadata (safe even if val/test not run)
    def _gm(key: str, default: float = nan) -> float:
        return float(best_val_metrics.get(key, default)) if best_val_metrics else default

    metadata = {
        "best_checkpoint_path": ckpt_path.name,
        "val_loss": _gm("val_loss"),
        "val_ppl": _gm("val_ppl"),
        "val_all_hit_rate": _gm("val_all_hit_rate"),
        "val_all_f1_score": _gm("val_all_f1_score"),
        "val_all_auprc": _gm("val_all_auprc"),
        "val_all_rev_mae": _gm("val_all_rev_mae"),
        "val_stop_hit_rate": _gm("val_stop_hit_rate"),
        "val_stop_f1_score": _gm("val_stop_f1_score"),
        "val_stop_auprc": _gm("val_stop_auprc"),
        "val_stop_rev_mae": _gm("val_stop_rev_mae"),
        "val_after_hit_rate": _gm("val_after_hit_rate"),
        "val_after_f1_score": _gm("val_after_f1_score"),
        "val_after_auprc": _gm("val_after_auprc"),
        "val_after_rev_mae": _gm("val_after_rev_mae"),
        "val_transition_hit_rate": _gm("val_transition_hit_rate"),
        "val_transition_f1_score": _gm("val_transition_f1_score"),
        "val_transition_auprc": _gm("val_transition_auprc"),
        "val_transition_rev_mae": _gm("val_transition_rev_mae"),
        "test_loss": t_loss,
        "test_ppl": t_ppl,
        "test_all_hit_rate": t_all["hit"],
        "test_all_f1_score": t_all["f1"],
        "test_all_auprc": t_all["auprc"],
        "test_all_rev_mae": t_all.get("rev_mae", nan),
        "test_stop_hit_rate": t_stop["hit"],
        "test_stop_f1_score": t_stop["f1"],
        "test_stop_auprc": t_stop["auprc"],
        "test_stop_rev_mae": t_stop.get("rev_mae", nan),
        "test_after_hit_rate": t_after["hit"],
        "test_after_f1_score": t_after["f1"],
        "test_after_auprc": t_after["auprc"],
        "test_after_rev_mae": t_after.get("rev_mae", nan),
        "test_transition_hit_rate": t_tr["hit"],
        "test_transition_f1_score": t_tr["f1"],
        "test_transition_auprc": t_tr["auprc"],
        "test_transition_rev_mae": t_tr.get("rev_mae", nan),
    }

    metadata_path = metrics_dir / f"LP_ProductGPT_{uid}_final.json"
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2))
    _upload_file(s3, metadata_path, bucket, f"LP_ProductGPT/performer/FeatureBased/metrics/{metadata_path.name}",
                 content_type="application/json")

    # Clean local artefacts last (optional)
    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        json_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Return correct object names for the caller
    return {
        "uid": uid,
        "fold_id": cfg["fold_id"],
        "val_loss": metadata["val_loss"],
        "val_f1":  metadata["val_all_f1_score"],
        "val_auprc": metadata["val_all_auprc"],
        "test_f1": metadata["test_all_f1_score"],
        "test_auprc": metadata["test_all_auprc"],
        "ckpt": ck_key.split("/")[-1],                   # name in checkpoints/
        "preds": f"{uid}_predictions.jsonl.gz",          # uploaded filename in CV/predictions/
    }


# ══════════════════════════════ 11. CLI ═══════════════════════════════
if __name__ == "__main__":
    cfg = get_config()
    cfg["seq_len_lp"] = cfg["lp_rate"] * cfg["seq_len_tgt"]
    train_model(cfg)
