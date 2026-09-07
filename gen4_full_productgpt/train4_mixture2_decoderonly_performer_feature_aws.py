from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import boto3
import botocore
import gzip
import torch
from tqdm.auto import tqdm
import random
# --- runtime knobs (before import deepspeed) ---
os.environ.setdefault("DS_BUILD_OPS", "0")                    # no fused kernels
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True                  # safer perf on Ampere+
torch.set_float32_matmul_precision("high")
import deepspeed
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lamb import Lamb # noqa: F401  (used by DeepSpeed JSON)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer, models, pre_tokenizers
from tqdm import tqdm
from typing import Callable, Optional, Dict, Any

# ────────────────────────────── project local
from config4 import get_config, get_weights_file_path, latest_weights_file_path
from dataset4_mixture import TransformerDataset, load_json_dataset
from model4_mixture2_decoderonly_feature_performer import build_transformer

# ────────────────────────────── global config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# ══════════════════════════════ 1. Constants ═══════════════════════════
PAD_ID = 0
DECISION_IDS = list(range(1, 10))  # 1‑9
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
    # categorical one‑hots
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
    """Load product‑level feature embeddings – (V, D) FloatTensor."""
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)

def export_user_mixture_weights(state_dict, index_to_uid, out_csv):
    import pandas as pd
    import torch

    key = "projection.output_head.user_mix_logits.weight"
    if key not in state_dict:
        raise KeyError(f"Missing {key} in checkpoint state_dict")

    mix_logits = state_dict[key].detach().cpu()              # [num_users, num_heads]
    mix_weights = torch.softmax(mix_logits, dim=-1)          # normalize across heads

    rows = []
    num_users, num_heads = mix_weights.shape

    for user_index in range(num_users):
        row = {
            "user_index": user_index,
            "uid": index_to_uid.get(user_index, "[MISSING]"),
        }
        for h in range(num_heads):
            row[f"mix_head_{h}_logit"] = float(mix_logits[user_index, h])
            row[f"mix_head_{h}_weight"] = float(mix_weights[user_index, h])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] saved user mixture weights to {out_csv}")

# ══════════════════════════════ 3. Tokenisers ═════════════════════════=
def _base_tokeniser(extra_vocab: Dict[str, int] | None = None) -> Tokenizer:
    """Word‑level tokeniser with a fixed numeric vocabulary."""
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

# ══════════════════════════ NEW: Class weight computation ═════════════
def compute_class_weights(train_loader, num_classes=9, tau=0.5, pad_id=0):
    """Inverse-frequency weights with temperature tau. Returns (9,) tensor."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in train_loader:
        labels = batch["label"].view(-1)
        for c in range(1, 10):
            counts[c - 1] += (labels == c).sum().item()
    counts = counts.clamp(min=1.0)
    N = counts.sum()
    freq = counts / N
    raw_weight = 1.0 / (float(num_classes) * freq)
    tempered = raw_weight.pow(tau)
    tempered = tempered / tempered.mean()
    print(f"[INFO] Class frequencies:  {(freq * 100).numpy().round(2)}%")
    print(f"[INFO] Class weights (tau={tau}): {tempered.float().numpy().round(4)}")
    return tempered.float()

# ══════════════════════════ NEW: Vector Scaling Calibrator ════════════
class VectorScaling(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits_dec):
        return F.softmax(self.a * logits_dec + self.b, dim=-1)


def fit_vector_scaling(logits_val, labels_val, n_classes=9, max_iter=200):
    cal = VectorScaling(n_classes).to(logits_val.device)
    labels_0based = (labels_val - 1).clamp(0, n_classes - 1).long()
    optimizer = torch.optim.LBFGS(cal.parameters(), lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        p = cal(logits_val)
        loss = F.nll_loss(torch.log(p + 1e-12), labels_0based)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"[CALIBRATOR] a = {cal.a.data.cpu().numpy().round(4)}")
    print(f"[CALIBRATOR] b = {cal.b.data.cpu().numpy().round(4)}")
    return cal

# ══════════════════════════════ 4. Losses ═════════════════════════════
class FocalLoss(nn.Module):
    """Multi‑class focal loss with optional class weights."""

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
            # return a zero with a live graph so backward() won’t crash
            return logits.sum() * 0.0
        
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() 

class SeqWeightedCrossEntropy(nn.Module):
    def __init__(self, weight: torch.Tensor | None, ignore_index: int):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,T,V), targets: (B,T)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        mask = targets.view(-1) != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0
        return loss[mask].mean()


class SeqFocalLoss(nn.Module):
    """
    Standard focal loss for multi-class:
      FL = alpha_y * (1 - pt)^gamma * CE_unweighted
    where pt comes from the unweighted CE (important).
    """
    def __init__(self, gamma: float, ignore_index: int, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        logits2d = logits.view(-1, V)
        targets1d = targets.view(-1)

        # unweighted CE for pt
        ce = F.cross_entropy(
            logits2d,
            targets1d,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        mask = targets1d != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0

        ce = ce[mask]
        t  = targets1d[mask]

        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            loss = loss * self.alpha[t].to(dtype=loss.dtype, device=loss.device)

        return loss.mean()

def make_loss_and_bias(cfg: dict, pad_id: int, device: torch.device):
    """
    Returns:
      loss_fn, logit_bias (or None), class_weight (float tensor length vocab)
    """
    V = cfg["vocab_size_tgt"]

    # class_weight is your "w" used for imbalance / cost-sensitive training
    class_weight = torch.ones(V, device=device, dtype=torch.float32)
    class_weight[9] = float(cfg["weight"])   # your upweighted class index (9)

    gamma = float(cfg.get("gamma", 0.0))

    if gamma == 0.0:
        # weighted CE (analytic correction is valid)
        loss_fn = SeqWeightedCrossEntropy(weight=class_weight, ignore_index=pad_id)

        # correction: softmax(z - log(w))
        logit_bias = torch.log(class_weight).to(device=device)  # log(1)=0, log(w)>0
        return loss_fn, logit_bias, class_weight

    # focal (analytic correction NOT valid)
    # You may still use alpha=class_weight as a focal alpha term
    loss_fn = SeqFocalLoss(gamma=gamma, ignore_index=pad_id, alpha=class_weight)
    logit_bias = None
    return loss_fn, logit_bias, class_weight

# ══════════════════════════════ 5. Utility fns ════════════════════════
def transition_mask(seq: torch.Tensor) -> torch.Tensor:  # (B, T)
    """Mask where *decision* changes wrt previous step."""
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev

# def perplexity(logits: torch.Tensor, targets: torch.Tensor, pad: int = PAD_ID) -> float:
#     logp = F.log_softmax(logits, dim=-1)
#     lp2d, tgt = logp.view(-1, logp.size(-1)), targets.view(-1)
#     mask = tgt != pad
#     if mask.sum() == 0:
#         return float("nan")
#     return torch.exp(F.nll_loss(lp2d[mask], tgt[mask], reduction="mean")).item()

class RepeatWithPermutation(torch.utils.data.Dataset):
    """
    Treat each base record as K distinct samples by repeating indices.
    The underlying dataset must support set_epoch() and/or use sample_index
    in its permutation seeding.
    """
    def __init__(self, base_ds, repeat_factor: int):
        self.base = base_ds
        self.K = int(repeat_factor)
        if self.K <= 0:
            raise ValueError("repeat_factor must be >= 1")

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base) * self.K

    def __getitem__(self, i: int):
        base_i = i % len(self.base)
        rep_i  = i // len(self.base)
        # IMPORTANT: pass a "sample_index" that changes across repeats
        # so permutations differ even within the same epoch.
        # Easiest: temporarily override by calling base.__getitem__ with i
        # only if base uses idx as seed. Otherwise, modify base to accept a seed.
        # return self.base.__getitem__(base_i)  # if your base uses idx for permutation seeding
        return self.base.__getitem__(base_i, sample_index = rep_i)

# ══════════════════════════════ 6. DataLoaders ═══════════════════════=
def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    mode = cfg.get("mode", "train")

    # ── load raw records ────────────────────────────────────────────────
    if mode == "infer":
        raw = load_json_dataset(cfg["test_filepath"], keep_uids=None)
    else:
        keep_uids = None
        if mode == "test":
            keep_uids = set(cfg["uids_test"])
        elif cfg.get("uids_trainval") is not None:
            keep_uids = set(cfg["uids_trainval"])
        raw = load_json_dataset(cfg["filepath"], keep_uids=keep_uids)

    # ── optional deterministic subsample (for cheap HP search) ──────────
    def _deterministic_subsample(raw_list, frac: float, seed: int):
        if frac >= 1.0:
            return raw_list
        n = len(raw_list)
        k = max(1, int(n * frac))
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        keep = set(idx[:k])
        return [raw_list[i] for i in range(n) if i in keep]

    data_frac = float(cfg.get("data_frac", 1.0))          # e.g., 0.05
    subsample_seed = int(cfg.get("subsample_seed", 33))   # deterministic
    raw = _deterministic_subsample(raw, data_frac, subsample_seed)

    # ── split ───────────────────────────────────────────────────────────
    if mode == "infer":
        tr_split, va_split, te_split = raw, [], []
    else:
        n = len(raw)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        n_test  = n - n_train - n_val

        g = torch.Generator().manual_seed(33)
        tr_split, va_split, te_split = random_split(raw, [n_train, n_val, n_test], generator=g)

    # ── tokenizers ─────────────────────────────────────────────────────
    tok_src = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()

    out_dir = Path(cfg["model_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_src.save(str(out_dir / "tokenizer_ai.json"))
    tok_tgt.save(str(out_dir / "tokenizer_tgt.json"))

    # ── dataset wrapper ────────────────────────────────────────────────
    def wrap(split, *, augment: bool) -> TransformerDataset:
        return TransformerDataset(
            split,
            tok_src,
            tok_tgt,
            cfg["seq_len_ai"],
            cfg["seq_len_tgt"],
            cfg["num_heads"],
            cfg["ai_rate"],
            pad_token=PAD_ID,
            augment_permute_obtained=augment,
            lto_len=4,
            obtained_len=10,
            prev_dec_len=1,
            keep_zeros_tail=True,
            base_seed=33,
        )

    def make_loader(ds, *, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    # ── inference: single loader ───────────────────────────────────────
    if mode == "infer":
        inf_ds = wrap(tr_split, augment=False)
        return make_loader(inf_ds, shuffle=False), None, None, tok_tgt

    # ── train/val/test ─────────────────────────────────────────────────
    augment_train = bool(cfg.get("augment_train", True))  # <<< IMPORTANT TOGGLE
    train_ds = wrap(tr_split, augment=augment_train)
    cfg["num_users"] = train_ds.num_users
    
    rep = int(cfg.get("permute_repeat", 1))
    if rep > 1:
        train_ds = RepeatWithPermutation(train_ds, repeat_factor=rep)

    val_ds  = wrap(va_split, augment=False)
    test_ds = wrap(te_split, augment=False)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds, shuffle=False)
    test_loader  = make_loader(test_ds, shuffle=False)

    return train_loader, val_loader, test_loader, tok_tgt

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

def _upload_and_unlink(local: Path, bucket: str, key: str, s3, *, gzip_json: bool=False) -> bool:
    """
    Uploads `local` to s3://bucket/key and unlinks local on success.
    If gzip_json=True, sets JSON + gzip headers.
    """
    if s3 is None or not local.exists():
        return False
    extra = {}
    if gzip_json:
        extra["ExtraArgs"] = {"ContentType": "application/json", "ContentEncoding": "gzip"}
    try:
        if extra:
            s3.upload_file(str(local), bucket, key, **extra)
        else:
            s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local} → s3://{bucket}/{key}")
        try:
            local.unlink()
        except Exception:
            pass
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-ERR] {e}")
        return False

def _unwrap_model(m):
    """Return underlying nn.Module if wrapped by DeepSpeed engine."""
    return m.module if hasattr(m, "module") else m

def _projection_mix_space(m) -> str:
    """
    Try to read model.projection_mix_space; fall back to 'prob'.
    """
    mm = _unwrap_model(m)
    return getattr(mm, "projection_mix_space", "prob")

def _to_logits_like(output: torch.Tensor, mix_space: str, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert model output to a 'logits-like' tensor usable by CE/focal/eval logic.
    - if mix_space='logit': return as-is
    - if mix_space='prob' : return log(prob), since CE(log p, y) == NLL(p, y)
    """
    if mix_space == "logit":
        return output
    if mix_space == "prob":
        return torch.log(output.clamp_min(eps))
    raise ValueError(f"Unknown mix_space={mix_space}")

def _align_labels_to_slots(
    tgt_full: torch.Tensor,     # (B, n_slots) or (B, Tsrc)
    x: torch.Tensor,            # (B, Tsrc)
    n_slots: int,
    pos: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Align labels to decision slots. Returns (B, n_slots).
    """
    if tgt_full.size(1) == n_slots:
        tgt_dec = tgt_full
    elif tgt_full.size(1) == x.size(1):
        tgt_dec = tgt_full[:, pos]
    elif tgt_full.size(1) > n_slots:
        tgt_dec = tgt_full[:, :n_slots]
    else:
        tgt_dec = F.pad(tgt_full, (0, n_slots - tgt_full.size(1)), value=pad_id)
    return tgt_dec

def _forward_model_for_eval(
    model,
    x: torch.Tensor,
    user_ids: Optional[torch.Tensor],
    projection_gate_mode: str,
):
    """
    Forward with the new transformer signature.
    """
    if user_ids is None:
        return model(x, projection_gate_mode=projection_gate_mode)
    return model(x, user_ids, projection_gate_mode=projection_gate_mode)

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
        max_seq_len=cfg["seq_len_ai"],
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        num_users=cfg["num_users"],
        dropout=cfg["dropout"],
        nb_features=cfg["nb_features"],
        kernel_type=cfg["kernel_type"],
        d_ff=cfg["d_ff"],
        feature_tensor=feat_tensor,
        special_token_ids=SPECIAL_IDS,
        projection_mix_space=cfg.get("projection_mix_space", "prob"),  # NEW
    )
# ══════════════════════════════ 9. Evaluation ════════════════════════
DECISION_CLASSES = np.arange(1, 10, dtype=np.int64)  # decisions 1..9

def _macro_f1_from_counts(tp: np.ndarray, pred_cnt: np.ndarray, true_cnt: np.ndarray) -> float:
    """
    Macro F1 over the 9 decision classes.
    Classes with zero support get F1=0 (matches sklearn with zero_division=0).
    """
    f1s = []
    for k in range(len(tp)):
        tpk = tp[k]
        fpk = pred_cnt[k] - tpk
        fnk = true_cnt[k] - tpk

        prec = tpk / (tpk + fpk) if (tpk + fpk) > 0 else 0.0
        rec  = tpk / (tpk + fnk) if (tpk + fnk) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")

@torch.no_grad()
def evaluate(
    loader,
    model,
    dev: torch.device,
    *,
    ai_rate: int,
    # logit_bias: torch.Tensor | None = None,
    logit_bias_9: torch.Tensor | None = None,
    calibrator: VectorScaling | None = None,
    compute_nll: bool = True,
    compute_rev_mae: bool = True,
    projection_gate_mode: str = "mean",   # NEW
) -> dict:
    """
    Returns metrics computed on decision positions only (labels in 1..9).

    Supports models whose output projection returns either:
      - logits  (mix_space='logit')
      - probs   (mix_space='prob')  -> internally converted to log-prob logits-like

    logit_bias:
      If provided, we do: logits_corr = logits_like - logit_bias
      (analytic class-weight correction; valid when using weighted CE training).
    """
    if loader is None:
        return {"nll": float("nan"), "hit": float("nan"), "f1_macro": float("nan"),
                "auprc_macro": float("nan"), "rev_mae": float("nan")}

    model.eval()
    mix_space = _projection_mix_space(model)

    # streaming counters for hit + macro-F1 over classes 1..9
    tp = np.zeros(9, dtype=np.int64)
    pred_cnt = np.zeros(9, dtype=np.int64)
    true_cnt = np.zeros(9, dtype=np.int64)
    correct = 0
    total = 0

    # AUPRC needs all scores/labels
    y_true_chunks: list[np.ndarray] = []
    y_score_chunks: list[np.ndarray] = []

    # optional proper scoring / revenue diagnostics
    nll_sum = 0.0
    nll_cnt = 0
    rev_sum = 0.0
    rev_cnt = 0

    rev_vec = torch.tensor([1, 10, 1, 10, 1, 10, 1, 10, 0], device=dev, dtype=torch.float32)

    for batch in loader:
        x = batch["aggregate_input"].to(dev)   # (B, Tsrc)
        tgt_full = batch["label"].to(dev)
        user_ids = batch["user_id"].to(dev) if "user_id" in batch else None

        pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=dev)

        # ──────────────────────────────────────────────────────
        # FIX: Get per-head logits for exact correction
        # ──────────────────────────────────────────────────────
        if user_ids is not None:
            out_tuple = model(
                x, user_ids,
                projection_gate_mode=projection_gate_mode,
                return_proj_alpha=True,
            )
        else:
            out_tuple = model(
                x,
                projection_gate_mode=projection_gate_mode,
                return_proj_alpha=True,
            )

        # out_tuple = (mixed_output, alpha)
        # We need head_logits too — call projection directly
        mm = _unwrap_model(model)

        # Re-forward through decoder to get hidden states
        # (model already computed this, but we need the intermediate)
        # Actually, let's use a cleaner approach: call with return_hidden
        if user_ids is not None:
            raw_out, hidden = model(
                x, user_ids,
                projection_gate_mode=projection_gate_mode,
                return_hidden=True,
            )
        else:
            raw_out, hidden = model(
                x,
                projection_gate_mode=projection_gate_mode,
                return_hidden=True,
            )

        # Now get per-head logits from the projection layer
        proj_result = mm.projection(
            hidden,
            user_idx=user_ids,
            gate_mode=projection_gate_mode,
            return_alpha=True,
            return_head_logits=True,
        )
        # proj_result = (mixed_out, alpha, head_logits)
        _, alpha, head_logits = proj_result  # alpha: (B,H), head_logits: (B,T,H,V)

        # Align to decision slots
        if head_logits.size(1) == x.size(1):
            head_logits = head_logits[:, pos, :, :]   # (B, n_slots, H, V)

        # ──────────────────────────────────────────────────────
        # FIX: Per-head correction in 9-class subspace
        # ──────────────────────────────────────────────────────
        head_logits_dec = head_logits[..., 1:10]      # (B, n_slots, H, 9)

        if calibrator is not None:
            # Calibrator operates on mixed logits (simpler path)
            # First mix the head logits, then calibrate
            alpha_bt = alpha[:, None, :, None]        # (B, 1, H, 1)
            mixed_logits_dec = (alpha_bt * head_logits_dec).sum(dim=2)  # (B, n_slots, 9)
            prob_dec = calibrator(mixed_logits_dec)    # (B, n_slots, 9)
        elif logit_bias_9 is not None:
            # EXACT per-head correction for mixture model:
            # 1. Correct each head's logits in 9-class space
            # 2. Softmax each head separately
            # 3. Mix the corrected probabilities
            bias = logit_bias_9.to(device=dev, dtype=head_logits_dec.dtype)
            corrected_head_logits = head_logits_dec - bias  # broadcast (B,n_slots,H,9) - (9,)
            corrected_head_probs = F.softmax(corrected_head_logits, dim=-1)  # (B,n_slots,H,9)
            alpha_bt = alpha[:, None, :, None]        # (B, 1, H, 1)
            prob_dec = (alpha_bt * corrected_head_probs).sum(dim=2)  # (B, n_slots, 9)
        else:
            # No correction: just softmax each head in 9-class space, then mix
            head_probs = F.softmax(head_logits_dec, dim=-1)  # (B, n_slots, H, 9)
            alpha_bt = alpha[:, None, :, None]
            prob_dec = (alpha_bt * head_probs).sum(dim=2)    # (B, n_slots, 9)

        pred_dec = prob_dec.argmax(dim=-1) + 1                       # labels in 1..9

        tgt_dec = _align_labels_to_slots(
            tgt_full=tgt_full,
            x=x,
            n_slots=prob_dec.size(1),
            pos=pos,
            pad_id=PAD_ID,
        )

        mask = (tgt_dec >= 1) & (tgt_dec <= 9)
        if mask.sum().item() == 0:
            continue

        y_true = tgt_dec[mask].long()
        y_pred = pred_dec[mask].long()

        scores_9 = prob_dec[mask]  # (N,9)

        # --- hit ---
        total += y_true.numel()
        correct += (y_true == y_pred).sum().item()

        # --- macro-F1 counts ---
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        for c in range(1, 10):
            idx = c - 1
            tmask = (y_true_np == c)
            pmask = (y_pred_np == c)
            true_cnt[idx] += int(tmask.sum())
            pred_cnt[idx] += int(pmask.sum())
            tp[idx] += int((tmask & pmask).sum())

        # --- AUPRC chunks ---
        y_true_chunks.append(y_true_np.astype(np.int64))
        y_score_chunks.append(scores_9.detach().float().cpu().numpy())

        # --- NLL ---
        if compute_nll:
            logp = torch.log(prob_dec + 1e-12)[mask]
            y0 = (y_true - 1).clamp(0, 8)
            lp = logp.gather(1, y0.unsqueeze(1)).squeeze(1)
            nll_sum += (-lp).sum().item()
            nll_cnt += lp.numel()

        if compute_rev_mae:
            exp_rev = (scores_9 * rev_vec.to(dtype=scores_9.dtype)).sum(-1)
            true_rev = rev_vec[(y_true - 1).clamp(0, 8)].to(dtype=exp_rev.dtype)
            rev_sum += torch.abs(exp_rev - true_rev).sum().item()
            rev_cnt += exp_rev.numel()

    if total == 0:
        return {"nll": float("nan"), "hit": float("nan"), "f1_macro": float("nan"),
                "auprc_macro": float("nan"), "rev_mae": float("nan")}

    hit = correct / total
    f1_macro = _macro_f1_from_counts(tp, pred_cnt, true_cnt)

    y_true_all = np.concatenate(y_true_chunks, axis=0)
    y_score_all = np.concatenate(y_score_chunks, axis=0)  # (N,9)
    y_bin = label_binarize(y_true_all, classes=DECISION_CLASSES)
    # auprc_macro = float(average_precision_score(y_bin, y_score_all, average="macro"))
    try:
        auprc = float(average_precision_score(y_bin, y_score_all, average="macro"))
    except Exception:
        auprc = float("nan")

    out = {
        "hit": float(hit),
        "f1_macro": float(f1_macro),
        "auprc_macro": float(auprc),
        "nll": float(nll_sum / max(1, nll_cnt)) if compute_nll else float("nan"),
        "rev_mae": float(rev_sum / max(1, rev_cnt)) if compute_rev_mae else float("nan"),
    }
    return out


def collect_val_logits(val_loader, model, device, ai_rate, projection_gate_mode="mean"):
    """Collect mixed 9-class decision logits from val set for calibrator fitting."""
    model.eval()
    mm = _unwrap_model(model)
    logits_chunks, label_chunks = [], []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["aggregate_input"].to(device)
            tgt_full = batch["label"].to(device)
            user_ids = batch["user_id"].to(device) if "user_id" in batch else None
            pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=device)

            # Get hidden states
            if user_ids is not None:
                _, hidden = model(x, user_ids, projection_gate_mode=projection_gate_mode, return_hidden=True)
            else:
                _, hidden = model(x, projection_gate_mode=projection_gate_mode, return_hidden=True)

            # Get per-head logits
            proj_result = mm.projection(hidden, user_idx=user_ids, gate_mode=projection_gate_mode,
                                        return_alpha=True, return_head_logits=True)
            _, alpha, head_logits = proj_result

            if head_logits.size(1) == x.size(1):
                head_logits = head_logits[:, pos, :, :]

            # Mix head logits in 9-class space (for calibrator input)
            head_logits_dec = head_logits[..., 1:10]
            alpha_bt = alpha[:, None, :, None]
            mixed_logits_dec = (alpha_bt * head_logits_dec).sum(dim=2)  # (B, n_slots, 9)

            n_slots = mixed_logits_dec.size(1)
            tgt_dec = _align_labels_to_slots(tgt_full, x, n_slots, pos, PAD_ID)
            mask = (tgt_dec >= 1) & (tgt_dec <= 9)
            if mask.sum() == 0:
                continue

            logits_chunks.append(mixed_logits_dec[mask])
            label_chunks.append(tgt_dec[mask].long())

    return torch.cat(logits_chunks), torch.cat(label_chunks)

# ══════════════════════════════ 10. Training loop ════════════════════
def train_model(cfg: Dict[str, Any],
                report_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
                stop_check_fn: Optional[Callable[[], bool]] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    uid = (
        f"mixture2_performerfeatures{cfg['nb_features']}"
        f"_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}"
        f"_heads{cfg['num_heads']}_lr{cfg['lr']}_w{cfg['weight']}"
        f"_fold{cfg['fold_id']}"          # <-- add this
    )

    # --- artefact folders --------------------------------------------------
    ckpt_dir    = Path(cfg["model_folder"]) / "checkpoints"
    metrics_dir = Path(cfg["model_folder"]) / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir   / f"FullProductGPT_{uid}.pt"
    json_path = metrics_dir / f"FullProductGPT_{uid}.json"

    s3 = _s3_client()
    bucket = cfg["s3_bucket"]
    ck_key = f"Mixture2/performer/FeatureBased/checkpoints/{ckpt_path.name}"
    js_key = f"Mixture2/performer/FeatureBased/metrics/{json_path.name}"
    print(f"[INFO] artefacts → s3://{bucket}/{ck_key} and s3://{bucket}/{js_key}")

    # --- data ----------------------------------------------------------
    train_dl, val_dl, test_dl, tok_tgt = build_dataloaders(cfg)
    pad_id = tok_tgt.token_to_id("[PAD]")

    base_train_ds = train_dl.dataset.base if hasattr(train_dl.dataset, "base") else train_dl.dataset

    train_uid_to_index = dict(base_train_ds.uid_to_index)
    train_index_to_uid = dict(base_train_ds.index_to_uid)

    # ── optional deterministic subsample for hyperparam search ──
    # cfg["data_frac"] in (0,1] e.g., 0.05 for 5% of records
    # data_frac = float(cfg.get("data_frac", 1.0))
    # subsample_seed = int(cfg.get("subsample_seed", 33))
    # raw = _deterministic_subsample(raw, data_frac, subsample_seed)

    # --- model ---------------------------------------------------------
    feat_tensor = load_feature_tensor(FEAT_FILE)
    model = build_model(cfg, feat_tensor).to(device)

    tau = float(cfg.get("tau", 0.5))
    dec_weights_9 = compute_class_weights(train_dl, num_classes=9, tau=tau, pad_id=pad_id)

    # Map into full vocab for loss function
    full_weights = torch.ones(cfg["vocab_size_tgt"], device=device)
    for c in range(9):
        full_weights[c + 1] = dec_weights_9[c]

    loss_fn = FocalLoss(cfg.get("gamma", 0.0), pad_id, full_weights)

    # Per-class logit bias for analytic correction (9-dim)
    logit_bias_9 = torch.log(dec_weights_9).to(device=device, dtype=torch.float32)
    print(f"[INFO] loss=FocalLoss(gamma={cfg.get('gamma', 0)}) with per-class weights (tau={tau})")
    print(f"[INFO] logit_bias_9 = {logit_bias_9.cpu().numpy().round(4)}")

    # ---- DeepSpeed ----------------------------------------------------
    ds_cfg = {
        "train_micro_batch_size_per_gpu": cfg["batch_size"],
        "zero_allow_untested_optimizer": True,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        # "optimizer": {
        #     "type": "Lamb",
        #     "params": {"lr": cfg["lr"], "eps": cfg["eps"], "weight_decay": cfg["weight_decay"]},
        # },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": cfg["lr"], "betas": [0.9, 0.999], "eps": cfg["eps"], "weight_decay": cfg["weight_decay"]},
        },
        "zero_optimization": {"stage": 1},
        # "fp16": {"enabled": True, "loss_scale": 0, "hysteresis": 2, "min_loss_scale": 1},
        "fp16": {"enabled": False},
        "lr_scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": cfg["min_lr"],
                "warmup_max_lr": cfg["lr"],
                "warmup_num_steps": cfg["warmup_steps"],
                "total_num_steps": cfg["num_epochs"] * len(train_dl),
                "decay_style": "cosine",
            },
        },
    }

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)

    # ---- (optional) preload ------------------------------------------
    preload = cfg["preload"]
    if preload:
        file = latest_weights_file_path(cfg) if preload == "latest" else get_weights_file_path(cfg, preload)
        if file and Path(file).exists():
            print("Preloading", file)
            state = torch.load(file, map_location="cpu")
            engine.module.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])

    # ---- training -----------------------------------------------------
    best_val_loss, patience = None, 0

    # ---- training -----------------------------------------------------
    best_val_metrics = {}
    best_val_nll = None
    best_val_epoch = -1
    patience = 0

    train_proj_gate_mode = cfg.get("train_projection_gate_mode", "user")
    eval_proj_gate_mode  = cfg.get("eval_projection_gate_mode", "mean")
    mix_space = _projection_mix_space(engine)   # reads engine.module.projection_mix_space if present

    for ep in range(cfg["num_epochs"]):
        if hasattr(train_dl.dataset, "set_epoch"):
            train_dl.dataset.set_epoch(ep)

        engine.train()

        # Existing decoder attention gate behavior (if your UserHeadGate supports it)
        if hasattr(engine.module, "gate"):
            if hasattr(engine.module.gate, "use_mean_gate"):
                engine.module.gate.use_mean_gate = False

        running = 0.0
        seen_user_ids = set()  # for projection mean gate cache (training users seen this epoch)

        for batch in tqdm(train_dl, desc=f"Ep {ep:02d}"):
            x = batch["aggregate_input"].to(device)
            tgt_full = batch["label"].to(device)
            u = batch["user_id"].to(device) if "user_id" in batch else None

            if u is not None:
                seen_user_ids.update(u.detach().cpu().tolist())

            pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)

            # NEW: call model with user IDs + projection gate mode
            raw_full = engine(x, u, projection_gate_mode=train_proj_gate_mode) if u is not None else engine(
                x, projection_gate_mode=train_proj_gate_mode
            )

            # Convert to logits-like so existing CE/focal losses still work even if output is prob
            logits_like_full = _to_logits_like(raw_full, mix_space=mix_space)

            # Align logits to decision slots
            if logits_like_full.size(1) == x.size(1):
                logits = logits_like_full[:, pos, :]   # (B, n_slots, V)
            else:
                logits = logits_like_full              # assume already (B, n_slots, V)

            # Align labels to slots (more robust than assuming already aligned)
            tgt = _align_labels_to_slots(
                tgt_full=tgt_full,
                x=x,
                n_slots=logits.size(1),
                pos=pos,
                pad_id=pad_id,
            )

            # Skip batches with no labels
            if not (tgt != pad_id).any():
                continue

            loss = loss_fn(logits, tgt)

            engine.zero_grad()
            engine.backward(loss)
            engine.step()
            running += loss.item()

        # ---- update decoder mean gate (existing gate) ----
        if hasattr(engine.module, "gate"):
            if hasattr(engine.module.gate, "update_mean_gate"):
                engine.module.gate.update_mean_gate()
            if hasattr(engine.module.gate, "use_mean_gate"):
                engine.module.gate.use_mean_gate = True

        # ---- update/cached mean projection gate (NEW) ----
        if eval_proj_gate_mode == "mean" and hasattr(engine.module, "set_projection_mean_gate_from_train_users"):
            if len(seen_user_ids) > 0:
                engine.module.set_projection_mean_gate_from_train_users(sorted(seen_user_ids))

        # ---- validation ----------------------------------------------
        if cfg.get("gamma", 0.0) != 0.0 and logit_bias is not None:
            raise RuntimeError("Analytic correction must be OFF when gamma>0 (focal).")

        # ──────────────────────────────────────────────────────
        # CHANGED: pass logit_bias_9 for per-head correction
        # ──────────────────────────────────────────────────────
        v = evaluate(val_dl, engine.module, device, ai_rate=cfg["ai_rate"],
                     logit_bias_9=logit_bias_9, calibrator=None,
                     compute_nll=True, compute_rev_mae=False,
                     projection_gate_mode=eval_proj_gate_mode)
        
        print(
            f"Epoch {ep:02d}  TrainLoss={running / max(1, len(train_dl)):.4f}  "
            f"ValNLL={v['nll']:.4f}  Hit={v['hit']:.4f}  "
            f"F1={v['f1_macro']:.4f}  AUPRC={v['auprc_macro']:.4f}"
        )

        if report_fn is not None:
            report_fn({
                "epoch": ep,
                "train_loss": running / max(1, len(train_dl)),
                "val_nll": v["nll"],
                "val_hit": v["hit"],
                "val_f1_macro": v["f1_macro"],
                "val_auprc_macro": v["auprc_macro"],
            })

        if stop_check_fn is not None and stop_check_fn():
            print("[INFO] External early-stop triggered.")
            break

        # ---- model selection / early stopping ----
        val_nll = v["nll"]
        if best_val_nll is None or val_nll < best_val_nll:
            best_val_nll, patience, best_val_epoch = val_nll, 0, ep
            best_val_metrics = {
                "val_nll": best_val_nll, "val_epoch": ep,
                "val_hit": v["hit"], "val_f1_macro": v["f1_macro"], "val_auprc_macro": v["auprc_macro"],
                "class_weights_9": dec_weights_9.cpu().tolist(),
                "logit_bias_9": logit_bias_9.cpu().tolist(),
                "tau": tau, "projection_mix_space": mix_space,
            }
            ckpt = {"epoch": ep, "best_val_nll": best_val_nll, "model_state_dict": engine.module.state_dict(),
                    "class_weights_9": dec_weights_9.cpu().tolist(),
                    "logit_bias_9": logit_bias_9.cpu().tolist(), "tau": tau,
                    "projection_mix_space": mix_space,
                    "train_uid_to_index": train_uid_to_index,
                    "train_index_to_uid": train_index_to_uid,
                    }
            torch.save(ckpt, ckpt_path)
            weights_csv = ckpt_dir / f"FullProductGPT_{uid}_user_mix_weights.csv"
            export_user_mixture_weights(
                ckpt["model_state_dict"],
                train_index_to_uid,
                weights_csv,
            )
            json_path.write_text(json.dumps(_json_safe(best_val_metrics), indent=2))
            if s3:
                s3.upload_file(str(json_path), bucket, js_key, ExtraArgs={"ContentType": "application/json"})
                s3.upload_file(str(ckpt_path), bucket, ck_key)
                s3.upload_file(str(weights_csv), bucket, f"Mixture2/performer/FeatureBased/weights/{weights_csv.name}")
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping.")
                break

    # Ensure local checkpoint is present for evaluation
    if not ckpt_path.exists() and s3 is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, ck_key, str(ckpt_path))
            print(f"[S3] downloaded best ckpt for test → s3://{bucket}/{ck_key}")
        except Exception as e:
            print(f"[WARN] Could not download ckpt for test: {e}")
    
    # --- test with calibrator ---
    t = {"nll": float("nan"), "hit": float("nan"), "f1_macro": float("nan"), "auprc_macro": float("nan"), "rev_mae": float("nan")}

    if ckpt_path.exists() or (s3 and not ckpt_path.exists()):
        if not ckpt_path.exists() and s3:
            try:
                s3.download_file(bucket, ck_key, str(ckpt_path))
            except Exception:
                pass

        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            engine.module.load_state_dict(state["model_state_dict"])

            # Fit calibrator
            print("\n[INFO] Fitting vector scaling calibrator...")
            logits_val, labels_val = collect_val_logits(val_dl, engine.module, device,
                                                        cfg["ai_rate"], eval_proj_gate_mode)
            calibrator = fit_vector_scaling(logits_val, labels_val)

            cal_path = ckpt_dir / f"calibrator_{uid}.pt"
            torch.save({"a": calibrator.a.data, "b": calibrator.b.data}, cal_path)

            t = evaluate(test_dl, engine.module, device, ai_rate=cfg["ai_rate"],
                         logit_bias_9=None, calibrator=calibrator,
                         compute_nll=True, compute_rev_mae=True,
                         projection_gate_mode=eval_proj_gate_mode)

            print(f"\n** TEST (calibrated) ** NLL={t['nll']:.4f}  Hit={t['hit']:.4f}  "
                  f"F1={t['f1_macro']:.4f}  AUPRC={t['auprc_macro']:.4f}  RevMAE={t['rev_mae']:.4f}")

# --- inference ---
    if cfg.get("do_infer", True):
        inf_dl, _, _, _ = build_dataloaders({**cfg, "mode": "infer"})
        tmp_pred = Path("/tmp") / f"{uid}_predictions.jsonl.gz"
        mm = _unwrap_model(engine)

        with gzip.open(tmp_pred, "wt", encoding="utf-8") as fp, torch.no_grad():
            for batch in tqdm(inf_dl, desc="Infer"):
                x = batch["aggregate_input"].to(device)
                uids = batch["uid"]
                u = batch["user_id"].to(device) if "user_id" in batch else None
                pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)

                if u is not None:
                    _, hidden = engine(x, u, projection_gate_mode="mean", return_hidden=True)
                else:
                    _, hidden = engine(x, projection_gate_mode="mean", return_hidden=True)

                proj_result = mm.projection(hidden, user_idx=u, gate_mode="mean",
                                            return_alpha=True, return_head_logits=True)
                _, alpha, head_logits = proj_result

                if head_logits.size(1) == x.size(1):
                    head_logits = head_logits[:, pos, :, :]

                head_logits_dec = head_logits[..., 1:10]

                if calibrator is not None:
                    alpha_bt = alpha[:, None, :, None]
                    mixed = (alpha_bt * head_logits_dec).sum(dim=2)
                    probs_dec = calibrator(mixed)
                elif logit_bias_9 is not None:
                    bias = logit_bias_9.to(device=dev, dtype=head_logits_dec.dtype)
                    corrected = F.softmax(head_logits_dec - bias, dim=-1)
                    alpha_bt = alpha[:, None, :, None]
                    probs_dec = (alpha_bt * corrected).sum(dim=2)
                else:
                    hp = F.softmax(head_logits_dec, dim=-1)
                    alpha_bt = alpha[:, None, :, None]
                    probs_dec = (alpha_bt * hp).sum(dim=2)

                for i, uid_str in enumerate(uids):
                    fp.write(json.dumps({"uid": uid_str, "probs_dec_1to9": probs_dec[i].cpu().numpy().tolist()}) + "\n")

        _upload_and_unlink(tmp_pred, bucket, f"CV/predictions/{tmp_pred.name}", s3, gzip_json=True)

    # ------------------ Final metadata (optional) ------------------
    final_meta = {
        "best_checkpoint_path": ckpt_path.name,
        **best_val_metrics,
        "test_nll": t["nll"],
        "test_hit": t["hit"],
        "test_f1_macro": t["f1_macro"],
        "test_auprc_macro": t["auprc_macro"],
        "test_rev_mae": t["rev_mae"],
    }

    final_meta_path = metrics_dir / f"FullProductGPT_{uid}_final.json"
    final_meta_path.write_text(json.dumps(_json_safe(final_meta), indent=2))
    _upload_and_unlink(final_meta_path, bucket, f"Mixture2/performer/FeatureBased/metrics/{final_meta_path.name}", s3)

    # Nothing else to save locally; the "best" ckpt & metrics were already sent to S3
    ckpt_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)

    # Return correct S3 object names
    return {
        "uid": uid,
        "fold_id": cfg["fold_id"],
        "best_val_nll": float(best_val_nll) if best_val_nll is not None else float("nan"),
        "val_hit": float(best_val_metrics.get("val_hit", float("nan"))),
        "val_f1_macro": float(best_val_metrics.get("val_f1_macro", float("nan"))),
        "val_auprc_macro": float(best_val_metrics.get("val_auprc_macro", float("nan"))),
        "test_hit": float(t["hit"]),
        "test_f1_macro": float(t["f1_macro"]),
        "test_auprc_macro": float(t["auprc_macro"]),
        "ckpt": ck_key.split("/")[-1],
        "preds": f"{uid}_predictions.jsonl.gz",
    }

# ══════════════════════════════ 11. CLI ═══════════════════════════════
if __name__ == "__main__":
    cfg = get_config()
    cfg.setdefault("projection_mix_space", "prob")   # "logit" or "prob"
    cfg.setdefault("train_projection_gate_mode", "user")
    cfg.setdefault("eval_projection_gate_mode", "mean")   # "mean" or "user"
    cfg.setdefault("infer_projection_gate_mode", "mean")
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]
    train_model(cfg)