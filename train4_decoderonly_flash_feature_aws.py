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
from dataset4_productgpt import TransformerDataset, load_json_dataset
from model4_decoderonly_feature_flash import build_transformer

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

# ══════════════════════════════ NEW: Class weight computation ═════════
def compute_class_weights(
    train_loader: DataLoader,
    num_classes: int = 9,
    tau: float = 0.5,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Count decision-class frequencies from the training set,
    return inverse-frequency weights with temperature tau.

    Args:
        train_loader: training DataLoader
        num_classes:  9 (decisions 1..9)
        tau:          temperature; 0 = uniform, 1 = full inverse-frequency
        pad_id:       label to ignore

    Returns:
        (9,) float tensor, index 0 → class 1, ..., index 8 → class 9
        Normalized so that mean weight = 1.0.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)

    for batch in train_loader:
        labels = batch["label"].view(-1)
        for c in range(1, 10):
            counts[c - 1] += (labels == c).sum().item()

    # Guard against zero counts
    counts = counts.clamp(min=1.0)

    N = counts.sum()
    K = float(num_classes)
    freq = counts / N                                      # (9,)
    raw_weight = 1.0 / (K * freq)                          # (9,)
    tempered = raw_weight.pow(tau)                          # (9,)

    # Normalize so mean weight = 1 (keeps loss scale stable)
    tempered = tempered / tempered.mean()

    print(f"[INFO] Class frequencies:  {(freq * 100).numpy().round(2)}%")
    print(f"[INFO] Class weights (tau={tau}): {tempered.float().numpy().round(4)}")

    return tempered.float()

# ══════════════════════════════ NEW: Vector Scaling Calibrator ════════
class VectorScaling(nn.Module):
    """
    Post-hoc calibration: learns per-class scale (a) and shift (b)
    on decision logits. 2*K = 18 parameters for K=9 classes.
    """
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits_dec: torch.Tensor) -> torch.Tensor:
        """
        logits_dec: (..., 9) raw decision logits (NOT softmaxed)
        returns:    (..., 9) calibrated probabilities summing to 1
        """
        return F.softmax(self.a * logits_dec + self.b, dim=-1)


def fit_vector_scaling(
    logits_val: torch.Tensor,
    labels_val: torch.Tensor,
    n_classes: int = 9,
    max_iter: int = 200,
) -> VectorScaling:
    """
    Fit vector scaling calibrator on validation decision logits.

    Args:
        logits_val: (N, 9) decision logits from val set (raw, no correction)
        labels_val: (N,)   ground truth labels in {1..9}
        n_classes:  9
        max_iter:   LBFGS iterations

    Returns:
        Fitted VectorScaling module
    """
    cal = VectorScaling(n_classes)

    # Move to same device as input
    device = logits_val.device
    cal = cal.to(device)

    labels_0based = (labels_val - 1).clamp(0, n_classes - 1).long()

    optimizer = torch.optim.LBFGS(cal.parameters(), lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        p = cal(logits_val)
        loss = F.nll_loss(torch.log(p + 1e-12), labels_0based)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Report fitted parameters
    print(f"[CALIBRATOR] a = {cal.a.data.cpu().numpy().round(4)}")
    print(f"[CALIBRATOR] b = {cal.b.data.cpu().numpy().round(4)}")

    return cal


def collect_val_logits(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    ai_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model on validation set, collect raw 9-class decision logits and labels.

    Returns:
        logits_all: (N, 9) raw decision logits
        labels_all: (N,)   labels in {1..9}
    """
    model.eval()
    logits_chunks = []
    label_chunks = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["aggregate_input"].to(device)
            tgt_full = batch["label"].to(device)

            pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=device)
            logits_full = model(x)

            if logits_full.size(1) == x.size(1):
                logits = logits_full[:, pos, :]
            else:
                logits = logits_full

            # Extract 9-class decision logits
            V_out = logits.size(-1)
            if V_out == 9:
                logits_dec = logits
            else:
                logits_dec = logits[..., 1:10]       # (B, n_slots, 9)

            # Align labels
            n_slots = logits_dec.size(1)
            if tgt_full.size(1) == n_slots:
                tgt_dec = tgt_full
            elif tgt_full.size(1) == x.size(1):
                tgt_dec = tgt_full[:, pos]
            elif tgt_full.size(1) > n_slots:
                tgt_dec = tgt_full[:, :n_slots]
            else:
                tgt_dec = F.pad(tgt_full, (0, n_slots - tgt_full.size(1)), value=PAD_ID)

            mask = (tgt_dec >= 1) & (tgt_dec <= 9)
            if mask.sum().item() == 0:
                continue

            logits_chunks.append(logits_dec[mask])       # (N_batch, 9)
            label_chunks.append(tgt_dec[mask].long())    # (N_batch,)

    return torch.cat(logits_chunks, dim=0), torch.cat(label_chunks, dim=0)

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

# ══════════════════════════════ 5. Utility fns ════════════════════════
def transition_mask(seq: torch.Tensor) -> torch.Tensor:  # (B, T)
    """Mask where *decision* changes wrt previous step."""
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev

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

# ══════════════════════════════ 8. Model builder ═════════════════════=
def build_model(cfg: Dict[str, Any], feat_tensor: torch.Tensor) -> nn.Module:
    return build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_ai"],
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
) -> dict:
    """
    Returns metrics computed on decision positions only (labels in 1..9).

    Supports BOTH model output styles:
      1) full-vocab logits (e.g., V=60, decision labels are token IDs 1..9)
      2) decision-only logits (V=9, corresponding to decision labels 1..9)

    logit_bias:
      If provided, subtracts a bias vector before softmax / NLL.
      - full-vocab case: bias is expected to align with full vocab (V)
      - decision-only case (V=9): if a full-vocab bias is provided, we use bias[1:10]

    Metrics returned:
      hit (accuracy), f1_macro, auprc_macro, plus optional nll and rev_mae.
    """
    if loader is None:
        return {
            "nll": float("nan"),
            "hit": float("nan"),
            "f1_macro": float("nan"),
            "auprc_macro": float("nan"),
            "rev_mae": float("nan"),
        }

    model.eval()

    # streaming counters for hit + macro-F1 over classes 1..9
    tp = np.zeros(9, dtype=np.int64)
    pred_cnt = np.zeros(9, dtype=np.int64)
    true_cnt = np.zeros(9, dtype=np.int64)
    correct = 0
    total = 0

    # AUPRC needs all scores/labels (only 9-class scores)
    y_true_chunks: list[np.ndarray] = []
    y_score_chunks: list[np.ndarray] = []

    # optional proper scoring / revenue diagnostics
    nll_sum = 0.0
    nll_cnt = 0
    rev_sum = 0.0
    rev_cnt = 0

    # revenue vector for decisions 1..9
    # decisions 1..8 alternate {1,10}; decision 9 -> 0
    rev_vec = torch.tensor([1, 10, 1, 10, 1, 10, 1, 10, 0], device=dev, dtype=torch.float32)

    def _align_labels_to_slots(
        tgt_full: torch.Tensor,
        n_slots: int,
        x_len: int,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Align labels to decision-slot axis."""
        if tgt_full.size(1) == n_slots:
            return tgt_full
        elif tgt_full.size(1) == x_len:
            return tgt_full[:, pos]
        elif tgt_full.size(1) > n_slots:
            return tgt_full[:, :n_slots]
        else:
            return F.pad(tgt_full, (0, n_slots - tgt_full.size(1)), value=PAD_ID)

    for batch in loader:
        x = batch["aggregate_input"].to(dev)   # (B, Tsrc)
        tgt_full = batch["label"].to(dev)      # (B, n_slots) OR (B, Tsrc)

        # decision positions on source/time axis
        pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=dev)

        # model output: either (B, Tsrc, V) or already (B, n_slots, V)
        logits_full = model(x)

        if logits_full.size(1) == x.size(1):
            logits = logits_full[:, pos, :]    # (B, n_slots, V_out)
        else:
            logits = logits_full               # (B, n_slots, V_out)

        V_out = logits.size(-1)

        # ──────────────────────────────────────────────────────
        # FIX: Extract decision logits BEFORE softmax
        # ──────────────────────────────────────────────────────
        if V_out == 9:
            logits_dec = logits                              # already 9-class
        else:
            logits_dec = logits[..., 1:10]                   # (B, n_slots, 9)

        # ──────────────────────────────────────────────────────
        # Compute probabilities: either calibrator or analytic
        # ──────────────────────────────────────────────────────
        if calibrator is not None:
            # Post-hoc calibration (takes raw logits, returns probs)
            prob_dec = calibrator(logits_dec)                 # (B, n_slots, 9)
        else:
            # Analytic correction then softmax
            if logit_bias_9 is not None:
                bias = logit_bias_9.to(device=logits_dec.device, dtype=logits_dec.dtype)
                logits_dec = logits_dec - bias                # (B, n_slots, 9) - (9,)
            prob_dec = F.softmax(logits_dec, dim=-1)          # (B, n_slots, 9), sums to 1

        pred = prob_dec.argmax(dim=-1) + 1                    # back to label space 1..9

        tgt_dec = _align_labels_to_slots(tgt_full, prob_dec.size(1), x.size(1), pos)

        mask = (tgt_dec >= 1) & (tgt_dec <= 9)
        if mask.sum().item() == 0:
            continue

        y_true = tgt_dec[mask].long()
        y_pred = pred[mask].long()
        scores_9 = prob_dec[mask]                             # (N, 9)

        # # Apply logit bias safely for both output styles
        # if logit_bias is not None:
        #     bias = logit_bias.to(device=logits.device, dtype=logits.dtype)
        #     if bias.numel() == V_out:
        #         logits = logits - bias
        #     elif V_out == 9 and bias.numel() >= 10:
        #         # full-vocab bias provided, but model outputs decision-only classes 1..9
        #         logits = logits - bias[1:10]
        #     else:
        #         # Shape mismatch; better to fail loudly than silently compute nonsense
        #         raise ValueError(
        #             f"logit_bias shape mismatch: logits last dim={V_out}, "
        #             f"logit_bias numel={bias.numel()}"
        #         )

        # # ------------------------------------------------------------
        # # Support BOTH output styles:
        # #   (A) decision-only logits: V_out == 9  (classes correspond to labels 1..9)
        # #   (B) full-vocab logits:    V_out >= 10 (decision labels are token IDs 1..9)
        # # ------------------------------------------------------------
        # if V_out == 9:
        #     # decision-only logits correspond to labels 1..9
        #     prob_dec = F.softmax(logits, dim=-1)          # (B, n_slots, 9)
        #     pred = prob_dec.argmax(dim=-1) + 1            # back to label space 1..9

        #     tgt_dec = _align_labels_to_slots(tgt_full, prob_dec.size(1), x.size(1), pos)

        #     mask = (tgt_dec >= 1) & (tgt_dec <= 9)        # (B, n_slots)
        #     if mask.sum().item() == 0:
        #         continue

        #     y_true = tgt_dec[mask].long()                 # (N,) in 1..9
        #     y_pred = pred[mask].long()                    # (N,) in 1..9

        #     prob_flat = prob_dec[mask]                    # (N, 9)
        #     if prob_flat.size(-1) != 9:
        #         continue

        #     scores_9 = prob_flat                          # already decision classes 1..9

        # else:
        #     # full-vocab logits
        #     prob = F.softmax(logits, dim=-1)              # (B, n_slots, V_out)
        #     pred = prob.argmax(dim=-1)                    # (B, n_slots), token-ID space

        #     tgt_dec = _align_labels_to_slots(tgt_full, prob.size(1), x.size(1), pos)

        #     mask = (tgt_dec >= 1) & (tgt_dec <= 9)        # only decision labels
        #     if mask.sum().item() == 0:
        #         continue

        #     y_true = tgt_dec[mask].long()                 # (N,) in 1..9
        #     y_pred = pred[mask].long()                    # (N,) token IDs (may be outside 1..9)

        #     prob_flat = prob[mask]                        # (N, V_out)
        #     if prob_flat.size(-1) < 10:
        #         # can't safely slice decision IDs 1..9
        #         continue

        #     scores_9 = prob_flat[:, 1:10]                 # (N, 9), token IDs 1..9

        # --- hit (accuracy on true decision positions) ---
        total += y_true.numel()
        correct += (y_true == y_pred).sum().item()

        # --- macro-F1 counts over classes 1..9 ---
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

        # --- NLL (on corrected logits) ---
        if compute_nll:
            # Use the SAME logits_dec (before or after correction) for NLL
            # For proper NLL, we use the corrected/calibrated distribution
            logp_dec = torch.log(prob_dec + 1e-12)[mask]      # (N, 9)
            y_true_0based = (y_true - 1).clamp(0, 8)
            lp_true = logp_dec.gather(1, y_true_0based.unsqueeze(1)).squeeze(1)
            nll_sum += (-lp_true).sum().item()
            nll_cnt += lp_true.numel()

            # if V_out == 9:
            #     # logits indexed 0..8 correspond to labels 1..9
            #     logp_flat = F.log_softmax(logits, dim=-1)[mask]  # (N, 9)
            #     y_true_0based = (y_true - 1).clamp(0, 8)
            #     lp_true = logp_flat.gather(1, y_true_0based.unsqueeze(1)).squeeze(1)
            # else:
            #     # full vocab logits use token IDs directly
            #     logp_flat = F.log_softmax(logits, dim=-1)[mask]  # (N, V_out)
            #     lp_true = logp_flat.gather(1, y_true.unsqueeze(1)).squeeze(1)

            # nll_sum += (-lp_true).sum().item()
            # nll_cnt += lp_true.numel()

        # --- revenue MAE ---
        if compute_rev_mae:
            exp_rev = (scores_9 * rev_vec.to(dtype=scores_9.dtype)).sum(dim=-1)  # (N,)
            true_rev = rev_vec[(y_true - 1).clamp(0, 8)].to(dtype=exp_rev.dtype)
            rev_sum += torch.abs(exp_rev - true_rev).sum().item()
            rev_cnt += exp_rev.numel()

    if total == 0:
        return {
            "nll": float("nan"),
            "hit": float("nan"),
            "f1_macro": float("nan"),
            "auprc_macro": float("nan"),
            "rev_mae": float("nan"),
        }

    hit = correct / total
    f1_macro = _macro_f1_from_counts(tp, pred_cnt, true_cnt)

    # AUPRC macro
    y_true_all = np.concatenate(y_true_chunks, axis=0)   # (N,)
    y_score_all = np.concatenate(y_score_chunks, axis=0) # (N, 9)
    y_bin = label_binarize(y_true_all, classes=DECISION_CLASSES)  # (N, 9)

    try:
        auprc_macro = float(average_precision_score(y_bin, y_score_all, average="macro"))
    except Exception:
        auprc_macro = float("nan")

    out = {
        "hit": float(hit),
        "f1_macro": float(f1_macro),
        "auprc_macro": float(auprc_macro),
    }
    out["nll"] = float(nll_sum / max(1, nll_cnt)) if compute_nll else float("nan")
    out["rev_mae"] = float(rev_sum / max(1, rev_cnt)) if compute_rev_mae else float("nan")
    return out

# ══════════════════════════════ 10. Training loop ════════════════════
def train_model(cfg: Dict[str, Any],
                report_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
                stop_check_fn: Optional[Callable[[], bool]] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    uid = (
        f"featurebased_flash"
        f"_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}"
        f"_heads{cfg['num_heads']}_lr{cfg['lr']}_w{cfg['weight']}"
        f"_fold{cfg['fold_id']}"
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
    ck_key = f"FullProductGPT/flash/FeatureBased/checkpoints/{ckpt_path.name}"
    js_key = f"FullProductGPT/flash/FeatureBased/metrics/{json_path.name}"
    print(f"[INFO] artefacts → s3://{bucket}/{ck_key} and s3://{bucket}/{js_key}")

    # --- data ----------------------------------------------------------
    train_dl, val_dl, test_dl, tok_tgt = build_dataloaders(cfg)
    pad_id = tok_tgt.token_to_id("[PAD]")

    # --- model ---------------------------------------------------------
    feat_tensor = load_feature_tensor(FEAT_FILE)
    model = build_model(cfg, feat_tensor).to(device)

    # ──────────────────────────────────────────────────────────
    # CHANGED: compute per-class weights from training data
    # ──────────────────────────────────────────────────────────
    tau = float(cfg.get("tau", 0.5))
    dec_weights_9 = compute_class_weights(train_dl, num_classes=9, tau=tau, pad_id=pad_id)

    # Map 9-class weights into full vocab space for loss function
    full_weights = torch.ones(cfg["vocab_size_tgt"], device=device)
    for c in range(9):
        full_weights[c + 1] = dec_weights_9[c]    # class 1 → index 1, ..., class 9 → index 9

    loss_fn = FocalLoss(cfg["gamma"], pad_id, full_weights)

  # ──────────────────────────────────────────────────────────
    # CHANGED: per-class logit bias for analytic correction (all 9 classes)
    # ──────────────────────────────────────────────────────────
    logit_bias_9 = torch.log(dec_weights_9).to(device=device, dtype=torch.float32)  # (9,)

    print(f"[INFO] loss=FocalLoss(gamma={cfg['gamma']}) with per-class weights (tau={tau})")
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
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},      # ← enables FlashAttention-2 on Ampere+ GPUs
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

    best_val_metrics = {
    "val_nll": float("nan"),
    "val_epoch": -1,
    "val_hit": float("nan"),
    "val_f1_macro": float("nan"),
    "val_auprc_macro": float("nan"),        
    "class_weights_9": dec_weights_9.cpu().tolist(),
    # "weight_class9": float(cfg["weight"]),
    # "logit_bias_9": float(logit_bias_9[9].item()),
    "logit_bias_9": logit_bias_9.cpu().tolist(),
    "tau": tau,
}

    best_val_nll = None          # or float("inf") if you prefer
    best_val_epoch = -1
    for ep in range(cfg["num_epochs"]):
        if hasattr(train_dl.dataset, "set_epoch"):
            train_dl.dataset.set_epoch(ep)
        engine.train()
        running = 0.0
        for batch in tqdm(train_dl, desc=f"Ep {ep:02d}"):
            x = batch["aggregate_input"].to(device)
            tgt = batch["label"].to(device)
            pos = torch.arange(cfg["ai_rate"] - 1, cfg["seq_len_ai"], cfg["ai_rate"], device=device)
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
        
        # ---- validation ----------------------------------------------
        # v = evaluate(
        #     val_dl,
        #     engine.module,            # underlying nn.Module
        #     device,
        #     ai_rate=cfg["ai_rate"],
        #     logit_bias=logit_bias,    # analytic odds correction
        #     compute_nll=True,         # selection metric
        #     compute_rev_mae=False,    # keep eval simple
        # )

        # ──────────────────────────────────────────────────────
        # CHANGED: pass logit_bias_9 (9-dim) instead of full-vocab bias
        # ──────────────────────────────────────────────────────
        v = evaluate(
            val_dl,
            engine.module,
            device,
            ai_rate=cfg["ai_rate"],
            logit_bias_9=logit_bias_9,    # (9,) per-class correction
            calibrator=None,              # no calibrator during training
            compute_nll=True,
            compute_rev_mae=False,
        )

        # Basic console log (single line)
        print(
            f"Epoch {ep:02d}  ValNLL={v['nll']:.4f}  "
            f"Hit={v['hit']:.4f}  F1={v['f1_macro']:.4f}  AUPRC={v['auprc_macro']:.4f}"
        )

        # ---- Ray Tune / external reporting ----
        if report_fn is not None:
            report_fn({
                "epoch": ep,
                "val_nll": v["nll"],                 # <<< use this in Ray Tune (mode="min")
                "val_hit": v["hit"],
                "val_f1_macro": v["f1_macro"],
                "val_auprc_macro": v["auprc_macro"],
            })

        # Allow scheduler to terminate the trial early
        if stop_check_fn is not None and stop_check_fn():
            print("[INFO] External early-stop triggered.")
            break

        print(f"[INFO] artefacts → s3://{bucket}/{ck_key} and s3://{bucket}/{js_key}")

        # ---- model selection / early stopping (by corrected unweighted NLL) ----
        val_nll = v["nll"]
        if best_val_nll is None or val_nll < best_val_nll:
            best_val_nll, patience, best_val_epoch = val_nll, 0, ep

            best_val_metrics = {
                "val_nll": best_val_nll,
                "val_epoch": best_val_epoch,
                "val_hit": v["hit"],
                "val_f1_macro": v["f1_macro"],
                "val_auprc_macro": v["auprc_macro"],
                # ──── CHANGED: save all 9 weights and biases ────
                "class_weights_9": dec_weights_9.cpu().tolist(),
                "logit_bias_9": logit_bias_9.cpu().tolist(),
                "tau": tau,
            }

            ckpt = {
                "epoch": ep,
                "best_val_nll": best_val_nll,
                "best_val_epoch": best_val_epoch,
                "model_state_dict": engine.module.state_dict(),
                # ──── CHANGED: save all 9 weights and biases ────
                "class_weights_9": dec_weights_9.cpu().tolist(),
                "logit_bias_9": logit_bias_9.cpu().tolist(),
                "tau": tau,
            }

            # save best checkpoint & metrics
            torch.save(ckpt, ckpt_path)
            json_path.write_text(json.dumps(_json_safe(best_val_metrics), indent=2))

            if s3:
                s3.upload_file(
                    str(json_path), bucket, js_key,
                    ExtraArgs={"ContentType": "application/json"},
                )
                s3.upload_file(str(ckpt_path), bucket, ck_key)

        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping (by val_nll).")
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
    # Default placeholders so final_meta never crashes
    t = {"nll": float("nan"), "hit": float("nan"), "f1_macro": float("nan"),
        "auprc_macro": float("nan"), "rev_mae": float("nan")}

    if ckpt_path.exists():
         state = torch.load(ckpt_path, map_location=device)
         engine.module.load_state_dict(state["model_state_dict"])
         print("\n[INFO] Fitting vector scaling calibrator on validation set...")
         logits_val, labels_val = collect_val_logits(val_dl, engine.module, device, cfg["ai_rate"])
         calibrator = fit_vector_scaling(logits_val, labels_val)
         
         cal_path = ckpt_dir / f"calibrator_{uid}.pt"
         torch.save({"a": calibrator.a.data, "b": calibrator.b.data}, cal_path)
         cal_s3_key = f"FullProductGPT/flash/FeatureBased/checkpoints/{cal_path.name}"
         if s3:
            s3.upload_file(str(cal_path), bucket, cal_s3_key)
            print(f"[S3] calibrator → s3://{bucket}/{cal_s3_key}")

        # Test with calibrator (preferred) AND with analytic correction (for comparison)
         t_cal = evaluate(
            test_dl,
            engine.module,
            device,
            ai_rate=cfg["ai_rate"],
            logit_bias_9=None,
            calibrator=calibrator,
            compute_nll=True,
            compute_rev_mae=True,
        )

         t_analytic = evaluate(
            test_dl,
            engine.module,
            device,
            ai_rate=cfg["ai_rate"],
            logit_bias_9=logit_bias_9,
            calibrator=None,
            compute_nll=True,
            compute_rev_mae=True,
        )

         t = t_cal  # Use calibrated version as primary

         print(
            f"\n** TEST (calibrated) ** NLL={t_cal['nll']:.4f}  Hit={t_cal['hit']:.4f}  "
            f"F1={t_cal['f1_macro']:.4f}  AUPRC={t_cal['auprc_macro']:.4f}  RevMAE={t_cal['rev_mae']:.4f}"
        )
         print(
            f"** TEST (analytic)   ** NLL={t_analytic['nll']:.4f}  Hit={t_analytic['hit']:.4f}  "
            f"F1={t_analytic['f1_macro']:.4f}  AUPRC={t_analytic['auprc_macro']:.4f}  RevMAE={t_analytic['rev_mae']:.4f}"
        )

    # ------------------ inference on full 30 campaigns ------------------
    if cfg.get("do_infer", True):
        inf_dl, _, _, _ = build_dataloaders({**cfg, "mode": "infer"})

        # Write to a gzipped temp file in /tmp to keep root disk small
        tmp_pred = Path("/tmp") / f"{uid}_predictions.jsonl.gz"

        with gzip.open(tmp_pred, "wt", encoding="utf-8") as fp, torch.no_grad():
            for batch in tqdm(inf_dl, desc="Infer 30-campaign set"):
                x   = batch["aggregate_input"].to(device)
                uids = batch["uid"]  # list[str] length B
                logits = engine(x)[:, cfg["ai_rate"]-1::cfg["ai_rate"], :]
                probs  = torch.softmax(logits, -1).cpu().numpy()   # (B, N, 60)
                for u, p in zip(uids, probs):
                    fp.write(json.dumps({"uid": u, "probs": p.tolist()}) + "\n")

        # Upload gzipped predictions and delete local temp
        pred_s3_key = f"CV/predictions/{tmp_pred.name}"
        _upload_and_unlink(tmp_pred, bucket, pred_s3_key, s3, gzip_json=True)

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
    _upload_and_unlink(final_meta_path, bucket, f"FullProductGPT/flash/FeatureBased/metrics/{final_meta_path.name}", s3)

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
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]
    train_model(cfg)