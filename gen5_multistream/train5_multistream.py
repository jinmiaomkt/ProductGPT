"""
Generation-5 trainer: multi-stream state-space Transformer.

This replaces train_multistream_patch.py, which was only a patch guide ("not
meant to be imported as-is; copy the blocks into your trainer") and which
nothing imported. This is a real, runnable trainer.

DELIBERATELY NO DEEPSPEED. Every gen-0..gen-4 trainer imports deepspeed at
module top level, which has no Windows build, so none of them can run on the
laptop. This uses plain PyTorch with AMP and gradient accumulation, so the
same file runs on the 6 GB laptop and on the SMU HPCC cluster unchanged.

Usage
-----
    python gen5_multistream/train5_multistream.py --profile pilot
    python gen5_multistream/train5_multistream.py --profile hpcc

Both read the data location from PRODUCTGPT_DATA (see paths.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import config5
from dataset_multistream import (
    TemporalRoleView,
    TransformerDataset,
    collate_multistream,
    load_json_dataset,
)
from model_multistream_state_space import build_transformer
from shared.features import load_feature_tensor

PAD_ID = 0
N_CLASSES = 9
# Revenue per decision: 1..8 alternate 1 / 10 draws, 9 = NotBuy earns nothing.
REV_VEC = [1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 0.0]

# Decision semantics, from Code/GenerateJSON.R via analysis/analyze_users_campaign28.py.
# 9 is NotBuy -- an ordinary class (the majority one), NOT end-of-sequence.
DECISION_LABELS = {
    1: "Buy1_Reg",  2: "Buy10_Reg",
    3: "Buy1_FigA", 4: "Buy10_FigA",
    5: "Buy1_FigB", 6: "Buy10_FigB",
    7: "Buy1_Wep",  8: "Buy10_Wep",
    9: "NotBuy",
}


# ────────────────────────────── utilities ──────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """bf16 where supported (no loss scaler needed), else fp16, else None."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ────────────────────────────── loss ──────────────────────────────
class FocalLoss(nn.Module):
    """Weighted cross-entropy with an optional focal term. gamma=0 -> plain CE."""

    def __init__(self, gamma: float = 0.0, ignore_index: int = PAD_ID,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        V = logits.size(-1)
        logits = logits.reshape(-1, V)
        targets = targets.reshape(-1)
        w = (self.class_weights.to(dtype=logits.dtype, device=logits.device)
             if self.class_weights is not None else None)
        ce = F.cross_entropy(logits, targets, weight=w,
                             ignore_index=self.ignore_index, reduction="none")
        ce = ce[targets != self.ignore_index]
        if ce.numel() == 0:
            return logits.sum() * 0.0
        if self.gamma > 0:
            ce = ((1 - torch.exp(-ce)) ** self.gamma) * ce
        return ce.mean()


def compute_class_weights(loader: DataLoader, tau: float) -> torch.Tensor:
    """Inverse-frequency weights over classes 1..9, tempered by tau, mean 1."""
    counts = torch.zeros(N_CLASSES, dtype=torch.float64)
    for batch in loader:
        lab = batch["label"].reshape(-1)
        for c in range(1, N_CLASSES + 1):
            counts[c - 1] += (lab == c).sum().item()
    counts = counts.clamp(min=1.0)
    freq = counts / counts.sum()
    w = (1.0 / (N_CLASSES * freq)) ** tau
    w = w / w.mean()
    return w.float()


# ────────────────────────────── metrics ──────────────────────────────
def macro_f1(tp: np.ndarray, pred_cnt: np.ndarray, true_cnt: np.ndarray) -> float:
    out = []
    for k in range(len(tp)):
        fp, fn = pred_cnt[k] - tp[k], true_cnt[k] - tp[k]
        prec = tp[k] / (tp[k] + fp) if (tp[k] + fp) > 0 else 0.0
        rec = tp[k] / (tp[k] + fn) if (tp[k] + fn) > 0 else 0.0
        out.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(out)) if out else float("nan")


def per_class_auprc(y_true: np.ndarray, scores: np.ndarray) -> List[float]:
    """
    Average precision for each of the 9 classes. NaN where a class has no
    support in this split (AP is undefined with no positives).
    """
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return [float("nan")] * N_CLASSES
    out = []
    for k in range(N_CLASSES):
        pos = (y_true == k + 1).astype(np.int8)
        out.append(float(average_precision_score(pos, scores[:, k]))
                   if pos.sum() else float("nan"))
    return out


def macro_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision averaged over classes that have support."""
    vals = [v for v in per_class_auprc(y_true, scores) if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def format_per_class(per_class: Dict[str, Dict[str, Any]]) -> str:
    """Readable table for the job log -- macro numbers hide which classes fail."""
    hdr = (f"{'id':>2}  {'label':<12} {'support':>8} {'pred':>8} "
           f"{'prec':>6} {'recall':>6} {'F1':>6} {'AUPRC':>6}")
    lines = [hdr, "-" * len(hdr)]
    for c in range(1, N_CLASSES + 1):
        d = per_class[str(c)]
        lines.append(
            f"{c:>2}  {d['label']:<12} {d['support']:>8,} {d['predicted']:>8,} "
            f"{d['precision']:>6.3f} {d['recall']:>6.3f} {d['f1']:>6.3f} "
            f"{d['auprc']:>6.3f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def set_unknown_user_to_mean(model: nn.Module, trained_idx: List[int]) -> bool:
    """
    Set the index-0 (unknown customer) embedding to the mean of the rows that
    training actually updated.

    This is Lu & Kannan (JMR 2025): "For out-of-sample customer predictions,
    where head weights are unknown, we use the average head weight from the
    training population omega-bar_h = (1/N) sum_n omega_nh". Averaging learned
    per-customer parameters gives a real population prior; leaving index 0 at
    its initialisation would feed the model an untrained vector for exactly
    the customers the out-of-sample cells are meant to measure.

    Call it before every evaluation, since the mean moves as training goes on.
    Returns False if the model has no user embedding.
    """
    if not trained_idx:
        return False
    done = False

    # Mixture head: average the SOFTMAXED weights, which is the quantity the
    # paper defines. Averaging logits and softmaxing is a different thing.
    head = getattr(model, "mixture_head", None)
    if head is not None:
        head.refresh_mean_alpha(trained_idx)
        done = True

    # Plain embedding: the analogous population mean in vector space.
    emb = getattr(model, "user_embed", None)
    if emb is not None:
        idx = torch.tensor(trained_idx, device=emb.weight.device, dtype=torch.long)
        emb.weight[0] = emb.weight.index_select(0, idx).mean(dim=0)
        done = True
    return done


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             adtype: Optional[torch.dtype]) -> Dict[str, float]:
    model.eval()
    tp = np.zeros(N_CLASSES, dtype=np.int64)
    pred_cnt = np.zeros(N_CLASSES, dtype=np.int64)
    true_cnt = np.zeros(N_CLASSES, dtype=np.int64)
    correct = total = 0
    nll_sum = 0.0
    rev_err = 0.0
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    rev = torch.tensor(REV_VEC, device=device)

    for batch in loader:
        lto = batch["lto"].to(device, non_blocking=True)
        obt = batch["obtained"].to(device, non_blocking=True)
        prev = batch["prev_decision"].to(device, non_blocking=True)
        uid = batch["user_id"].to(device, non_blocking=True)
        tgt = batch["label"].to(device, non_blocking=True)

        ctx = (torch.autocast("cuda", dtype=adtype)
               if adtype is not None else torch.autocast("cpu", enabled=False))
        with ctx:
            logits = model(lto, obt, prev, uid)
        logits = logits.float()

        dec = logits[..., 1:1 + N_CLASSES]           # (B,S,9)
        prob = F.softmax(dec, dim=-1)

        mask = (tgt >= 1) & (tgt <= N_CLASSES)
        if mask.sum() == 0:
            continue
        y = tgt[mask]                                # 1..9
        p = prob[mask]                               # (N,9)
        pred = p.argmax(-1) + 1

        nll_sum += float(-torch.log(p[torch.arange(p.size(0), device=device),
                                     y - 1].clamp_min(1e-12)).sum())
        exp_rev = (p * rev).sum(-1)
        rev_err += float((exp_rev - rev[y - 1]).abs().sum())

        correct += int((pred == y).sum())
        total += int(y.numel())
        for k in range(1, N_CLASSES + 1):
            tp[k - 1] += int(((pred == k) & (y == k)).sum())
            pred_cnt[k - 1] += int((pred == k).sum())
            true_cnt[k - 1] += int((y == k).sum())
        ys.append(y.cpu().numpy())
        ps.append(p.cpu().numpy())

    if total == 0:
        return {k: float("nan") for k in
                ("nll", "hit", "f1_macro", "auprc_macro", "rev_mae", "n")}

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    aps = per_class_auprc(y_all, p_all)

    # Per-class breakdown. Macro averages hide which classes actually work;
    # here the rare Buy10_* decisions are the 10x-revenue events, so whether
    # they are genuinely predicted or quietly ignored is the interesting part.
    per_class: Dict[str, Dict[str, Any]] = {}
    for k in range(N_CLASSES):
        c = k + 1
        tpk, pk, tk = int(tp[k]), int(pred_cnt[k]), int(true_cnt[k])
        prec = tpk / pk if pk else 0.0
        rec = tpk / tk if tk else 0.0
        per_class[str(c)] = {
            "label": DECISION_LABELS[c],
            "support": tk,
            "predicted": pk,
            "true_positives": tpk,
            "precision": prec,
            "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
            "auprc": aps[k],
        }

    # ---- drift diagnostics (EXPERIMENTS.md R13) ----------------------------
    # Separates "the model is merely under-trained and sits near the class
    # frequencies" from "the model learned calibration-period structure that
    # does not transfer". Computed from arrays already held for AUPRC, so the
    # extra cost is negligible.
    true_dist = np.bincount(y_all - 1, minlength=N_CLASSES).astype(np.float64)
    true_dist /= true_dist.sum()
    pred_dist = p_all.astype(np.float64).mean(axis=0)
    # Oracle prior-matched NLL: rescale class probabilities so their average
    # equals this cell's empirical class frequencies (one Saerens-style step).
    # If this recovers most of a late epoch's lost NLL, the degradation was a
    # shift in class FREQUENCIES; if not, it is a shift in P(y | history).
    # "Oracle" because it uses this cell's own labels -- a diagnostic, never a
    # reportable number.
    ratio = true_dist / np.clip(pred_dist, 1e-12, None)
    p_adj = p_all.astype(np.float64) * ratio
    p_adj /= p_adj.sum(axis=1, keepdims=True)
    nll_pm = float(-np.log(np.clip(
        p_adj[np.arange(len(y_all)), y_all - 1], 1e-12, None)).mean())
    # The best CONSTANT predictor is the cell's own class frequencies; its NLL
    # is their entropy. A model near this number has learned almost nothing
    # beyond "how common is each decision".
    nz = true_dist[true_dist > 0]
    marginal_entropy = float(-(nz * np.log(nz)).sum())

    return {
        "nll": nll_sum / total,
        "nll_prior_matched": nll_pm,
        "marginal_entropy": marginal_entropy,
        "tv_pred_true": float(0.5 * np.abs(pred_dist - true_dist).sum()),
        "true_dist": [round(float(x), 6) for x in true_dist],
        "pred_dist": [round(float(x), 6) for x in pred_dist],
        "hit": correct / total,
        "f1_macro": macro_f1(tp, pred_cnt, true_cnt),
        "auprc_macro": float(np.mean([v for v in aps if not math.isnan(v)]))
                       if any(not math.isnan(v) for v in aps) else float("nan"),
        "rev_mae": rev_err / total,
        "n": float(total),
        "per_class": per_class,
    }


# ────────────────────────────── data ──────────────────────────────
def _load_uid_split(uids_dir: Path, raw: List[Dict[str, Any]]
                    ) -> Tuple[List[int], List[int], List[int]]:
    """
    Partition records by explicit uid lists written by scripts/export_gen5_split.py.

    Preferred over the derived split for anything reportable: the derived one
    depends on record ORDER in the JSON, so regenerating the data silently
    changes the test set. These files pin the held-out users so the gen-5
    LSTM/GRU baselines hold out exactly the same ones as the transformer.
    """
    parts: Dict[str, set] = {}
    for name in ("train", "val", "test"):
        f = uids_dir / f"uids_{name}.txt"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found. Generate the split first:\n"
                f"    python scripts/export_gen5_split.py --out {uids_dir}"
            )
        parts[name] = {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                       if ln.strip()}

    overlap = ((parts["train"] & parts["val"]) | (parts["train"] & parts["test"])
               | (parts["val"] & parts["test"]))
    if overlap:
        raise ValueError(
            f"uid split files overlap in {len(overlap)} uid(s) -- a user would appear "
            "in more than one split, which invalidates the held-out metrics."
        )

    by_uid: Dict[str, int] = {}
    for i, rec in enumerate(raw):
        by_uid.setdefault(TransformerDataset._uid(rec), i)

    out = []
    for name in ("train", "val", "test"):
        idxs = [by_uid[u] for u in parts[name] if u in by_uid]
        missing = len(parts[name]) - len(idxs)
        if missing:
            print(f"[split] WARNING: {missing} uid(s) listed for {name} are not in "
                  f"this data file (cohort changed?)")
        out.append(sorted(idxs))

    unassigned = len(raw) - sum(len(o) for o in out)
    if unassigned:
        print(f"[split] note: {unassigned} record(s) in the file are in no split file "
              "and will be ignored")
    return out[0], out[1], out[2]


def build_loaders(cfg: Dict[str, Any],
                  uids_dir: Optional[Path] = None
                  ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    path = config5.data_path(cfg)
    print(f"[data] {path}")
    raw = load_json_dataset(str(path))
    print(f"[data] {len(raw)} users in file")

    if uids_dir is not None:
        print(f"[split] using frozen uid split from {uids_dir}")
        tr_i, va_i, te_i = _load_uid_split(uids_dir, raw)
    else:
        print("[split] deriving split from seed (NOT frozen -- depends on record "
              "order in the JSON; use --uids-dir for reportable runs)")
        rng = random.Random(cfg["seed"])
        idx = list(range(len(raw)))
        rng.shuffle(idx)
        if cfg.get("max_users"):
            idx = idx[: int(cfg["max_users"])]
            print(f"[data] subsampled to {len(idx)} users (max_users)")

        n = len(idx)
        n_tr = int(cfg["train_frac"] * n)
        n_va = int(cfg["val_frac"] * n)
        tr_i, va_i, te_i = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    def subset(ii: List[int]) -> List[Dict[str, Any]]:
        return [raw[i] for i in ii]

    # ---- both: cross a user holdout with the temporal one -----------------
    if cfg.get("split_mode", "user") == "both":
        shift_b = bool(cfg.get("shift_obtained", True))
        common_b = dict(
            ai_rate=cfg["ai_rate"], lto_len=cfg["lto_len"],
            obtained_len=cfg["obtained_len"], prev_dec_len=cfg["prev_dec_len"],
            max_events=cfg["max_events"], base_seed=cfg["seed"],
            shift_obtained=shift_b, truncate="tail",
        )
        keep = raw if not cfg.get("max_users") else [raw[i] for i in idx]
        base = TransformerDataset(
            keep, augment_permute_obtained=cfg["augment_permute_obtained"],
            **common_b)
        if not base.has_holdout:
            raise SystemExit("split_mode=both needs the holdout flags in the data.")

        # Partition users. The held-out users are absent from training
        # entirely, so their embedding index is never learned -- which is the
        # point of the cold-start cells.
        # Design follows Lu & Kannan (JMR 2025), Table 4: customers are split
        # in two, periods are split in two, and the four cells are reported
        # separately. Validation comes from held-out CUSTOMERS inside the
        # calibration period -- not from a slice of time -- so the entire
        # holdout period stays untouched by model selection.
        flag = cfg.get("holdout_flag", "feature")
        # The customer partition uses split_seed, NOT the training seed, so
        # multi-seed runs share one partition and their differences reflect
        # training noise rather than which customers landed in which cell.
        rng_u = random.Random(cfg.get("split_seed", cfg["seed"]))
        order = list(range(len(base)))
        rng_u.shuffle(order)

        n_out = max(1, int(cfg.get("user_holdout_frac", 0.5) * len(order)))
        out_users, in_users = order[:n_out], order[n_out:]
        val_mode = cfg.get("val_mode", "customers")
        if val_mode == "late":
            # No validation customers are carved out: every in-sample customer
            # trains on the early calibration campaigns and is validated on the
            # late ones, mirroring the in-sample x holdout cell one step back.
            val_users, train_users = [], list(in_users)
        else:
            n_val = max(1, int(cfg.get("val_user_frac", 0.1) * len(in_users)))
            val_users, train_users = in_users[:n_val], in_users[n_val:]

        # Only train_users ever update an embedding row. Everyone else shares
        # index 0, which is set to the MEAN of the trained rows before each
        # evaluation -- Lu & Kannan's omega-bar, "the average head weight from
        # the training population". A learned average is a real prior; an
        # untrained row is noise.
        unknown = {base._uid_cache[i] for i in val_users + out_users}
        for u in unknown:
            base.uid_to_index[u] = 0
        trained_idx = sorted({base.uid_to_index[base._uid_cache[i]]
                              for i in train_users} - {0})

        cal = TemporalRoleView(base, "calibration", holdout_flag=flag)
        hol = TemporalRoleView(base, "holdout", holdout_flag=flag)
        boundary = 28 if flag == "feature" else 29

        if val_mode == "late":
            vf = int(cfg.get("val_from", 27))
            if not vf < boundary:
                raise SystemExit(f"val_from={vf} must be < holdout boundary {boundary}")
            early = TemporalRoleView(base, "calibration_early",
                                     holdout_flag=flag, val_from=vf)
            late = TemporalRoleView(base, "calibration_late",
                                    holdout_flag=flag, val_from=vf)
            train_ds = early.subset(train_users)
            val_ds = late.subset(train_users)
            print(f"[split] VALIDATION = LATE CALIBRATION: in-sample customers "
                  f"train on campaigns < {vf} and validate on {vf}..{boundary - 1}. "
                  "Sits immediately before the holdout, so early stopping can "
                  "see temporal drift (EXPERIMENTS.md R13).")
        else:
            train_ds = cal.subset(train_users)
            val_ds = cal.subset(val_users)
        # NOT cold starts: labels are masked, inputs are not, so an
        # out-of-sample customer's history is still visible to the model.
        tests = {
            "insample_users_holdout_period": hol.subset(train_users),
            "outsample_users_calib_period": cal.subset(out_users),
            "outsample_users_holdout_period": hol.subset(out_users),
        }
        print(f"[split] BOTH (Lu & Kannan design), holdout_flag={flag}: "
              f"calibration = campaigns < {boundary}, holdout = >= {boundary}")
        print(f"[split]   {len(train_users)} train / {len(val_users)} val / "
              f"{len(out_users)} out-of-sample customers")
        print(f"[split]   {len(unknown)} customers share the unknown-user "
              f"embedding (index 0 = mean of {len(trained_idx)} trained rows)")
        print(f"[split]   train {train_ds.scored_events():,} events | "
              f"val {val_ds.scored_events():,} events")
        for k, v in tests.items():
            print(f"[split]   test  {k:<31} {v.scored_events():,} events")

        def mk_b(ds, shuffle):
            return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                              collate_fn=collate_multistream,
                              num_workers=cfg.get("num_workers", 0),
                              pin_memory=torch.cuda.is_available())

        cfg["_trained_user_indices"] = trained_idx
        return (mk_b(train_ds, True), mk_b(val_ds, False),
                {k: mk_b(v, False) for k, v in tests.items()}, base.num_users)

    # ---- temporal split: every user in every role, split by campaign ------
    if cfg.get("split_mode", "user") == "temporal":
        shift_t = bool(cfg.get("shift_obtained", True))
        common_t = dict(
            ai_rate=cfg["ai_rate"], lto_len=cfg["lto_len"],
            obtained_len=cfg["obtained_len"], prev_dec_len=cfg["prev_dec_len"],
            max_events=cfg["max_events"], base_seed=cfg["seed"],
            shift_obtained=shift_t,
            # Keep each user's LAST max_events. Head truncation would delete
            # the late campaigns that define val and test.
            truncate="tail",
        )
        keep = raw if not cfg.get("max_users") else [raw[i] for i in idx]
        base = TransformerDataset(
            keep, augment_permute_obtained=cfg["augment_permute_obtained"],
            **common_t)
        if not base.has_holdout:
            raise SystemExit(
                "split_mode=temporal needs FeatureBasedHoldout and "
                "IndexBasedHoldout in the data file; this one has neither.")
        views = {r: TemporalRoleView(base, r) for r in ("train", "val", "test")}
        print(f"[split] TEMPORAL — all {len(base)} users appear in every role; "
              "split is by campaign")
        print("[split] train: campaigns <= 27 | val: campaign 28 | "
              "test: campaigns >= 29")
        for r, v in views.items():
            print(f"[split]   {r:<5} scores {v.scored_events():,} events")

        def mk_t(ds, shuffle):
            return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                              collate_fn=collate_multistream,
                              num_workers=cfg.get("num_workers", 0),
                              pin_memory=torch.cuda.is_available())

        return (mk_t(views["train"], True), mk_t(views["val"], False),
                mk_t(views["test"], False), base.num_users)

    shift = bool(cfg.get("shift_obtained", True))
    if not shift:
        print("[data] WARNING: shift_obtained=False -- the obtained stream "
              "carries o_t, which determines y_t==9 exactly. NotBuy metrics "
              "from this run are leakage, not prediction.")
    common = dict(
        ai_rate=cfg["ai_rate"], lto_len=cfg["lto_len"],
        obtained_len=cfg["obtained_len"], prev_dec_len=cfg["prev_dec_len"],
        max_events=cfg["max_events"], base_seed=cfg["seed"],
        shift_obtained=shift,
    )
    train_ds = TransformerDataset(
        subset(tr_i), augment_permute_obtained=cfg["augment_permute_obtained"], **common)
    # Reuse the train uid->index map so a user embedding means the same thing
    # in every split, and unseen users fall back to index 0.
    val_ds = TransformerDataset(subset(va_i), uid_to_index=train_ds.uid_to_index, **common)
    test_ds = TransformerDataset(subset(te_i), uid_to_index=train_ds.uid_to_index, **common)

    print(f"[data] train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)} users")
    if train_ds.has_ipt:
        print("[data] IPT field present (not consumed by the model yet)")

    def mk(ds, shuffle):
        return DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=shuffle,
            collate_fn=collate_multistream,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), train_ds.num_users


# ────────────────────────────── train ──────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pilot", choices=["pilot", "hpcc"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last.pt in the output dir if present. "
                         "Use this on HPCC so a job killed by the walltime "
                         "limit can be requeued and continue.")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="Stop cleanly after this many minutes and save "
                         "last.pt, so the run ends before PBS kills it. "
                         "Set it a little under the job's walltime.")
    # --- knobs for the regularisation sweep ---------------------------------
    ap.add_argument("--no-user-embedding", action="store_true",
                    help="Drop the per-user embedding. It is a lookup over "
                         "TRAINING users only; because splits are disjoint by "
                         "user, every val/test user falls back to index 0, so "
                         "at evaluation it is a constant carrying no "
                         "information while accounting for ~29%% of "
                         "parameters. Pure memorisation capacity.")
    ap.add_argument("--mix-heads", type=int, default=None,
                    help="H for Lu & Kannan's per-customer mixture over H "
                         "output projections. 0 = single shared projection. "
                         "Out-of-sample customers get the population mean "
                         "alpha-bar. More interpretable than a plain user "
                         "embedding: alpha_n is a soft membership over H "
                         "behavioural patterns.")
    ap.add_argument("--augment", action="store_true",
                    help="Permute the obtained-product slots within each event. "
                         "Encodes the prior that inventory is a set, not a "
                         "sequence.")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="Training seed: initialisation and data order. The "
                         "customer partition uses split_seed instead, so runs "
                         "differing only in --seed share one partition.")
    ap.add_argument("--val-mode", choices=["customers", "late"], default=None,
                    help="customers: held-out customers scored on the whole "
                         "calibration period (Lu & Kannan). late: in-sample "
                         "customers scored on the LAST calibration campaigns, "
                         "so early stopping can see temporal drift. R13 showed "
                         "'customers' cannot detect it.")
    ap.add_argument("--val-from", type=int, default=None,
                    help="First campaign of the late validation block "
                         "(val-mode=late). Default 27: one campaign, about the "
                         "size of the whole holdout block.")
    ap.add_argument("--arch", choices=["transformer", "gru", "lstm"], default=None,
                    help="Sequence encoder. gru/lstm share the transformer's "
                         "feature lookup and within-event pooling and differ "
                         "only in how events combine over time.")
    ap.add_argument("--track-holdout", action="store_true",
                    help="DIAGNOSTIC: score the holdout cells every epoch and "
                         "record them in history.json under a ho_ prefix. Never "
                         "used for selection or early stopping. Slows each "
                         "epoch by roughly the holdout evaluation cost.")
    ap.add_argument("--split-mode", choices=["user", "temporal", "both"], default=None,
                    help="user: hold out whole users (disjoint by uid). "
                         "temporal: hold out later campaigns for every user, "
                         "using the FeatureBasedHoldout / IndexBasedHoldout "
                         "flags the R generator writes. Temporal is what the "
                         "dataset was designed for and answers 'can we "
                         "predict this user's future?' rather than 'can we "
                         "predict a stranger?'")
    ap.add_argument("--no-shift-obtained", action="store_true",
                    help="Reproduce the pre-fix behaviour where the obtained "
                         "stream carries o_t. This LEAKS the label: an "
                         "all-zero block determines y_t==9 exactly. For "
                         "before/after comparison only.")
    ap.add_argument("--uids-dir", default=None,
                    help="Directory holding uids_train/val/test.txt from "
                         "scripts/export_gen5_split.py. Use this for any run "
                         "whose numbers you intend to report or compare "
                         "against another model -- without it the split is "
                         "re-derived from record order and can shift if the "
                         "data file is regenerated.")
    args = ap.parse_args()

    cfg = config5.get_config(args.profile)
    for k, v in (("num_epochs", args.epochs), ("max_events", args.max_events),
                 ("batch_size", args.batch_size), ("max_users", args.max_users),
                 ("data_file", args.data_file), ("seed", args.seed)):
        if v is not None:
            cfg[k] = v

    set_seed(cfg["seed"])
    device = pick_device()
    adtype = amp_dtype(device) if cfg.get("amp") else None
    print(f"[env] device={device} amp={adtype}")
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"total={human(torch.cuda.get_device_properties(0).total_memory)}")

    if args.no_shift_obtained:
        cfg["shift_obtained"] = False
    if args.split_mode is not None:
        cfg["split_mode"] = args.split_mode
    if args.no_user_embedding:
        cfg["use_user_embedding"] = False
    if args.mix_heads is not None:
        cfg["num_mix_heads"] = args.mix_heads
    if args.augment:
        cfg["augment_permute_obtained"] = True
    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    if args.patience is not None:
        cfg["patience"] = args.patience
    if args.val_mode is not None:
        cfg["val_mode"] = args.val_mode
    if args.val_from is not None:
        cfg["val_from"] = args.val_from
    if args.arch is not None:
        cfg["arch"] = args.arch
    cfg["track_holdout"] = bool(args.track_holdout)
    if cfg["track_holdout"]:
        print("[cfg] --track-holdout: holdout cells scored EVERY epoch as a "
              "diagnostic. These never drive selection or early stopping.")
    print(f"[cfg] split_mode={cfg.get('split_mode', 'user')} "
          f"user_embedding={cfg['use_user_embedding']} "
          f"augment={cfg['augment_permute_obtained']} "
          f"dropout={cfg['dropout']} patience={cfg['patience']} "
          f"mix_heads={cfg.get('num_mix_heads', 0)} "
          f"shift_obtained={cfg['shift_obtained']}")
    uids_dir = Path(args.uids_dir) if args.uids_dir else None
    cfg["uids_dir"] = str(uids_dir) if uids_dir else None
    train_dl, val_dl, test_dl, num_users = build_loaders(cfg, uids_dir=uids_dir)

    feat = load_feature_tensor(config5.feature_path())
    arch = cfg.get("arch", "transformer")
    if arch != "transformer":
        if cfg.get("num_mix_heads", 0):
            raise SystemExit("--mix-heads is transformer-only; drop it for gru/lstm")
        from model_recurrent_baseline import build_recurrent_baseline
        model = build_recurrent_baseline(
            cell=arch,
            vocab_size_src=cfg["vocab_size_src"],
            vocab_size_tgt=cfg["vocab_size_tgt"],
            d_model=cfg["d_model"],
            n_layers=cfg["N"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            feature_tensor=feat,
            num_users=num_users,
            use_user_embedding=cfg["use_user_embedding"],
        ).to(device)
    else:
        model = build_transformer(
            vocab_size_src=cfg["vocab_size_src"],
            vocab_size_tgt=cfg["vocab_size_tgt"],
            max_seq_len=cfg["max_events"],
            d_model=cfg["d_model"],
            n_layers=cfg["N"],
            n_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            feature_tensor=feat,
            ai_rate=cfg["ai_rate"],
            num_users=num_users,
            lto_len=cfg["lto_len"],
            obtained_len=cfg["obtained_len"],
            prev_dec_len=cfg["prev_dec_len"],
            use_user_embedding=cfg["use_user_embedding"],
            num_mix_heads=cfg.get("num_mix_heads", 0),
        ).to(device)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par:,} parameters | d_model={cfg['d_model']} N={cfg['N']} "
          f"heads={cfg['num_heads']} max_events={cfg['max_events']}")

    print("[loss] counting class frequencies ...")
    w9 = compute_class_weights(train_dl, cfg["tau"])
    print(f"[loss] class weights (tau={cfg['tau']}): {w9.numpy().round(3)}")
    full_w = torch.ones(cfg["vocab_size_tgt"])
    full_w[1:1 + N_CLASSES] = w9
    loss_fn = FocalLoss(cfg["gamma"], PAD_ID, full_w).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], eps=cfg["eps"])
    accum = max(1, int(cfg["grad_accum"]))
    steps_per_epoch = max(1, math.ceil(len(train_dl) / accum))
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(cfg["min_lr"] / cfg["lr"], 0.5 * (1 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=(adtype == torch.float16))

    out_dir = config5.output_dir(cfg)
    ckpt_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    hist_path = out_dir / "history.json"
    print(f"[out] {out_dir}")

    best_nll, best_epoch, patience = float("inf"), -1, 0
    history: List[Dict[str, Any]] = []
    start_epoch = 0

    # ---- resume (HPCC: a walltime kill should not lose the run) ----------
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        sched.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_nll = float(state.get("best_nll", float("inf")))
        best_epoch = int(state.get("best_epoch", -1))
        patience = int(state.get("patience", 0))
        history = state.get("history", [])
        print(f"[resume] continuing from epoch {start_epoch} "
              f"(best val_nll so far {best_nll:.4f} at epoch {best_epoch})")
    elif args.resume:
        print(f"[resume] no {last_path.name} found; starting fresh")

    t_start = time.time()
    budget_s = args.time_budget_min * 60 if args.time_budget_min else None
    stopped_early_for_time = False

    for ep in range(start_epoch, cfg["num_epochs"]):
        model.train()
        if hasattr(train_dl.dataset, "set_epoch"):
            train_dl.dataset.set_epoch(ep)
        t0 = time.time()
        running, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_dl):
            lto = batch["lto"].to(device, non_blocking=True)
            obt = batch["obtained"].to(device, non_blocking=True)
            prev = batch["prev_decision"].to(device, non_blocking=True)
            uid = batch["user_id"].to(device, non_blocking=True)
            tgt = batch["label"].to(device, non_blocking=True)
            if not (tgt != PAD_ID).any():
                continue

            ctx = (torch.autocast("cuda", dtype=adtype)
                   if adtype is not None else torch.autocast("cpu", enabled=False))
            with ctx:
                logits = model(lto, obt, prev, uid)
                loss = loss_fn(logits.float(), tgt) / accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += loss.detach().item() * accum
            nb += 1

            if (i + 1) % accum == 0 or (i + 1) == len(train_dl):
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            if i % 25 == 0:
                mem = (f" mem={human(torch.cuda.max_memory_allocated())}"
                       if device.type == "cuda" else "")
                print(f"  ep{ep} step {i}/{len(train_dl)} loss={running/max(1,nb):.4f}{mem}",
                      flush=True)

        tr_loss = running / max(1, nb)
        # Validation customers are out-of-sample by design, so they use the
        # population-mean embedding. Refresh it as the trained rows move.
        set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", []))
        v = evaluate(model, val_dl, device, adtype)
        dt = time.time() - t0
        print(f"[ep {ep:02d}] train_loss={tr_loss:.4f}  val_nll={v['nll']:.4f}  "
              f"hit={v['hit']:.4f}  f1={v['f1_macro']:.4f}  auprc={v['auprc_macro']:.4f}  "
              f"revMAE={v['rev_mae']:.3f}  ({dt:.0f}s)")
        # Keep history.json a readable learning curve: the per-class block goes
        # only into final.json, where it describes the selected model.
        rec = {"epoch": ep, "train_loss": tr_loss, "secs": dt,
               **{k: val for k, val in v.items() if k != "per_class"}}

        # DIAGNOSTIC ONLY (--track-holdout). Scores the holdout cells every
        # epoch so we can see whether holdout performance degrades as training
        # continues -- the question of whether validation, which lives in the
        # CALIBRATION period, can detect temporal overfitting at all.
        #
        # These numbers must NEVER drive checkpoint selection or early
        # stopping; that would be selecting on the test set. They are written
        # under a "ho_" prefix so they cannot be mistaken for validation
        # metrics, and selection below still reads v["nll"] only.
        if cfg.get("track_holdout") and isinstance(test_dl, dict):
            for cell_name, cell_dl in test_dl.items():
                # The out-of-sample x calibration cell is ~1.26M events -- about
                # three times the two holdout cells together -- and says nothing
                # about temporal drift, so it is skipped unless asked for.
                if "holdout_period" not in cell_name and not cfg.get("track_all_cells"):
                    continue
                hm = evaluate(model, cell_dl, device, adtype)
                for k2 in ("nll", "hit", "f1_macro", "auprc_macro", "rev_mae",
                           "nll_prior_matched", "marginal_entropy",
                           "tv_pred_true", "true_dist", "pred_dist"):
                    rec[f"ho_{cell_name}_{k2}"] = hm[k2]
            print(f"         [diagnostic] holdout nll="
                  f"{rec.get('ho_outsample_users_holdout_period_nll', float('nan')):.4f}"
                  f"  f1={rec.get('ho_outsample_users_holdout_period_f1_macro', float('nan')):.4f}",
                  flush=True)

        history.append(rec)
        hist_path.write_text(json.dumps(history, indent=2))

        improved = v["nll"] < best_nll
        if improved:
            best_nll, best_epoch, patience = v["nll"], ep, 0
            torch.save({
                "epoch": ep, "val_nll": best_nll,
                "model_state_dict": model.state_dict(),
                "cfg": cfg, "num_users": num_users,
                "class_weights_9": w9.tolist(),
            }, ckpt_path)
            print(f"          saved best -> {ckpt_path.name}")
        else:
            patience += 1

        # Always refresh last.pt: this is what --resume reads. It carries the
        # optimiser and scheduler state too, so a requeued job continues the
        # schedule instead of restarting warmup.
        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "best_nll": best_nll, "best_epoch": best_epoch, "patience": patience,
            "history": history, "cfg": cfg, "num_users": num_users,
            "class_weights_9": w9.tolist(),
        }, last_path)

        if not improved and patience >= cfg["patience"]:
            print("[early stop] val_nll stopped improving")
            break

        if budget_s is not None and (time.time() - t_start) > budget_s:
            print(f"[time budget] {args.time_budget_min:.0f} min reached after "
                  f"epoch {ep}; stopping cleanly. Resubmit with --resume to continue.")
            stopped_early_for_time = True
            break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[test] loaded best epoch {state['epoch']}")
    # split_mode="both" gives several test cells; the others give one loader.
    cells = test_dl if isinstance(test_dl, dict) else {"test": test_dl}
    if set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", [])):
        print(f"[eval] unknown-customer embedding set to the mean of "
              f"{len(cfg['_trained_user_indices']):,} trained rows "
              "(Lu & Kannan omega-bar)")
    results: Dict[str, Any] = {}
    for name, dl in cells.items():
        m = evaluate(model, dl, device, adtype)
        results[name] = m
        if "per_class" in m:
            print(f"\n** {name}: per-class breakdown **")
            print(format_per_class(m["per_class"]))
        print(f"\n** {name} ** nll={m['nll']:.4f} hit={m['hit']:.4f} "
              f"f1={m['f1_macro']:.4f} auprc={m['auprc_macro']:.4f} "
              f"revMAE={m['rev_mae']:.3f} n={int(m['n'])}")

    if len(results) > 1:
        print("\n** TEST CELLS **")
        print(f"  {'cell':<24}{'n':>10}{'nll':>9}{'hit':>8}{'F1':>8}{'AUPRC':>8}")
        print("  " + "-" * 65)
        for name, m in results.items():
            print(f"  {name:<24}{int(m['n']):>10,}{m['nll']:>9.4f}"
                  f"{m['hit']:>8.4f}{m['f1_macro']:>8.4f}{m['auprc_macro']:>8.4f}")

    t = results.get("test") or next(iter(results.values()))
    (out_dir / "final.json").write_text(json.dumps(
        {"best_val_nll": best_nll, "best_epoch": best_epoch,
         "test": t, "test_cells": results,
         "cfg": cfg, "params": n_par,
         "stopped_for_time_budget": stopped_early_for_time,
         "epochs_completed": len(history)}, indent=2))
    if stopped_early_for_time:
        print("[note] run ended on the time budget, not on convergence. "
              "Resubmit the same job with --resume to continue.")
    if device.type == "cuda":
        print(f"[env] peak GPU memory: {human(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
