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


def macro_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision per class, averaged over classes with support."""
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return float("nan")
    vals = []
    for k in range(N_CLASSES):
        pos = (y_true == k + 1).astype(np.int8)
        if pos.sum() == 0:
            continue
        vals.append(average_precision_score(pos, scores[:, k]))
    return float(np.mean(vals)) if vals else float("nan")


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
    return {
        "nll": nll_sum / total,
        "hit": correct / total,
        "f1_macro": macro_f1(tp, pred_cnt, true_cnt),
        "auprc_macro": macro_auprc(y_all, p_all),
        "rev_mae": rev_err / total,
        "n": float(total),
    }


# ────────────────────────────── data ──────────────────────────────
def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    path = config5.data_path(cfg)
    print(f"[data] {path}")
    raw = load_json_dataset(str(path))
    print(f"[data] {len(raw)} users in file")

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

    common = dict(
        ai_rate=cfg["ai_rate"], lto_len=cfg["lto_len"],
        obtained_len=cfg["obtained_len"], prev_dec_len=cfg["prev_dec_len"],
        max_events=cfg["max_events"], base_seed=cfg["seed"],
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
    args = ap.parse_args()

    cfg = config5.get_config(args.profile)
    for k, v in (("num_epochs", args.epochs), ("max_events", args.max_events),
                 ("batch_size", args.batch_size), ("max_users", args.max_users),
                 ("data_file", args.data_file)):
        if v is not None:
            cfg[k] = v

    set_seed(cfg["seed"])
    device = pick_device()
    adtype = amp_dtype(device) if cfg.get("amp") else None
    print(f"[env] device={device} amp={adtype}")
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"total={human(torch.cuda.get_device_properties(0).total_memory)}")

    train_dl, val_dl, test_dl, num_users = build_loaders(cfg)

    feat = load_feature_tensor(config5.feature_path())
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
        v = evaluate(model, val_dl, device, adtype)
        dt = time.time() - t0
        print(f"[ep {ep:02d}] train_loss={tr_loss:.4f}  val_nll={v['nll']:.4f}  "
              f"hit={v['hit']:.4f}  f1={v['f1_macro']:.4f}  auprc={v['auprc_macro']:.4f}  "
              f"revMAE={v['rev_mae']:.3f}  ({dt:.0f}s)")
        history.append({"epoch": ep, "train_loss": tr_loss, **v, "secs": dt})
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
    t = evaluate(model, test_dl, device, adtype)
    print(f"\n** TEST ** nll={t['nll']:.4f} hit={t['hit']:.4f} f1={t['f1_macro']:.4f} "
          f"auprc={t['auprc_macro']:.4f} revMAE={t['rev_mae']:.3f} n={int(t['n'])}")

    (out_dir / "final.json").write_text(json.dumps(
        {"best_val_nll": best_nll, "best_epoch": best_epoch, "test": t,
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
