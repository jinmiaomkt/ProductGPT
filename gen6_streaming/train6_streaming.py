"""
Generation-6 trainer: the streaming hybrid on full customer histories.

WHAT IS THE SAME AS GEN 5 (on purpose -- one factor at a time)
-------------------------------------------------------------
The data file, the Lu & Kannan customer x period split, the late-calibration
validation (campaign 27), the class weights, the loss, the optimiser and its
parameter groups, the schedule, early stopping, the metrics and the files a
run writes (best.pt, last.pt, history.json, final.json with the same keys).
All of it is imported from gen5_multistream/train5_multistream.py.

The defaults reproduce the b36 triplet incumbent's configuration (INCUMBENT
below), except model_rev, which is 2: gen 6 refuses the rev-1 stock timing.

WHAT CHANGES
------------
1. No truncation. Every customer is read from their first occasion
   (max_events=None), so the 9% longest customers keep their early history.
2. The encoder: sliding-window attention plus carried GRU state
   (gen6_streaming/model6_streaming.py), trained `chunk` occasions at a time.
3. Batching by length. Customers are bucketed by how many rows they need, so a
   batch is not padded to one long customer's length. A customer's rows after
   their last scored occasion are dropped: the model is causal, so those rows
   cannot change any scored prediction.

USAGE
-----
    python gen6_streaming/train6_streaming.py --profile pilot --max-users 200 --epochs 1
    python gen6_streaming/train6_streaming.py --profile hpcc --seed 1

Reads the data location from PRODUCTGPT_DATA (see paths.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "shared"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config5  # noqa: E402
import dataset_multistream  # noqa: E402
import model_multistream_state_space as mm  # noqa: E402
import train5_multistream as t5  # noqa: E402
from dataset_multistream import collate_multistream  # noqa: E402
from censor6 import MODES as CHOP_MODES, STATES as CHOP_STATES, ChopView, collate_prefix  # noqa: E402
from model6_streaming import build_from_cfg  # noqa: E402
from shared.features import load_feature_tensor  # noqa: E402

PAD_ID = 0
N_CLASSES = 9
# R45 batch fields the model consumes (see censor6.collate_prefix)
CHOP_KEYS = ("prefix_obtained", "prefix_prev", "censored")

# gen 5's evaluate() builds the model's extra kwargs with inventory_kwargs(); wrap
# it so evaluation passes the R45 fields too. It looks the name up at call time,
# so this reaches evaluate() unchanged.
_inventory_kwargs5 = t5.inventory_kwargs


def _inventory_kwargs6(model, batch, device):
    kw = _inventory_kwargs5(model, batch, device)
    if getattr(model, "gen", 5) == 6:
        for k in CHOP_KEYS:
            if k in batch:
                kw[k] = batch[k].to(device, non_blocking=True)
    return kw


t5.inventory_kwargs = _inventory_kwargs6

# The b36 triplet incumbent (runs/gen5_hpcc_S1024_b4_b36_triplet_s*), read from
# its final.json on 19 Oct 2026. model_rev is the one deliberate difference.
INCUMBENT: Dict[str, Any] = {
    "vocab_level": 7,
    "batch_size": 4, "grad_accum": 2,
    "d_model": 96, "N": 6, "num_heads": 8, "d_ff": 288,
    "dropout": 0.1, "lr": 2.19e-4, "weight_decay": 0.05, "warmup_frac": 0.1,
    "num_epochs": 40, "patience": 5,
    "attn_time_bias": "ordinal", "attn_recency_bias": True,
    "sat_layers": 2, "kernel": "attr", "decay": "exp", "decay_init": 2.0,
    "decay_freeze": True, "tier": True, "camp_fe": True,
    "pity_stream": False, "use_user_embedding": False, "num_mix_heads": 0,
    "val_mode": "late", "val_from": 27, "holdout_flag": "feature",
    "split_mode": "both", "tau": 0.5, "gamma": 0.0,
}


# ─────────────────────────────── batching ───────────────────────────────
def scored_lengths(ds) -> List[int]:
    """Rows each customer needs: up to and including their last scored
    occasion (0 = nothing scored in this view, so the customer is skipped)."""
    out = []
    for i in range(len(ds)):
        lab = ds[i]["label"]
        hit = torch.nonzero((lab >= 1) & (lab <= N_CLASSES)).view(-1)
        out.append(int(hit[-1]) + 1 if hit.numel() else 0)
    return out


class LengthBucketSampler(Sampler[List[int]]):
    """Batches of customers with similar lengths.

    Shuffled: random pools of `pool` batches, sorted by length inside each
    pool, cut into batches, batch order shuffled -- randomness in which
    customers meet, little padding. Not shuffled: one sort, for evaluation.
    """

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool,
                 seed: int = 0, pool: int = 50):
        self.idx = [i for i, n in enumerate(lengths) if n > 0]
        self.len = {i: lengths[i] for i in self.idx}
        self.bs, self.shuffle, self.seed, self.pool = int(batch_size), bool(shuffle), int(seed), int(pool)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        if not self.shuffle:
            order = sorted(self.idx, key=lambda i: self.len[i])
            return [order[k:k + self.bs] for k in range(0, len(order), self.bs)]
        rng = np.random.default_rng(self.seed + 7919 * self.epoch)
        perm = [self.idx[j] for j in rng.permutation(len(self.idx))]
        span = self.bs * self.pool
        out = []
        for k in range(0, len(perm), span):
            grp = sorted(perm[k:k + span], key=lambda i: self.len[i])
            out.extend(grp[j:j + self.bs] for j in range(0, len(grp), self.bs))
        return [out[j] for j in rng.permutation(len(out))]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return math.ceil(len(self.idx) / self.bs)


def collate_scored(items):
    """collate_multistream, then drop rows after the batch's last scored row.
    The sampler never yields a customer with nothing scored."""
    b = collate_multistream(items)
    cut = max(int(((it["label"] >= 1) & (it["label"] <= N_CLASSES)).nonzero().max()) + 1
              for it in items)
    for k, v in list(b.items()):
        if torch.is_tensor(v) and v.dim() >= 2 and k not in ("inv_init_count", "inv_init_last"):
            b[k] = v[:, :cut]
    return collate_prefix(items, b)          # R45 chop fields, if any (never cut)


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    lengths = scored_lengths(ds)
    sampler = LengthBucketSampler(lengths, batch_size, shuffle, seed)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_scored,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
    rows = sum(lengths)
    return dl, rows, sum(1 for n in lengths if n > 0)


# ─────────────────────────────── loss ───────────────────────────────
def weighted_ce_sum(logits: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor,
                    gamma: float = 0.0) -> torch.Tensor:
    """Sum of gen 5's per-occasion loss over the scored occasions.

    gen 5's FocalLoss is mean_{valid}(w_y * ce) (focal-modulated if gamma>0);
    dividing this sum by the batch's valid count reproduces it exactly while
    letting each chunk contribute its own part.
    """
    V = logits.size(-1)
    lg = logits.reshape(-1, V)
    y = tgt.reshape(-1)
    ce = F.cross_entropy(lg, y, weight=w.to(lg.dtype), ignore_index=PAD_ID, reduction="none")
    ce = ce[y != PAD_ID]
    if ce.numel() == 0:
        return lg.sum() * 0.0
    if gamma > 0:
        ce = ((1 - torch.exp(-ce)) ** gamma) * ce
    return ce.sum()


# ─────────────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pilot", choices=["pilot", "hpcc"])
    ap.add_argument("--window", type=int, default=384,
                    help="occasions each attention query sees, itself included; 0 = unlimited")
    ap.add_argument("--chunk", type=int, default=512,
                    help="occasions per training chunk (gradient truncation length)")
    ap.add_argument("--eval-chunk", type=int, default=1024,
                    help="chunk length for evaluation (does not change the function)")
    ap.add_argument("--eval-batch-size", type=int, default=16,
                    help="customers per evaluation batch (no gradient, so larger is fine)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--vocab-level", type=int, default=None, choices=[6, 7])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup-frac", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--sat-layers", type=int, default=None)
    ap.add_argument("--kernel", choices=["none", "attr", "learned"], default=None)
    ap.add_argument("--decay", choices=["none", "exp"], default=None)
    ap.add_argument("--decay-init", type=float, default=None)
    ap.add_argument("--decay-freeze", type=int, default=None)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--camp-fe", type=int, default=None)
    ap.add_argument("--alibi", type=int, default=None, help="1 = ordinal ALiBi (default), 0 = none")
    ap.add_argument("--pity-stream", type=int, default=None)
    ap.add_argument("--val-from", type=int, default=None)
    ap.add_argument("--uids-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--track-holdout", action="store_true",
                    help="DIAGNOSTIC: score the holdout cells every epoch (never used for selection)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=None)
    # R45: controlled left-censoring (gen6_streaming/censor6.py)
    ap.add_argument("--chop", choices=list(CHOP_MODES), default="none",
                    help="chop the start of each history: calendar = first DEPTH active "
                         "campaigns, count = first DEPTH occasions")
    ap.add_argument("--chop-depth", type=int, default=0)
    ap.add_argument("--chop-state", choices=list(CHOP_STATES), default="zero",
                    help="zero = rows gone and state restarts; true = rows gone from context "
                         "but fed to the computed state; labels = only their labels masked")
    ap.add_argument("--chop-where", choices=["train", "eval", "both"], default="both",
                    help="train = training customers (training and validation inputs); "
                         "eval = the test cells only; both = everywhere")
    ap.add_argument("--censor-flag", type=int, default=0,
                    help="1 = learned censoring vector and starting GRU state for chopped customers")
    ap.add_argument("--allow-laptop", action="store_true",
                    help="override the guard that refuses to train on Windows (see below)")
    args = ap.parse_args()

    # GUARD. On 19 Oct 2026 three pilot runs of this trainer on the laptop's
    # NVIDIA GPU (RTX PRO 500 Blackwell) each crashed the machine within about a
    # minute: a fatal PCIe hardware error on the GPU, then bugcheck 0x20001
    # (HYPERVISOR_ERROR). Training runs on HPCC only. Override only after IT has
    # cleared the GPU.
    if sys.platform == "win32" and not args.allow_laptop:
        raise SystemExit(
            "[guard] gen-6 training runs on HPCC only (scripts/gen6_train_hpcc.pbs). "
            "On 19 Oct 2026 every attempt to train on this laptop's GPU crashed the "
            "machine (fatal PCIe error, bugcheck 0x20001). Pass --allow-laptop only "
            "after IT has cleared the GPU.")

    cfg = config5.get_config(args.profile)
    cfg.update(INCUMBENT)
    if args.vocab_level is not None:
        cfg["vocab_level"] = args.vocab_level
    config5.apply_vocab_level(cfg, cfg["vocab_level"])
    for k, v in (("num_epochs", args.epochs), ("batch_size", args.batch_size),
                 ("grad_accum", args.grad_accum), ("max_users", args.max_users),
                 ("data_file", args.data_file), ("seed", args.seed),
                 ("d_model", args.d_model), ("N", args.n_layers), ("num_heads", args.n_heads),
                 ("d_ff", args.d_ff), ("dropout", args.dropout), ("lr", args.lr),
                 ("weight_decay", args.weight_decay), ("warmup_frac", args.warmup_frac),
                 ("patience", args.patience), ("sat_layers", args.sat_layers),
                 ("kernel", args.kernel), ("decay", args.decay), ("decay_init", args.decay_init),
                 ("val_from", args.val_from), ("num_workers", args.num_workers)):
        if v is not None:
            cfg[k] = v
    for k, v in (("decay_freeze", args.decay_freeze), ("tier", args.tier),
                 ("camp_fe", args.camp_fe), ("pity_stream", args.pity_stream)):
        if v is not None:
            cfg[k] = bool(v)
    if args.alibi is not None:
        cfg["attn_recency_bias"] = bool(args.alibi)
        cfg["attn_time_bias"] = "ordinal" if args.alibi else "none"
    cfg.update({"gen": 6, "run_name": "gen6_streaming", "max_events": None,
                "window": int(args.window), "chunk": int(args.chunk),
                "eval_chunk": int(args.eval_chunk), "model_rev": mm.MODEL_REV,
                "encoder": "gru_attn", "fuse": "stack", "inventory": "slots",
                "use_offer_inventory_attn": True, "use_obtained": True,
                "track_holdout": bool(args.track_holdout),
                "chop": args.chop, "chop_depth": int(args.chop_depth),
                "chop_state": args.chop_state, "chop_where": args.chop_where,
                "censor_flag": bool(args.censor_flag)})
    if cfg["chop"] != "none" and cfg["chop_depth"] <= 0:
        raise SystemExit("--chop needs --chop-depth > 0")
    if cfg.get("max_users") and cfg.get("split_mode") == "both":
        print("[data] note: max_users subsamples customers BEFORE the split (pilot only)")

    dataset_multistream.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])
    t5.set_seed(cfg["seed"])
    device = t5.pick_device()
    adtype = t5.amp_dtype(device) if cfg.get("amp") else None
    print(f"[env] device={device} amp={adtype} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"total={t5.human(torch.cuda.get_device_properties(0).total_memory)}")
    shown = {k: cfg[k] for k in sorted(cfg) if not k.startswith("_")}
    print("[cfg] " + json.dumps(shown, default=str, sort_keys=True))

    uids_dir = Path(args.uids_dir) if args.uids_dir else None
    cfg["uids_dir"] = str(uids_dir) if uids_dir else None
    tr0, va0, te0, num_users = t5.build_loaders(cfg, uids_dir=uids_dir)
    nw = int(cfg.get("num_workers", 0))

    def chopped(ds, side: str):
        """R45: training customers' inputs (training and validation) are side
        'train'; the test cells are side 'eval'."""
        if cfg["chop"] == "none" or cfg["chop_where"] not in (side, "both"):
            return ds
        return ChopView(ds, cfg["chop"], cfg["chop_depth"], cfg["chop_state"], cfg["val_from"])

    if cfg["chop"] != "none":
        print(f"[chop] mode={cfg['chop']} depth={cfg['chop_depth']} state={cfg['chop_state']} "
              f"where={cfg['chop_where']} censor_flag={cfg['censor_flag']}")
    train_dl, tr_rows, tr_n = make_loader(chopped(tr0.dataset, "train"), cfg["batch_size"], True,
                                          cfg["seed"], nw)
    ebs = int(args.eval_batch_size)
    cfg["eval_batch_size"] = ebs
    val_dl, va_rows, va_n = make_loader(chopped(va0.dataset, "train"), ebs, False, 0, nw)
    tests = {}
    for name, dl in te0.items():
        tests[name], rows, n = make_loader(chopped(dl.dataset, "eval"), ebs, False, 0, nw)
        print(f"[data] test {name:<31} {n:,} customers, {rows:,} rows to process")
    print(f"[data] train {tr_n:,} customers, {tr_rows:,} rows | val {va_n:,} customers, "
          f"{va_rows:,} rows (no truncation; rows after a customer's last scored "
          f"occasion are dropped)")

    feat = load_feature_tensor(config5.feature_file_path(cfg),
                               id_column=cfg.get("feature_id_column"),
                               first_prod_id=cfg["first_prod_id"],
                               last_prod_id=cfg["last_prod_id"],
                               max_token_id=cfg["vocab_size_src"] - 1)
    model = build_from_cfg(cfg, feat, num_users).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] gen6 streaming hybrid: {n_par:,} parameters | window={model.window or 'all'} "
          f"chunk={model.chunk} d_model={cfg['d_model']} N={cfg['N']} heads={cfg['num_heads']}")

    print("[loss] counting class frequencies ...")
    w9 = t5.compute_class_weights(train_dl, cfg["tau"])
    print(f"[loss] class weights (tau={cfg['tau']}): {w9.numpy().round(3)}")
    full_w = torch.ones(cfg["vocab_size_tgt"])
    full_w[1:1 + N_CLASSES] = w9
    full_w = full_w.to(device)

    STRUCT = ("half_life_raw", "tier_raw", "attr_w", "attr_self",
              "camp_bias", "k_emb", "k_q", "k_k")
    struct, rest = [], []
    for n_, p_ in model.named_parameters():
        (struct if any(k in n_ for k in STRUCT) else rest).append(p_)
    groups = [{"params": rest, "weight_decay": cfg["weight_decay"]}]
    if struct:
        groups.append({"params": struct, "weight_decay": 0.0,
                       "lr": cfg["lr"] * float(cfg.get("struct_lr_mult", 10.0))})
        print(f"[cfg] {len(struct)} structural parameters: no weight decay, "
              f"lr x{cfg.get('struct_lr_mult', 10.0)}")
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=cfg["eps"])
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
    ckpt_path, last_path, hist_path = out_dir / "best.pt", out_dir / "last.pt", out_dir / "history.json"
    print(f"[out] {out_dir}")

    best_nll, best_epoch, patience = float("inf"), -1, 0
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        for k in ("window", "chunk", "model_rev"):
            if state["cfg"].get(k) != cfg[k]:
                raise SystemExit(f"[resume] {last_path} has {k}={state['cfg'].get(k)}, this run "
                                 f"{cfg[k]}. Start a new TAG instead.")
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
        print(f"[resume] continuing from epoch {start_epoch} (best val_nll {best_nll:.4f} "
              f"at epoch {best_epoch})")
    elif args.resume:
        print(f"[resume] no {last_path.name} found; starting fresh")

    def autocast():
        return (torch.autocast("cuda", dtype=adtype) if adtype is not None
                else torch.autocast("cpu", enabled=False))

    def evaluate(dl) -> Dict[str, Any]:
        model.chunk = int(cfg["eval_chunk"])
        try:
            return t5.evaluate(model, dl, device, adtype)
        finally:
            model.chunk = int(cfg["chunk"])

    t_start = time.time()
    budget_s = args.time_budget_min * 60 if args.time_budget_min else None
    stopped_early_for_time = False

    for ep in range(start_epoch, cfg["num_epochs"]):
        model.train()
        train_dl.batch_sampler.set_epoch(ep)
        t0 = time.time()
        running, nb, n_chunks = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, batch in enumerate(train_dl):
            lto = batch["lto"].to(device, non_blocking=True)
            obt = batch["obtained"].to(device, non_blocking=True)
            prev = batch["prev_decision"].to(device, non_blocking=True)
            uid = batch["user_id"].to(device, non_blocking=True)
            tgt = batch["label"].to(device, non_blocking=True)
            camp = batch["campaign"].to(device, non_blocking=True) if "campaign" in batch else None
            extra = {k: batch[k].to(device, non_blocking=True) for k in CHOP_KEYS if k in batch}
            n_valid = int((tgt != PAD_ID).sum())
            if n_valid == 0:
                continue
            got = [0.0]

            def on_chunk(c0: int, c1: int, logits: torch.Tensor) -> None:
                loss = weighted_ce_sum(logits.float(), tgt[:, c0:c1], full_w,
                                       cfg["gamma"]) / n_valid / accum
                with torch.autocast(device.type, enabled=False):
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                got[0] += float(loss.detach()) * accum

            with autocast():
                model.stream_forward(lto, obt, prev, uid, camp, chunk=cfg["chunk"],
                                     on_chunk=on_chunk, **extra)
            n_chunks += math.ceil(lto.size(1) / cfg["chunk"])
            running += got[0]
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

            if i % 50 == 0:
                mem = (f" mem={t5.human(torch.cuda.max_memory_allocated())}"
                       if device.type == "cuda" else "")
                print(f"  ep{ep} step {i}/{len(train_dl)} rows={lto.size(1)} "
                      f"loss={running / max(1, nb):.4f}{mem}", flush=True)

        tr_loss = running / max(1, nb)
        t5.set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", []))
        v = evaluate(val_dl)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        print(f"[ep {ep:02d}] train_loss={tr_loss:.4f}  val_nll={v['nll']:.4f}  "
              f"hit={v['hit']:.4f}  f1={v['f1_macro']:.4f}  auprc={v['auprc_macro']:.4f}  "
              f"revMAE={v['rev_mae']:.3f}  chunks={n_chunks}  peak={t5.human(peak)}  ({dt:.0f}s)",
              flush=True)
        rec = {"epoch": ep, "train_loss": tr_loss, "secs": dt, "chunks": n_chunks,
               "peak_mem": peak, **{k: val for k, val in v.items() if k != "per_class"}}
        if cfg.get("track_holdout"):
            for cell_name, cell_dl in tests.items():
                if "holdout_period" not in cell_name:
                    continue
                hm = evaluate(cell_dl)
                for k2 in ("nll", "hit", "f1_macro", "auprc_macro", "rev_mae"):
                    rec[f"ho_{cell_name}_{k2}"] = hm[k2]
        history.append(rec)
        hist_path.write_text(json.dumps(history, indent=2))

        improved = v["nll"] < best_nll
        if improved:
            best_nll, best_epoch, patience = v["nll"], ep, 0
            torch.save({"epoch": ep, "val_nll": best_nll, "model_state_dict": model.state_dict(),
                        "cfg": cfg, "num_users": num_users, "class_weights_9": w9.tolist()},
                       ckpt_path)
            print(f"          saved best -> {ckpt_path.name}")
        else:
            patience += 1
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_nll": best_nll, "best_epoch": best_epoch, "patience": patience,
                    "history": history, "cfg": cfg, "num_users": num_users,
                    "class_weights_9": w9.tolist()}, last_path)
        if not improved and patience >= cfg["patience"]:
            print("[early stop] val_nll stopped improving")
            break
        if budget_s is not None and (time.time() - t_start) > budget_s:
            print(f"[time budget] {args.time_budget_min:.0f} min reached after epoch {ep}; "
                  "stopping cleanly. Resubmit with --resume to continue.")
            stopped_early_for_time = True
            break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[test] loaded best epoch {state['epoch']}")
    t5.set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", []))
    results: Dict[str, Any] = {}
    for name, dl in tests.items():
        m = evaluate(dl)
        results[name] = m
        if "per_class" in m:
            print(f"\n** {name}: per-class breakdown **")
            print(t5.format_per_class(m["per_class"]))
        print(f"\n** {name} ** nll={m['nll']:.4f} hit={m['hit']:.4f} f1={m['f1_macro']:.4f} "
              f"auprc={m['auprc_macro']:.4f} revMAE={m['rev_mae']:.3f} n={int(m['n'])}")
    print("\n** TEST CELLS **")
    print(f"  {'cell':<32}{'n':>10}{'nll':>9}{'hit':>8}{'F1':>8}{'AUPRC':>8}")
    for name, m in results.items():
        print(f"  {name:<32}{int(m['n']):>10,}{m['nll']:>9.4f}{m['hit']:>8.4f}"
              f"{m['f1_macro']:>8.4f}{m['auprc_macro']:>8.4f}")
    t = results.get("test") or next(iter(results.values()))
    (out_dir / "final.json").write_text(json.dumps(
        {"best_val_nll": best_nll, "best_epoch": best_epoch, "test": t, "test_cells": results,
         "cfg": cfg, "params": n_par, "stopped_for_time_budget": stopped_early_for_time,
         "epochs_completed": len(history)}, indent=2, default=str))
    if device.type == "cuda":
        print(f"[env] peak GPU memory: {t5.human(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
