#!/usr/bin/env python3
"""
train4_mixture_flash_aws.py

Training script for FlashAttention + Mixture-Head model.
Key differences from train4_flash_aws.py:
  - Each consumer gets a user_id index
  - Model outputs mixed probabilities from H heads weighted by user gate
  - Loss: NLL on mixed probabilities (mixture-aware)
  - Unseen users at test time use mean gate
"""
from __future__ import annotations

import json, math, os, socket, sys, time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config4 import get_config
from model4_mixture_flash import build_transformer

# ── Constants ──
FEATURE_COLS = [
    "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
    "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow",
    "WeaponTypeMagic", "WeaponTypePolearm",
    "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire",
    "EthnicityThunder", "EthnicityWind",
    "GenderFemale", "GenderMale",
    "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
    "type_figure", "MinimumAttack", "MaximumAttack",
    "MinSpecialEffect", "MaxSpecialEffect",
    "SpecialEffectEfficiency", "SpecialEffectExpertise",
    "SpecialEffectAttack", "SpecialEffectSuper",
    "SpecialEffectRatio", "SpecialEffectPhysical", "SpecialEffectLife", "LTO",
]
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
UNK_PROD_ID = 59
MAX_TOKEN_ID = 68

S3_BUCKET = "productgptbucket"
S3_PREFIX = "FullProductGPT/mixture_flash/FeatureBased"

FEAT_XLSX = "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx"
DATA_PATH_TRAIN = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
DATA_PATH_ALL   = "/home/ec2-user/data/clean_list_int_wide4_simple6.json"
FOLD_SPEC_URI   = "s3://productgptbucket/folds/productgptfolds.json"

s3 = boto3.client("s3")


def load_feature_tensor() -> torch.Tensor:
    df = pd.read_excel(FEAT_XLSX, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        tid = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= tid <= LAST_PROD_ID:
            arr[tid] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


def load_fold_spec(uri: str) -> dict:
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(uri) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
# Dataset with user IDs
# ═══════════════════════════════════════════════════════════
class MixtureDataset(Dataset):
    """
    Loads sequences and assigns each UID a persistent integer index.
    """
    def __init__(self, json_path: str, uid_to_index: Dict[str, int],
                 ai_rate: int = 15, seq_len_tgt: int = 1024):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with open(json_path) as f:
            rows = json.load(f)

        # Build tokenizer
        PAD_ID = 0
        vocab = {
            "[PAD]": 0,
            **{str(i): i for i in range(1, 10)},
            "[SOS]": 10, "[EOS]": 11, "[UNK]": 12,
            **{str(i): i for i in range(13, UNK_PROD_ID + 1)},
        }
        tok_src = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tok_src.pre_tokenizer = pre_tokenizers.Whitespace()
        tok_src.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")

        tok_tgt = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tok_tgt.pre_tokenizer = pre_tokenizers.Whitespace()
        vocab_tgt = {
            "[PAD]": 0, **{str(i): i for i in range(1, 10)},
            "[SOS]": 10, "[EOS]": 11, "[UNK]": 12,
        }
        tok_tgt.model = models.WordLevel(vocab=vocab_tgt, unk_token="[UNK]")

        seq_len_ai = seq_len_tgt * ai_rate

        self.x, self.y, self.user_ids, self.uids = [], [], [], []
        for row in rows:
            uid = str(row["uid"][0] if isinstance(row["uid"], list) else row["uid"])
            user_id = uid_to_index.get(uid, 0)  # 0 = UNK

            agg = row["AggregateInput"]
            src_txt = " ".join(map(str, agg)) if isinstance(agg, (list, tuple)) else str(agg)
            ai_ids = tok_src.encode(src_txt).ids[:seq_len_ai]
            if len(ai_ids) < seq_len_ai:
                ai_ids += [PAD_ID] * (seq_len_ai - len(ai_ids))

            dec = row["Decision"]
            tgt_txt = " ".join(map(str, dec)) if isinstance(dec, (list, tuple)) else str(dec)
            tgt_ids = tok_tgt.encode(tgt_txt).ids[:seq_len_tgt]
            if len(tgt_ids) < seq_len_tgt:
                tgt_ids += [PAD_ID] * (seq_len_tgt - len(tgt_ids))

            self.x.append(torch.tensor(ai_ids, dtype=torch.long))
            self.y.append(torch.tensor(tgt_ids, dtype=torch.long))
            self.user_ids.append(user_id)
            self.uids.append(uid)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.user_ids[idx]


def collate_fn(batch):
    xs, ys, uids = zip(*batch)
    return (torch.stack(xs), torch.stack(ys),
            torch.tensor(uids, dtype=torch.long))


# ═══════════════════════════════════════════════════════════
# Loss: NLL on mixture probabilities
# ═══════════════════════════════════════════════════════════
class MixtureFocalLoss(nn.Module):
    """
    Focal-like loss on the mixed probabilities.
    Since the model outputs probs (not logits), we use -log(p) directly.
    """
    def __init__(self, gamma: float = 0.0, label_smoothing: float = 0.0,
                 ignore_index: int = 0, num_classes: int = 10,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None
                             else torch.ones(num_classes))

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        probs:   (B, T, C) probabilities from mixture
        targets: (B, T)    class indices
        """
        B, T, C = probs.shape
        probs_flat = probs.reshape(-1, C)
        targets_flat = targets.reshape(-1)

        # Mask
        mask = targets_flat != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        probs_m = probs_flat[mask]
        targets_m = targets_flat[mask]

        # Label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / C
            one_hot = torch.zeros_like(probs_m)
            one_hot.scatter_(1, targets_m.unsqueeze(1), 1.0)
            one_hot = one_hot * (1 - self.label_smoothing) + smooth
            log_p = torch.log(probs_m.clamp(min=1e-7))
            loss = -(one_hot * log_p).sum(dim=-1)
        else:
            # Gather prob of true class
            p_true = probs_m.gather(1, targets_m.unsqueeze(1)).squeeze(1)
            log_p = torch.log(p_true.clamp(min=1e-7))

            if self.gamma > 0:
                focal = (1 - p_true) ** self.gamma
                loss = -focal * log_p
            else:
                loss = -log_p

        # Class weights
        w = self.class_weights[targets_m]
        loss = (loss * w).mean()
        return loss


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════
def train_model(cfg: dict,
                report_fn=None,
                stop_check_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    fold_id = cfg["fold_id"]
    spec = load_fold_spec(FOLD_SPEC_URI)
    uids_test = {u for u, f in spec["assignment"].items() if f == fold_id}
    uids_trainval = {u for u in spec["assignment"] if u not in uids_test}

    # Build UID→index mapping from trainval
    uid_to_index = {uid: i + 1 for i, uid in enumerate(sorted(uids_trainval))}
    num_users = len(uid_to_index) + 1  # +1 for UNK=0
    print(f"[INFO] num_users={num_users} (UNK=0 + {len(uid_to_index)} trainval)")

    # Dataset
    ai_rate = cfg["ai_rate"]
    seq_len_tgt = cfg["seq_len_tgt"]

    ds_train = MixtureDataset(DATA_PATH_TRAIN, uid_to_index, ai_rate, seq_len_tgt)

    n = len(ds_train)
    n_train = int(0.8 * n)
    n_val = n - n_train
    g = torch.Generator().manual_seed(33)
    train_set, val_set = random_split(ds_train, [n_train, n_val], generator=g)

    # Subsample training if data_frac < 1
    data_frac = cfg.get("data_frac", 1.0)
    if data_frac < 1.0:
        k = max(1, int(len(train_set) * data_frac))
        g2 = torch.Generator().manual_seed(cfg.get("subsample_seed", 33))
        train_set, _ = random_split(train_set, [k, len(train_set) - k], generator=g2)

    bs = cfg["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    # Model
    feat_tensor = load_feature_tensor()
    special_ids = [0, 10, 11, 12, 57, 58]

    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg.get("vocab_size_src", MAX_TOKEN_ID + 1),
        max_seq_len=seq_len_tgt * ai_rate,
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        feature_tensor=feat_tensor,
        special_token_ids=special_ids,
        num_users=num_users,
        num_mixture_heads=cfg.get("num_mixture_heads", 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params: {n_params:,}")

    # Class weights (inverse frequency)
    all_labels = []
    for _, y, _ in train_loader:
        all_labels.extend(y[y > 0].tolist())
    counts = Counter(all_labels)
    total = sum(counts.values())
    tau = cfg.get("tau", 0.5)
    cw = torch.ones(cfg["vocab_size_tgt"], device=device)
    for c in range(1, 10):
        freq = counts.get(c, 1) / total
        cw[c] = (1.0 / freq) ** tau
    cw = cw / cw[1:10].mean()

    loss_fn = MixtureFocalLoss(
        gamma=cfg.get("gamma", 0.0),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        ignore_index=0,
        num_classes=cfg["vocab_size_tgt"],
        class_weights=cw,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    warmup_steps = cfg.get("warmup_steps", 500)
    total_steps = len(train_loader) * cfg["num_epochs"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_nll = float("inf")
    patience = 0
    patience_limit = cfg.get("patience", 8)
    ckpt_path = Path(f"/tmp/MixtureFlash_{cfg.get('model_basename', 'model')}.pt")
    global_step = 0

    for epoch in range(cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        for xb, yb, uid_b in tqdm(train_loader, desc=f"Ep {epoch:02d}"):
            xb, yb, uid_b = xb.to(device), yb.to(device), uid_b.to(device)

            # Forward: model outputs probs (B, T_full, V)
            probs_full = model(xb, uid_b)

            # Extract decision positions
            pos = torch.arange(ai_rate - 1, xb.size(1), ai_rate, device=device)
            probs_dec = probs_full[:, pos, :]  # (B, n_decisions, V)

            n_dec = probs_dec.size(1)
            tgt = yb[:, :n_dec]

            loss = loss_fn(probs_dec, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        # ── Validation ──
        model.eval()
        val_nll_sum, val_tokens = 0.0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for xb, yb, uid_b in val_loader:
                xb, yb, uid_b = xb.to(device), yb.to(device), uid_b.to(device)

                probs_full = model(xb, uid_b)
                pos = torch.arange(ai_rate - 1, xb.size(1), ai_rate, device=device)
                probs_dec = probs_full[:, pos, :]

                n_dec = probs_dec.size(1)
                tgt = yb[:, :n_dec]

                # NLL
                mask = tgt != 0
                if mask.any():
                    p_true = probs_dec[mask].gather(
                        1, tgt[mask].unsqueeze(1)).squeeze(1)
                    nll = -torch.log(p_true.clamp(min=1e-7))
                    val_nll_sum += nll.sum().item()
                    val_tokens += mask.sum().item()

                    pred = probs_dec[mask][:, 1:].argmax(dim=-1) + 1
                    val_preds.extend(pred.cpu().tolist())
                    val_labels.extend(tgt[mask].cpu().tolist())

        val_nll = val_nll_sum / max(1, val_tokens)
        val_preds_np = np.array(val_preds)
        val_labels_np = np.array(val_labels)
        val_hit = float(np.mean(val_preds_np == val_labels_np)) if len(val_preds) > 0 else 0.0
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels_np, val_preds_np, average="macro", zero_division=0) if len(val_preds) > 0 else 0.0

        print(f"Epoch {epoch}  ValNLL={val_nll:.4f}  Hit={val_hit:.4f}  F1={val_f1:.4f}")

        metrics = {
            "epoch": epoch,
            "val_nll": val_nll,
            "val_hit": val_hit,
            "val_f1_macro": val_f1,
        }

        if report_fn:
            report_fn(metrics)

        # Checkpoint
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience = 0

            # Save mean gate from training users
            train_user_ids = sorted(set(uid_to_index.values()))
            model.set_mean_gate_from_train_users(train_user_ids)

            save_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_nll": val_nll,
                "val_hit": val_hit,
                "val_f1_macro": val_f1,
                "num_users": num_users,
                "num_mixture_heads": cfg.get("num_mixture_heads", 4),
                "uid_to_index": uid_to_index,
                "config": cfg,
            }
            torch.save(save_dict, ckpt_path)

            # Upload to S3
            s3_ckpt = f"{S3_PREFIX}/checkpoints/{ckpt_path.name}"
            s3.upload_file(str(ckpt_path), S3_BUCKET, s3_ckpt)
            print(f"[INFO] artefacts → s3://{S3_BUCKET}/{s3_ckpt}")
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping (by val_nll).")
                break

    # ── Test inference ──
    if cfg.get("do_infer", False):
        print("\n[INFO] Running test inference...")
        # Load best checkpoint
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        # Load full data for inference
        ds_all = MixtureDataset(DATA_PATH_ALL, uid_to_index, ai_rate, seq_len_tgt)
        all_loader = DataLoader(ds_all, batch_size=bs, shuffle=False, collate_fn=collate_fn)

        # For unseen users, use mean gate
        # The model already has mean_gate set from training

        test_uids_set = uids_test
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb, uid_b in all_loader:
                xb, yb, uid_b = xb.to(device), yb.to(device), uid_b.to(device)

                # For unseen users (uid_b == 0), temporarily enable mean gate
                seen_mask = uid_b > 0

                if seen_mask.all():
                    probs_full = model(xb, uid_b)
                elif (~seen_mask).all():
                    model.gate.use_mean_gate = True
                    probs_full = model(xb, uid_b)
                    model.gate.use_mean_gate = False
                else:
                    # Mixed batch: process separately
                    probs_full = torch.zeros(xb.size(0), xb.size(1),
                                            cfg["vocab_size_tgt"], device=device)
                    # Seen users
                    idx_seen = seen_mask.nonzero(as_tuple=True)[0]
                    if len(idx_seen) > 0:
                        probs_full[idx_seen] = model(xb[idx_seen], uid_b[idx_seen])

                    # Unseen users
                    idx_unseen = (~seen_mask).nonzero(as_tuple=True)[0]
                    if len(idx_unseen) > 0:
                        model.gate.use_mean_gate = True
                        probs_full[idx_unseen] = model(xb[idx_unseen], uid_b[idx_unseen])
                        model.gate.use_mean_gate = False

                pos = torch.arange(ai_rate - 1, xb.size(1), ai_rate, device=device)
                probs_dec = probs_full[:, pos, :]

        print(f"\n** BEST ** val_nll={best_val_nll:.4f}")

    return {
        "best_val_nll": best_val_nll,
        "fold_id": fold_id,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = get_config()
    cfg.update({
        "mode": "train",
        "fold_id": 0,
        "ai_rate": 15,
        "num_epochs": 200,
        "data_frac": 1.0,
        "do_infer": True,
        "subsample_seed": 33,

        # Best flash HP from Phase A
        "d_model": 128,
        "num_heads": 8,
        "N": 6,
        "d_ff": 384,
        "dropout": 0.221486,
        "lr": 0.00089497,
        "tau": 0.303938,
        "gamma": 0.0,
        "warmup_steps": 500,
        "batch_size": 4,
        "label_smoothing": 0.0930536,
        "weight_decay": 0.01,
        "weight": 1,

        # Mixture-specific
        "num_mixture_heads": 4,
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]
    cfg["model_basename"] = "mixture_flash_fold0"

    train_model(cfg)