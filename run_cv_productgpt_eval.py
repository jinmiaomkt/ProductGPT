#!/usr/bin/env python3
"""
10-fold CV training + evaluation that:
  • trains a fresh model per fold (8 train folds, 1 val fold, 1 test fold)
  • prints per-fold metrics to stdout (pretty tables)
  • writes tidy CSVs directly to S3 (no local .pt/.jsonl artefacts)

Drop-in hooks to integrate with your existing code:
  - build_model(hparams) -> torch.nn.Module
  - get_dataloaders_for_fold(fold_idx, num_folds, batch_size) -> dict with 'train', 'val', 'test'
  - infer_step(model, batch) -> dict with keys: 'y_true', 'y_pred', 'task', 'group'

Assumptions:
  - Multiple binary tasks (e.g., BuyFigure / BuyOne / ...), possibly stratified by 'group' (Calibration/HoldoutA/HoldoutB)
  - ROC-AUC is the main metric; feel free to extend to F1/AUPRC as needed

S3:
  - Uses boto3 and uploads from memory buffers (no local temp files)
  - Provide --s3-bucket and --s3-prefix; script creates two CSVs per run:
      1) metrics_per_fold.csv
      2) rocauc_pivot_table.csv (Task x Group with [val, test] columns)

"""
from __future__ import annotations
import argparse
import io
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# ----------------------------- User hooks ----------------------------------
# Replace these three hooks with your project-specific code.

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

# --- project-local imports already available in your tree
# (paths/names must match your repo)
from dataset4_productgpt import TransformerDataset, load_json_dataset
from model4_decoderonly_feature_performer import build_transformer

from config4 import get_config
CFG = get_config()
CFG["seq_len_ai"] = CFG["ai_rate"] * CFG["seq_len_tgt"]  # like in your trainer
CFG["cv_seed"] = 33  # optional, reproducible folds

# If these are defined elsewhere in your file, reuse them; otherwise import.
# From your train4_* file:
#   - PAD_ID, SPECIAL_IDS, FIRST_PROD_ID, LAST_PROD_ID, UNK_PROD_ID
#   - load_feature_tensor (reads SelectedFigureWeaponEmbeddingIndex.xlsx)
#   - FEATURE_COLS (not needed here but fine to import)

# ===== Binary task definition (same as predict_productgpt_and_eval.py) =====
BIN_TASKS = {
    "BuyNone":    [9],
    "BuyOne":     [1, 3, 5, 7],
    "BuyTen":     [2, 4, 6, 8],
    "BuyRegular": [1, 2],
    "BuyFigure":  [3, 4, 5, 6],
    "BuyWeapon":  [7, 8],
}
TASK_POSSETS = {k: set(v) for k, v in BIN_TASKS.items()}

def _period_group(idx_h: int, feat_h: int) -> str:
    # mirrors your predict script
    if feat_h == 0:                return "Calibration"
    if feat_h == 1 and idx_h == 0: return "HoldoutA"
    if idx_h == 1:                 return "HoldoutB"
    return "UNASSIGNED"

# ----------------- 1) build_model ------------------------------------------
def build_model(hparams: dict) -> torch.nn.Module:
    """
    Construct your Performer decoder-only model with product features.
    Required hparams keys (set sensible defaults if absent):
      - vocab_size_tgt, vocab_size_src
      - seq_len_ai, d_model, N (layers), num_heads, d_ff, dropout
      - nb_features, kernel_type
      - feature_xlsx (path to SelectedFigureWeaponEmbeddingIndex.xlsx)
      - special_token_ids (if not using the global SPECIAL_IDS)
    """
    feature_xlsx = hparams.get(
        "feature_xlsx",
        "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx"  # <-- adjust if needed
    )
    feat_tensor = load_feature_tensor(Path(feature_xlsx))  # (V, D_feat)

    special_ids = hparams.get("special_token_ids", SPECIAL_IDS)

    model = build_transformer(
        vocab_size_tgt = hparams["vocab_size_tgt"],
        vocab_size_src = hparams["vocab_size_src"],
        max_seq_len    = hparams["seq_len_ai"],
        d_model        = hparams["d_model"],
        n_layers       = hparams["N"],
        n_heads        = hparams["num_heads"],
        dropout        = hparams.get("dropout", 0.1),
        nb_features    = hparams["nb_features"],
        kernel_type    = hparams["kernel_type"],
        d_ff           = hparams["d_ff"],
        feature_tensor = feat_tensor,
        special_token_ids = special_ids,
    )
    return model


# ----------------- 2) get_dataloaders_for_fold ------------------------------
def get_dataloaders_for_fold(fold_idx: int, num_folds: int, batch_size: int) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Return {'train','val','test'} DataLoaders for this fold.
    Expectations in `hparams` (read from global closure or add as globals here):
      - a config dict `CFG` / or inject via module-global: filepath, seq_len_ai, seq_len_tgt,
        num_heads, ai_rate, pad token id is derived from tokenizer_tgt we build below.
    This function:
      1) loads the full JSON dataset
      2) splits users into K folds (deterministic)
      3) filters raw examples by explicit UID sets
      4) wraps with TransformerDataset and DataLoader
    """
    # You can place CFG in a module-global variable before calling CVRunner.
    # Or, pull them from environment. Below we assume a global get_config() was already used
    # to build `CFG` externally and injected into this module via a global variable.
    global CFG
    assert isinstance(CFG, dict), "Please set global CFG = get_config() before running CV."

    # Load once, then split by UID
    raw = load_json_dataset(CFG["filepath"], keep_uids=None)
    def _uid(rec):  # consistent with your code handling list/scalar uid
        u = rec["uid"]
        return str(u[0] if isinstance(u, list) else u)

    all_uids = sorted({ _uid(r) for r in raw })
    rng = np.random.RandomState(CFG.get("cv_seed", 33))
    shuffled = all_uids[:]
    rng.shuffle(shuffled)
    folds = [shuffled[i::num_folds] for i in range(num_folds)]

    test_u = set(folds[fold_idx])
    val_u  = set(folds[(fold_idx + 1) % num_folds])   # next chunk for validation
    train_u = set(all_uids) - test_u - val_u

    # Filter raw lists
    tr_raw = [r for r in raw if _uid(r) in train_u]
    va_raw = [r for r in raw if _uid(r) in val_u]
    te_raw = [r for r in raw if _uid(r) in test_u]

    # Tokenizers (reuse your builder so PAD aligns)
    tok_src = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()
    pad_id  = tok_tgt.token_to_id("[PAD]")

    # Wrap into your dataset
    def _wrap(records):
        return TransformerDataset(
            records,
            tok_src, tok_tgt,
            CFG["seq_len_ai"],
            CFG["seq_len_tgt"],
            CFG["num_heads"],
            CFG["ai_rate"],
            pad_token=PAD_ID,
        )

    dl = lambda ds, shuf: DataLoader(ds, batch_size=batch_size, shuffle=shuf)

    return {
        "train": dl(_wrap(tr_raw), True),
        "val":   dl(_wrap(va_raw), False),
        "test":  dl(_wrap(te_raw), False),
    }


# ----------------- 3) infer_step --------------------------------------------
def infer_step(model: torch.nn.Module, batch: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Forward one batch and return a *flat* view suitable for ROC-AUC aggregation.
    We:
      - compute probs over decisions (classes 1..9) at decision positions (ai_rate-1, ...)
      - for each (item, time) we create *one row per binary task*
      - y_true ∈ {0,1} and y_pred ∈ [0,1] is the summed prob over that task’s classes
      - task: 'BuyFigure' | ... | 'BuyNone'
      - group: 'Calibration' | 'HoldoutA' | 'HoldoutB' if idx/feat are available, else 'ALL'
    Batch expectations (from your TransformerDataset):
      - batch["aggregate_input"] : LongTensor [B, L]
      - batch["label"]           : LongTensor [B, T] target decisions (1..9, PAD elsewhere)
      - OPTIONAL:
          batch["idx_holdout"]   : LongTensor [B, T] (0/1)
          batch["feat_holdout"]  : LongTensor [B, T] (0/1)
    """
    x   = batch["aggregate_input"].to(device)           # (B, L)
    tgt = batch["label"].to(device)                     # (B, T) decisions 1..9 or PAD

    # Optional grouping tensors (if your dataset emits them; otherwise use 'ALL')
    idx_h  = batch.get("idx_holdout")
    feat_h = batch.get("feat_holdout")
    has_groups = idx_h is not None and feat_h is not None
    if has_groups:
        idx_h  = idx_h.to(device)
        feat_h = feat_h.to(device)

    ai_rate = CFG["ai_rate"]                            # stride between decision positions
    # positions where the model predicts decisions (aligns with your training/eval)
    pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=device)

    # logits -> softmax probs over full vocab (V=60), then slice decisions 1..9
    logits = model(x)[:, pos, :]                        # (B, T, V)
    probs  = F.softmax(logits, dim=-1)[:,:,1:10]        # (B, T, 9) for classes 1..9

    B, T, _ = probs.shape
    # Flatten so we align each time step with its label
    probs_f = probs.reshape(-1, 9)                      # (B*T, 9)
    tgt_f   = tgt.reshape(-1)                           # (B*T,)
    # mask: valid time steps (exclude PAD and specials)
    valid = (tgt_f != PAD_ID) & (tgt_f >= 1) & (tgt_f <= 9)
    probs_f = probs_f[valid]
    tgt_f   = tgt_f[valid]

    # Optional groups flattened
    if has_groups:
        idx_f  = idx_h.reshape(-1)[valid]
        feat_f = feat_h.reshape(-1)[valid]

    # For each row (position) we will emit |BIN_TASKS| rows (one per task)
    y_true_list = []
    y_pred_list = []
    task_list   = []
    group_list  = []

    # vectorized: prepare per-task masks of decisions
    # probs index 0..8 corresponds to decisions 1..9
    posmasks = {t: torch.zeros(9, device=probs_f.device, dtype=probs_f.dtype) for t in BIN_TASKS}
    for task, cls in BIN_TASKS.items():
        posmasks[task][torch.as_tensor([c-1 for c in cls], device=probs_f.device)] = 1.0

    # Compute p_bin for each task by summing probs over its decision set
    for task, mask_vec in posmasks.items():
        p_bin = (probs_f * mask_vec).sum(dim=1)        # (N,)
        y_bin = (tgt_f.unsqueeze(1) == torch.as_tensor(list(TASK_POSSETS[task]), device=tgt_f.device).unsqueeze(0)).any(dim=1).to(torch.int)

        y_true_list.append(y_bin)
        y_pred_list.append(p_bin)
        task_list.append(torch.full_like(y_bin, fill_value=0, dtype=torch.long))  # placeholder; we’ll overwrite with strings
        # group per row
        if has_groups:
            groups = [ _period_group(int(i.item()), int(f.item())) for i,f in zip(idx_f, feat_f) ]
        else:
            groups = ["ALL"] * y_bin.numel()
        group_list.append(groups)

    # Concatenate across tasks
    y_true = torch.cat(y_true_list, dim=0)                  # (N * |tasks|,)
    y_pred = torch.cat(y_pred_list, dim=0)                  # (N * |tasks|,)

    # Build parallel task and group lists expanded to the same length
    tasks_out: list[str] = []
    groups_out: list[str] = []
    n_per_task = y_true_list[0].numel() if y_true_list else 0
    for task_name, groups in zip(BIN_TASKS.keys(), group_list):
        tasks_out.extend([task_name] * n_per_task)
        groups_out.extend(groups)

    return {
        "y_true": y_true.detach().cpu(),                    # Tensor [M]
        "y_pred": y_pred.detach().cpu(),                    # Tensor [M]
        "task":   tasks_out,                                 # List[str] len M
        "group":  groups_out,                                # List[str] len M
    }


# ----------------------------- CV Runner -----------------------------------

@dataclass
class FoldMetrics:
    fold: int
    split: str  # 'val' or 'test'
    task: str
    group: str
    roc_auc: float


class CVRunner:
    def __init__(self, num_folds: int, batch_size: int, hparams: dict, device: str = None):
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.hparams = hparams
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.backends.cudnn.benchmark = True

    def train_one_fold(self, fold_idx: int) -> Tuple[torch.nn.Module, Dict[str, torch.utils.data.DataLoader]]:
        model = build_model(self.hparams).to(self.device)
        dls = get_dataloaders_for_fold(fold_idx, self.num_folds, self.batch_size)

        # ======= Your training loop here =======
        # Example skeleton (replace with your optimizer/scheduler/epochs/early stopping):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.hparams.get("lr", 1e-3))
        epochs = self.hparams.get("epochs", 1)
        for ep in range(epochs):
            for batch in dls['train']:
                optimizer.zero_grad(set_to_none=True)
                # your forward + loss here; placeholder:
                raise NotImplementedError("Implement training step: forward, loss, backward, step")
        # =======================================

        return model, dls

    @torch.no_grad()
    def evaluate_split(self, model: torch.nn.Module, dl: torch.utils.data.DataLoader, split_name: str) -> List[FoldMetrics]:
        model.eval()
        device = self.device

        # Accumulate per (task, group)
        bucket: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}

        for batch in dl:
            out = infer_step(model, batch, device)
            y_true = out['y_true'].detach().cpu().numpy().astype(int)
            y_pred = out['y_pred'].detach().cpu().numpy().astype(float)
            tasks = np.array(out['task'])
            groups = np.array(out['group'])

            for t, g in set(zip(tasks, groups)):
                mask = (tasks == t) & (groups == g)
                if not mask.any():
                    continue
                key = (str(t), str(g))
                if key not in bucket:
                    bucket[key] = []
                # extend with (pred, true)
                bucket[key].extend([(float(p), int(y)) for p, y in zip(y_pred[mask], y_true[mask])])

        # Compute ROC-AUC per (task, group)
        metrics: List[FoldMetrics] = []
        for (task, group), pairs in sorted(bucket.items()):
            preds = np.array([p for p, _ in pairs], dtype=float)
            trues = np.array([y for _, y in pairs], dtype=int)
            try:
                auc = float(roc_auc_score(trues, preds))
            except ValueError:
                # e.g., only one class present; define AUC as NaN
                auc = float('nan')
            metrics.append((task, group, auc))

        # Pack with split name; fold number is added by caller
        return [FoldMetrics(fold=-1, split=split_name, task=t, group=g, roc_auc=a) for (t, g, a) in metrics]

    def run_all(self) -> pd.DataFrame:
        all_rows: List[FoldMetrics] = []
        for fold in range(self.num_folds):
            print(f"\n========== FOLD {fold+1}/{self.num_folds} ==========")
            t0 = time.time()
            model, dls = self.train_one_fold(fold)
            print(f"Training finished in {time.time()-t0:.1f}s")

            # Evaluate val and test
            val_rows = self.evaluate_split(model, dls['val'], split_name='val')
            test_rows = self.evaluate_split(model, dls['test'], split_name='test')
            for r in val_rows: r.fold = fold
            for r in test_rows: r.fold = fold

            # Pretty print per fold
            df_fold = pd.DataFrame([asdict(r) for r in (val_rows + test_rows)])
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df_fold.sort_values(['split','task','group']).to_string(index=False))

            all_rows.extend(val_rows + test_rows)

            # Free GPU memory between folds
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return pd.DataFrame([asdict(r) for r in all_rows])

# ----------------------------- S3 utils -------------------------------------

def s3_upload_dataframe(df: pd.DataFrame, bucket: str, key: str):
    import boto3
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    data = csv_buf.getvalue().encode('utf-8')
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType='text/csv')
    print(f"[S3] Uploaded s3://{bucket}/{key}  (rows={len(df)})")


def make_rocauc_pivot(df_all: pd.DataFrame) -> pd.DataFrame:
    """Make a Task×Group pivot with columns val and test, averaged over folds."""
    # Average over folds first
    avg = (
        df_all
        .groupby(['split','task','group'], as_index=False)['roc_auc']
        .mean()
    )
    # Pivot to two columns (val, test)
    pivot = avg.pivot_table(index=['task','group'], columns='split', values='roc_auc')
    # Order columns if present
    cols = [c for c in ['val','test'] if c in pivot.columns]
    pivot = pivot[cols].reset_index().sort_values(['task','group'])
    return pivot

# ----------------------------- Main -----------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-folds', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--hparams', type=str, default='{}', help='JSON dict of hyperparameters')
    ap.add_argument('--s3-bucket', type=str, required=True)
    ap.add_argument('--s3-prefix', type=str, required=True, help='e.g., ProductGPT/CV/exp_001')
    ap.add_argument('--run-tag', type=str, default=None, help='optional suffix for CSV names')
    return ap.parse_args()

def main():
    args = parse_args()
    try:
        import json
        hparams = json.loads(args.hparams)
    except Exception as e:
        print(f"Failed to parse --hparams as JSON: {e}")
        return 2

    runner = CVRunner(num_folds=args.num_folds, batch_size=args.batch_size, hparams=hparams)
    df_all = runner.run_all()  # columns: fold, split, task, group, roc_auc

    # Print global summary
    print("\n=============  BINARY ROC-AUC (mean over folds)  ==============")
    pivot = make_rocauc_pivot(df_all)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=============================================================")

    # Upload CSVs to S3
    ts = time.strftime('%Y%m%d-%H%M%S')
    base = args.s3_prefix.rstrip('/')
    tag = (args.run_tag + '-') if args.run_tag else ''

    s3_upload_dataframe(df_all, args.s3_bucket, f"{base}/{tag}metrics_per_fold-{ts}.csv")
    s3_upload_dataframe(pivot,  args.s3_bucket, f"{base}/{tag}rocauc_pivot_table-{ts}.csv")

    return 0


if __name__ == '__main__':
    sys.exit(main())


# --- PATCH: enable explicit CV splits in `build_dataloaders` (drop-in for your existing file) ---
# Add the following block near the top of `build_dataloaders(cfg)` *after* you create/locate `raw`
# and *before* the 80/10/10 random_split, so explicit UID splits bypass random splitting.
#
#     # ------------- train / val / test splits -------------
#     if all(k in cfg for k in ("uids_train","uids_val","uids_test")):
#         tok_src = build_tokenizer_src()
#         tok_tgt = build_tokenizer_tgt()
#
#         out_dir = Path(cfg["model_folder"])
#         out_dir.mkdir(parents=True, exist_ok=True)
#         tok_src.save(str(out_dir / "tokenizer_ai.json"))
#         tok_tgt.save(str(out_dir / "tokenizer_tgt.json"))
#
#         # Filter raw by explicit UID sets
#         keepT = set(cfg["uids_train"]) ; keepV = set(cfg["uids_val"]) ; keepE = set(cfg["uids_test"]) 
#         def _keep(u):
#             return str(u[0] if isinstance(u,list) else u)
#         tr_raw = [r for r in raw if _keep(r["uid"]) in keepT]
#         va_raw = [r for r in raw if _keep(r["uid"]) in keepV]
#         te_raw = [r for r in raw if _keep(r["uid"]) in keepE]
#
#         def _wrap(_raw):
#             return TransformerDataset(
#                 _raw, tok_src, tok_tgt,
#                 cfg["seq_len_ai"], cfg["seq_len_tgt"], cfg["num_heads"], cfg["ai_rate"],
#                 pad_token=PAD_ID,
#             )
#         make_loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
#         return (
#             make_loader(_wrap(tr_raw), True),
#             make_loader(_wrap(va_raw), False),
#             make_loader(_wrap(te_raw), False),
#             tok_tgt,
#         )
#
# Leave the existing 80/10/10 branch intact as the fallback when explicit UID sets are *not* supplied.


# ==============================
# run_cv10_feature_performer.py
# ==============================
# 10-fold CV driver for your existing training stack (DeepSpeed, Performer, feature tensor).
# - Uses the small patch above to let `train_model(cfg)` consume explicit train/val/test UID sets.
# - Trains a fresh model per fold (8 train, 1 val, 1 test) with fold_id injected into the run UID.
# - Prints per-fold metrics to stdout (mirrors the formatting in train_model/evaluate).
# - Saves a tidy per-fold CSV (val/test metrics) to S3 and no large local artefacts.
# - Optionally writes a local consolidated CSV under /tmp for quick inspection.

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Import your project entry points
from config4 import get_config
from dataset4_productgpt import load_json_dataset
from train4_decoderonly_performer_feature_aws import train_model  # uses patched build_dataloaders

try:
    import boto3
except Exception:
    boto3 = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--s3-bucket", type=str, help="Override cfg['s3_bucket'] if set")
    p.add_argument("--s3-prefix", type=str, default="CV/results/",
                   help="S3 key prefix to place the consolidated CSV (inside the bucket)")
    p.add_argument("--local-out", type=str, default="/tmp/cv10_metrics.csv",
                   help="Optional local CSV for quick look")
    return p.parse_args()


def _flat_uid(u):
    return str(u[0] if isinstance(u, list) else u)


def make_folds(uids: List[str], K: int, seed: int) -> List[List[str]]:
    rs = np.random.RandomState(seed)
    uids = np.array(uids)
    rs.shuffle(uids)
    # chunk into K folds as equal as possible
    return [uids[i::K].tolist() for i in range(K)]


def s3_upload_df(df: pd.DataFrame, bucket: str, key: str):
    if not boto3:
        print(f"[WARN] boto3 not available; skip S3 upload for s3://{bucket}/{key}")
        return
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
    print(f"[S3] Uploaded s3://{bucket}/{key}  (rows={len(df)})")


def main():
    args = parse_args()
    cfg: Dict[str, Any] = get_config()

    # Allow overriding bucket; ensure it's present for train_model
    if args.s3_bucket:
        cfg["s3_bucket"] = args.s3_bucket
    assert "s3_bucket" in cfg and cfg["s3_bucket"], "cfg['s3_bucket'] must be set (or pass --s3-bucket)."

    # Compute full AI sequence length
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    # Load full dataset once to enumerate user IDs
    raw = load_json_dataset(cfg["filepath"], keep_uids=None)
    all_uids = [_flat_uid(r["uid"]) for r in raw]
    uniq_uids = sorted(set(all_uids))
    print(f"[INFO] Loaded {len(raw)} records → {len(uniq_uids)} unique users for CV")

    folds = make_folds(uniq_uids, args.num_folds, args.seed)

    all_rows = []
    t_start = time.time()
    for k in range(args.num_folds):
        test_u = set(folds[k])
        val_u  = set(folds[(k + 1) % args.num_folds])  # next chunk is validation
        train_u = set(uniq_uids) - test_u - val_u

        cfg_k = dict(cfg)  # shallow copy
        cfg_k.update({
            "fold_id": k,
            "uids_train": list(train_u),
            "uids_val":   list(val_u),
            "uids_test":  list(test_u),
        })

        print(f"\n========== FOLD {k+1}/{args.num_folds} ==========")
        print(f"train={len(train_u)}  val={len(val_u)}  test={len(test_u)}  (users)")

        # Train + eval using your existing routine (uploads best ckpt/metrics to S3 and cleans locals)
        summary = train_model(cfg_k)
        # `summary` includes: uid, fold_id, val_loss, val_f1, val_auprc, test_f1, test_auprc, ckpt, preds

        row = {
            "fold": k,
            "val_loss": summary.get("val_loss"),
            "val_f1": summary.get("val_f1"),
            "val_auprc": summary.get("val_auprc"),
            "test_f1": summary.get("test_f1"),
            "test_auprc": summary.get("test_auprc"),
            "ckpt_name": summary.get("ckpt"),
            "preds_name": summary.get("preds"),
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print("\n=============  CV10 SUMMARY (F1 & AUPRC)  ==============")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("======================================================")

    # Macro means
    means = df[["val_f1","val_auprc","test_f1","test_auprc"]].mean(numeric_only=True)
    print("Mean over folds:")
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")

    # Save consolidated CSV locally (optional) and to S3
    Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.local_out, index=False)
    print(f"[LOCAL] Wrote {args.local_out}")

    bucket = cfg["s3_bucket"]
    prefix = args.s3_prefix.strip("/")
    ts = time.strftime('%Y%m%d-%H%M%S')
    key = f"{prefix}/cv10_summary-{ts}.csv"
    s3_upload_df(df, bucket, key)

    dur = time.time() - t_start
    print(f"[DONE] 10-fold CV finished in {dur/60:.1f} min. Consolidated CSV: s3://{bucket}/{key}")


if __name__ == "__main__":
    sys.exit(main())
