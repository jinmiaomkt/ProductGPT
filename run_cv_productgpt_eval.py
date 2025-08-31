#!/usr/bin/env python3
"""
10-fold CV training + evaluation (fixed imports + builders).

WHAT WAS FIXED
- Missing imports: Path, pandas, roc_auc_score
- Tokenizers: build_tokenizer_src/tgt added, PAD_ID derived from tgt tokenizer
- Special/product IDs defined; SPECIAL_IDS uses real PAD
- load_feature_tensor implemented (Excel -> (V,D) FloatTensor)
- DataLoader now passes pad_token=pad_id (not undefined PAD_ID)
"""

from __future__ import annotations
import argparse
import io
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# --- NEW imports that were missing
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

# --- project-local imports (exist in your repo)
from dataset4_productgpt import TransformerDataset, load_json_dataset
from model4_decoderonly_feature_performer import build_transformer
from config4 import get_config
from train1_decision_only_performer_aws import _build_tok  # fallback tokenizer builder

CFG = get_config()
CFG["seq_len_ai"] = CFG["ai_rate"] * CFG["seq_len_tgt"]
CFG["cv_seed"] = 33  # reproducible folds

# ===== Binary tasks (same as your eval script)
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
    if feat_h == 0:                return "Calibration"
    if feat_h == 1 and idx_h == 0: return "HoldoutA"
    if idx_h == 1:                 return "HoldoutB"
    return "UNASSIGNED"

# ===== Product / special token constants + feature tensor loader
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
MAX_TOKEN_ID = UNK_PROD_ID

FEATURE_COLS = [
    "Rarity","MaxLife","MaxOffense","MaxDefense",
    "WeaponTypeOneHandSword","WeaponTypeTwoHandSword","WeaponTypeArrow","WeaponTypeMagic","WeaponTypePolearm",
    "EthnicityIce","EthnicityRock","EthnicityWater","EthnicityFire","EthnicityThunder","EthnicityWind",
    "GenderFemale","GenderMale","CountryRuiYue","CountryDaoQi","CountryZhiDong","CountryMengDe",
    "type_figure","MinimumAttack","MaximumAttack","MinSpecialEffect","MaxSpecialEffect","SpecialEffectEfficiency",
    "SpecialEffectExpertise","SpecialEffectAttack","SpecialEffectSuper","SpecialEffectRatio","SpecialEffectPhysical",
    "SpecialEffectLife","LTO",
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

# ===== Tokenizers & PAD/SPECIAL IDs
_TOK_SRC = None
_TOK_TGT = None
PAD_ID: int | None = None
SPECIAL_IDS: List[int] | None = None

def build_tokenizer_src() -> Tokenizer:
    """Load src tokenizer from model_folder or fallback."""
    global _TOK_SRC
    if _TOK_SRC is not None:
        return _TOK_SRC
    model_dir = Path(CFG["model_folder"])
    path = model_dir / "tokenizer_ai.json"
    if path.exists():
        _TOK_SRC = Tokenizer.from_file(str(path))
    else:
        # Fallback to your builder (numeric/word-level)
        _TOK_SRC = _build_tok()
    return _TOK_SRC

def build_tokenizer_tgt() -> Tokenizer:
    """Load tgt tokenizer from model_folder or fallback."""
    global _TOK_TGT
    if _TOK_TGT is not None:
        return _TOK_TGT
    model_dir = Path(CFG["model_folder"])
    path = model_dir / "tokenizer_tgt.json"
    if path.exists():
        _TOK_TGT = Tokenizer.from_file(str(path))
    else:
        _TOK_TGT = _build_tok()
    return _TOK_TGT

def ensure_tokenizers_and_ids():
    """Initialize tokenizers, PAD_ID, and SPECIAL_IDS once."""
    global PAD_ID, SPECIAL_IDS
    tok_tgt = build_tokenizer_tgt()
    if PAD_ID is None:
        PAD_ID = tok_tgt.token_to_id("[PAD]")  # must match training
    if SPECIAL_IDS is None:
        SPECIAL_IDS = [PAD_ID, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

# ----------------- 1) build_model ------------------------------------------
def build_model(hparams: dict) -> torch.nn.Module:
    """
    Construct Performer decoder-only model with product features.
    """
    ensure_tokenizers_and_ids()
    feature_xlsx = hparams.get("feature_xlsx", "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")
    feat_tensor = load_feature_tensor(Path(feature_xlsx))

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
def get_dataloaders_for_fold(fold_idx: int, num_folds: int, batch_size: int) -> Dict[str, DataLoader]:
    """
    Return {'train','val','test'} DataLoaders for this fold.
    """
    ensure_tokenizers_and_ids()
    assert isinstance(CFG, dict), "Please set global CFG = get_config() before running CV."

    raw = load_json_dataset(CFG["filepath"], keep_uids=None)

    def _uid(rec):
        u = rec["uid"]
        return str(u[0] if isinstance(u, list) else u)

    all_uids = sorted({ _uid(r) for r in raw })
    rng = np.random.RandomState(CFG.get("cv_seed", 33))
    shuffled = all_uids[:]
    rng.shuffle(shuffled)
    folds = [shuffled[i::num_folds] for i in range(num_folds)]

    test_u = set(folds[fold_idx])
    val_u  = set(folds[(fold_idx + 1) % num_folds])
    train_u = set(all_uids) - test_u - val_u

    tr_raw = [r for r in raw if _uid(r) in train_u]
    va_raw = [r for r in raw if _uid(r) in val_u]
    te_raw = [r for r in raw if _uid(r) in test_u]

    tok_src = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()
    pad_id  = tok_tgt.token_to_id("[PAD]")

    def _wrap(records):
        return TransformerDataset(
            records,
            tok_src, tok_tgt,
            CFG["seq_len_ai"], CFG["seq_len_tgt"],
            CFG["num_heads"], CFG["ai_rate"],
            pad_token=pad_id,           # <-- FIX: use real pad_id
        )

    mk = lambda ds, shuf: DataLoader(ds, batch_size=batch_size, shuffle=shuf)
    return {"train": mk(_wrap(tr_raw), True),
            "val":   mk(_wrap(va_raw), False),
            "test":  mk(_wrap(te_raw), False)}

# ----------------- 3) infer_step --------------------------------------------
def infer_step(model: torch.nn.Module, batch: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Forward one batch and return a flat view suitable for ROC-AUC aggregation.
    """
    ensure_tokenizers_and_ids()
    assert PAD_ID is not None

    x   = batch["aggregate_input"].to(device)  # (B, L)
    tgt = batch["label"].to(device)            # (B, T) decision IDs 1..9 or PAD

    idx_h  = batch.get("idx_holdout")
    feat_h = batch.get("feat_holdout")
    has_groups = idx_h is not None and feat_h is not None
    if has_groups:
        idx_h  = idx_h.to(device)
        feat_h = feat_h.to(device)

    ai_rate = CFG["ai_rate"]
    pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=device)

    logits = model(x)[:, pos, :]                 # (B, T, V)
    probs  = F.softmax(logits, dim=-1)[:, :, 1:10]  # (B, T, 9) → classes 1..9

    B, T, _ = probs.shape
    probs_f = probs.reshape(-1, 9)               # (B*T, 9)
    tgt_f   = tgt.reshape(-1)                    # (B*T,)
    valid = (tgt_f != PAD_ID) & (tgt_f >= 1) & (tgt_f <= 9)
    probs_f = probs_f[valid]
    tgt_f   = tgt_f[valid]

    if has_groups:
        idx_f  = idx_h.reshape(-1)[valid]
        feat_f = feat_h.reshape(-1)[valid]

    # build per-task masks once
    posmasks = {}
    for task, cls in BIN_TASKS.items():
        mask = torch.zeros(9, device=probs_f.device, dtype=probs_f.dtype)
        mask[torch.tensor([c-1 for c in cls], device=probs_f.device)] = 1.0
        posmasks[task] = mask

    y_true_list, y_pred_list, task_list, group_list = [], [], [], []
    for task, mask_vec in posmasks.items():
        p_bin = (probs_f * mask_vec).sum(dim=1)  # (N,)
        # y_bin: is tgt ∈ positive set?
        posset = torch.tensor(sorted(list(TASK_POSSETS[task])), device=tgt_f.device)
        y_bin = (tgt_f.unsqueeze(1) == posset.unsqueeze(0)).any(dim=1).to(torch.int)

        y_true_list.append(y_bin)
        y_pred_list.append(p_bin)
        if has_groups:
            groups = [ _period_group(int(i.item()), int(f.item())) for i,f in zip(idx_f, feat_f) ]
        else:
            groups = ["ALL"] * y_bin.numel()
        group_list.append(groups)

    y_true = torch.cat(y_true_list, dim=0).cpu()
    y_pred = torch.cat(y_pred_list, dim=0).cpu()

    tasks_out: List[str] = []
    groups_out: List[str] = []
    n_per_task = y_true_list[0].numel() if y_true_list else 0
    for task_name, groups in zip(BIN_TASKS.keys(), group_list):
        tasks_out.extend([task_name] * n_per_task)
        groups_out.extend(groups)

    return {"y_true": y_true, "y_pred": y_pred, "task": tasks_out, "group": groups_out}

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

    def train_one_fold(self, fold_idx: int) -> Tuple[torch.nn.Module, Dict[str, DataLoader]]:
        # ensure PAD/SPECIAL set before building model
        ensure_tokenizers_and_ids()

        # hparams must include model sizes + vocab sizes
        hp = dict(self.hparams)
        hp.setdefault("feature_xlsx", "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")
        hp.setdefault("seq_len_ai", CFG["seq_len_ai"])
        hp.setdefault("vocab_size_tgt", CFG["vocab_size_tgt"])
        hp.setdefault("vocab_size_src", CFG["vocab_size_src"])
        hp.setdefault("d_model", 32)
        hp.setdefault("N", 6)
        hp.setdefault("num_heads", 4)
        hp.setdefault("d_ff", 32)
        hp.setdefault("dropout", 0.1)
        hp.setdefault("nb_features", 16)
        hp.setdefault("kernel_type", CFG.get("kernel_type", "favor+"))

        model = build_model(hp).to(self.device)
        dls = get_dataloaders_for_fold(fold_idx, self.num_folds, self.batch_size)

        # ======= TRAINING LOOP (plug your loss here) =======
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.get("lr", 1e-3))
        epochs = hp.get("epochs", 1)
        for ep in range(epochs):
            for batch in dls['train']:
                optimizer.zero_grad(set_to_none=True)
                # TODO: implement your forward + loss; placeholder raises to remind you
                raise NotImplementedError("Implement training step (forward, loss, backward, step).")
        # ===================================================

        return model, dls

    @torch.no_grad()
    def evaluate_split(self, model: torch.nn.Module, dl: DataLoader, split_name: str) -> List[FoldMetrics]:
        model.eval()
        device = self.device

        # Accumulate per (task, group)
        bucket: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}

        for batch in dl:
            out = infer_step(model, batch, device)
            y_true = out['y_true'].numpy().astype(int)
            y_pred = out['y_pred'].numpy().astype(float)
            tasks = np.array(out['task'])
            groups = np.array(out['group'])

            for t, g in set(zip(tasks, groups)):
                mask = (tasks == t) & (groups == g)
                if not mask.any():
                    continue
                key = (str(t), str(g))
                if key not in bucket:
                    bucket[key] = []
                bucket[key].extend([(float(p), int(y)) for p, y in zip(y_pred[mask], y_true[mask])])

        metrics: List[FoldMetrics] = []
        for (task, group), pairs in sorted(bucket.items()):
            preds = np.array([p for p, _ in pairs], dtype=float)
            trues = np.array([y for _, y in pairs], dtype=int)
            try:
                auc = float(roc_auc_score(trues, preds))
            except ValueError:
                auc = float('nan')
            metrics.append((task, group, auc))

        return [FoldMetrics(fold=-1, split=split_name, task=t, group=g, roc_auc=a) for (t, g, a) in metrics]

    def run_all(self) -> pd.DataFrame:
        all_rows: List[FoldMetrics] = []
        for fold in range(self.num_folds):
            print(f"\n========== FOLD {fold+1}/{self.num_folds} ==========")
            t0 = time.time()
            model, dls = self.train_one_fold(fold)
            print(f"Training finished in {time.time()-t0:.1f}s")

            val_rows = self.evaluate_split(model, dls['val'], split_name='val')
            test_rows = self.evaluate_split(model, dls['test'], split_name='test')
            for r in val_rows: r.fold = fold
            for r in test_rows: r.fold = fold

            df_fold = pd.DataFrame([asdict(r) for r in (val_rows + test_rows)])
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df_fold.sort_values(['split','task','group']).to_string(index=False))

            all_rows.extend(val_rows + test_rows)

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
    boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=data, ContentType='text/csv')
    print(f"[S3] Uploaded s3://{bucket}/{key}  (rows={len(df)})")

def make_rocauc_pivot(df_all: pd.DataFrame) -> pd.DataFrame:
    """Task×Group pivot with columns val and test, averaged over folds."""
    avg = (df_all.groupby(['split','task','group'], as_index=False)['roc_auc'].mean())
    pivot = avg.pivot_table(index=['task','group'], columns='split', values='roc_auc')
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
    df_all = runner.run_all()

    print("\n=============  BINARY ROC-AUC (mean over folds)  ==============")
    pivot = make_rocauc_pivot(df_all)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=============================================================")

    ts = time.strftime('%Y%m%d-%H%M%S')
    base = args.s3_prefix.rstrip('/')
    tag = (args.run_tag + '-') if args.run_tag else ''
    s3_upload_dataframe(df_all, args.s3_bucket, f"{base}/{tag}metrics_per_fold-{ts}.csv")
    s3_upload_dataframe(pivot,  args.s3_bucket, f"{base}/{tag}rocauc_pivot_table-{ts}.csv")
    return 0

if __name__ == '__main__':
    sys.exit(main())
