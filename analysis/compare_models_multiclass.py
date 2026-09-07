#!/usr/bin/env python3
# =============================================================
#  compare_models_multiclass.py
#  Hit‑rate (accuracy), macro‑F1, macro‑AUPRC for ProductGPT,
#  LSTM, GRU — using argmax over alternatives.
#  Robust to both “long” and “wide” JSON‑Lines predictions.
# =============================================================
import json, sys, ast, gzip
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
)

# -------------------------------------------------------------
#  CONFIGURATION  (edit as needed)
# -------------------------------------------------------------
FILES = {
    "ProductGPT": Path(
        "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/"
        "TuningResult/FeatureBasedFull/"
        "FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_predictions.jsonl"
    ),
    "LSTM": Path(
        "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/"
        "TuningResult/LSTM/lstm_predictions.jsonl"
    ),
    "GRU": Path(
        "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/"
        "TuningResult/GRU/gru_predictions.jsonl"
    ),
}

# If your “probs” field is an *array* (not a dict) you must tell the
# script what order the alternatives appear in that array:
ALT_ORDER = [
    "BuyFigure", "BuyNone", "BuyOne", "BuyRegular", "BuyTen"
]

# -------------------------------------------------------------
#  COMMON ALIASES                                             –
# -------------------------------------------------------------
CANDIDATE_SPLIT_FIELDS = ["split", "period", "phase", "group", "set"]
CANDIDATE_ID_FIELDS    = ["event_id", "uid", "id"]
CANDIDATE_TRUE_FIELDS  = ["y_true", "true", "label", "label_idx"]

# Will be standardised to these canonical names
CANON_SPLIT = "split"
CANON_ID    = "event_id"
CANON_TASK  = "task"
CANON_TRUE  = "y_true"
CANON_PRED  = "y_pred"

# -------------------------------------------------------------
def robust_jsonl_reader(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSON‑Lines (optionally .gz) file and return list[dict].
       Falls back to ast.literal_eval for single‑quotes or NaNs.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    rows, bad = [], []
    with opener(path, "rt", encoding="utf‑8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    rows.append(ast.literal_eval(line))
                except Exception:
                    if len(bad) < 3:
                        print(f"[warn] skipped malformed line {ln}: {line[:80]}…")
                    bad.append(ln)
    if bad:
        print(f"[warn] skipped {len(bad)} malformed lines in {path.name}")
    return rows


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ID / split / truth columns to canonical names where present."""
    # Split
    for c in CANDIDATE_SPLIT_FIELDS:
        if c in df.columns:
            df.rename(columns={c: CANON_SPLIT}, inplace=True)
            break
    else:
        df[CANON_SPLIT] = "ALL"

    # Event‑ID
    for c in CANDIDATE_ID_FIELDS:
        if c in df.columns:
            df.rename(columns={c: CANON_ID}, inplace=True)
            break
    else:
        df.insert(0, CANON_ID, np.arange(len(df)))

    # y_true
    for c in CANDIDATE_TRUE_FIELDS:
        if c in df.columns:
            df.rename(columns={c: CANON_TRUE}, inplace=True)
            break

    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide df (one row per event with a 'probs' field) to long."""
    if "probs" not in df.columns:
        # Already long format
        return df

    records = []
    for row in df.itertuples(index=False):
        probs = getattr(row, "probs")
        # Determine the alternative → prob mapping
        if isinstance(probs, dict):
            alt_prob_pairs = probs.items()
        elif isinstance(probs, list):
            alt_prob_pairs = zip(ALT_ORDER, probs)
        else:
            raise TypeError("`probs` must be dict or list/tuple.")

        # Identify the true alternative
        true_alt = getattr(row, CANON_TRUE, None)

        for alt, p in alt_prob_pairs:
            records.append(
                {
                    CANON_ID:   getattr(row, CANON_ID),
                    CANON_SPLIT:getattr(row, CANON_SPLIT),
                    CANON_TASK: alt,
                    CANON_TRUE: 1 if alt == true_alt else 0,
                    CANON_PRED: float(p),
                }
            )
    return pd.DataFrame.from_records(records)


def load_long_df(path: Path) -> pd.DataFrame:
    """Load either ‘long’ or ‘wide’ predictions into a *long* DataFrame."""
    df = pd.DataFrame(robust_jsonl_reader(path))
    if df.empty:
        raise RuntimeError(f"{path} produced an empty DataFrame.")

    df = standardise_columns(df)

    # If already has task & y_pred we assume long format
    if {CANON_TASK, CANON_PRED}.issubset(df.columns):
        # ensure we have y_true (some generators omit non‑chosen alt rows)
        if CANON_TRUE not in df.columns:
            raise RuntimeError(
                f"{path} lacks a {CANON_TRUE!r} column. Provide true labels."
            )
        return df

    # Otherwise treat as wide
    df_long = wide_to_long(df)
    missing = {CANON_TASK, CANON_PRED, CANON_TRUE} - set(df_long.columns)
    if missing:
        raise RuntimeError(
            f"After wide→long, still missing columns: {missing}. "
            "Check your file format."
        )
    return df_long


# ---------- utilities: long → matrices ------------------------
def long_to_wide(df: pd.DataFrame):
    alts = df[CANON_TASK].unique().tolist()
    alt_idx = {a: i for i, a in enumerate(alts)}

    probs, labels, splits = defaultdict(lambda: np.zeros(len(alts))), {}, {}
    for row in df.itertuples(index=False):
        e_id   = getattr(row, CANON_ID)
        task   = getattr(row, CANON_TASK)
        split  = getattr(row, CANON_SPLIT)
        ytrue  = getattr(row, CANON_TRUE)
        yprob  = getattr(row, CANON_PRED)

        probs[e_id][alt_idx[task]] = yprob
        if ytrue == 1:
            labels[e_id] = alt_idx[task]
        splits[e_id] = split

    event_ids   = list(probs.keys())
    prob_matrix = np.vstack([probs[e] for e in event_ids])
    true_labels = np.array([labels[e] for e in event_ids], dtype=int)
    split_vec   = np.array([splits[e] for e in event_ids])

    return prob_matrix, true_labels, split_vec, alts

# -------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------
records = []
for model, path in FILES.items():
    df_long = load_long_df(path)
    prob, y, split_vec, alts = long_to_wide(df_long)
    y_bin = label_binarize(y, classes=np.arange(len(alts)))

    for split in np.unique(split_vec):
        mask = split_vec == split
        if mask.sum() == 0:
            continue
        y_s, prob_s, y_bin_s = y[mask], prob[mask], y_bin[mask]

        y_pred = prob_s.argmax(axis=1)
        records.append(
            {
                "Model":  model,
                "Split":  split,
                "Hit":    accuracy_score(y_s, y_pred),
                "F1":     f1_score(y_s, y_pred, average="macro"),
                "AUPRC":  average_precision_score(y_bin_s, prob_s, average="macro"),
            }
        )

metrics_df = (
    pd.DataFrame(records)
      .sort_values(["Model", "Split"])
      .reset_index(drop=True)
)

print("\n====  MULTI‑CLASS MODEL COMPARISON  ====\n")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# If you need LaTeX / CSV output:
# metrics_df.to_latex("model_metrics_multiclass.tex", index=False, float_format="%.4f")
# metrics_df.to_csv("model_metrics_multiclass.csv", index=False)
