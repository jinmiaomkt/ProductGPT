#!/usr/bin/env python3
# =============================================================
#  compare_models_multiclass.py
#  Hit-rate (accuracy), macro-F1, macro-AUPRC for ProductGPT,
#  LSTM, GRU — argmax across mutually-exclusive alternatives.
# =============================================================
import json, ast, gzip, sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# -------------------------------------------------------------
# 0. FILE LOCATIONS – EDIT AS NEEDED
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

LABELS_PATH = Path(
    "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/"
    "clean_list_int_wide4_simple6.json"
)  # one row per uid with ["uid","split","label"]

# -------------------------------------------------------------
# 1. OPTIONAL: explicit alternative order if probs is a list
# -------------------------------------------------------------
# Leave None to auto-detect from dict keys.
ALTERNATIVES = None  # e.g. ["BuyFigure","BuyNone","BuyOne","BuyRegular","BuyTen","BuyWeapon"]

# -------------------------------------------------------------
# 2. Robust JSONL reader
# -------------------------------------------------------------
def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # skip logging lines that start with '['
            if line.startswith("[") and "]" in line.split(maxsplit=1)[0]:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except Exception:
                    if ln <= 3:
                        print(f"[warn] skipped malformed line {ln}: {line[:80]}…")
                    continue

# -------------------------------------------------------------
# 3. Load ground-truth labels & split
# -------------------------------------------------------------
def load_labels(path: Path) -> pd.DataFrame:
    rows = list(read_jsonl(path))
    if not rows:
        raise RuntimeError(f"No rows read from {path}")
    df = pd.DataFrame(rows)
    expected = {"uid", "label", "split"}
    if not expected.issubset(df.columns):
        raise RuntimeError(
            f"Label file must contain columns {expected}. "
            f"Found: {list(df.columns)[:10]}"
        )
    df["split"] = (
        df["split"]
          .astype(str).str.strip().str.title()
          .str.replace("Holdouta","HoldoutA", regex=False)
          .str.replace("Holdoutb","HoldoutB", regex=False)
    )
    return df[["uid","label","split"]]

labels_df = load_labels(LABELS_PATH)
label_set  = sorted(labels_df["label"].unique())

# -------------------------------------------------------------
# 4. Load predictions, merge with labels
# -------------------------------------------------------------
def probs_to_array(p, alt_order):
    """Convert p (dict or list) to an ndarray aligned with alt_order."""
    if isinstance(p, dict):
        return np.array([p[a] for a in alt_order], dtype=float)
    else:  # assume list/tuple
        return np.asarray(p, dtype=float)

records = []
for model, path in FILES.items():
    preds_raw = list(read_jsonl(path))
    if not preds_raw:
        raise RuntimeError(f"No prediction rows read from {path}")

    # --- auto / explicit alternative order
    if ALTERNATIVES is not None:
        alt_order = list(ALTERNATIVES)
    else:
        # grab keys from first dict-style probs
        first_probs = next(p["probs"] for p in preds_raw
                           if isinstance(p["probs"], dict))
        alt_order = list(first_probs.keys())
    alt_index = {a: i for i, a in enumerate(alt_order)}

    # build DF with uid, prob_vector
    uid_list, prob_mat = [], []
    for rec in preds_raw:
        uid_list.append(rec["uid"])
        prob_mat.append(probs_to_array(rec["probs"], alt_order))
    pred_df = pd.DataFrame({"uid": uid_list})
    pred_df["prob_vec"] = prob_mat

    # --- merge with labels
    merged = pred_df.merge(labels_df, on="uid", how="inner")
    if merged.empty:
        raise RuntimeError(
            f"No overlapping uids between predictions ({path.name}) and labels."
        )

    # For macro-AUPRC we need one-hot y_true matrix
    y_true_idx = merged["label"].map(alt_index).to_numpy()
    y_bin      = label_binarize(y_true_idx, classes=np.arange(len(alt_order)))
    prob_array = np.stack(merged["prob_vec"].to_numpy())

    merged["y_true_idx"] = y_true_idx
    merged["prob_array"] = list(prob_array)  # keep handy

    # --- per split metrics
    for split in ["Calibration","HoldoutA","HoldoutB"]:
        m_split = merged[merged["split"] == split]
        if m_split.empty:
            continue

        y_s      = m_split["y_true_idx"].to_numpy()
        prob_s   = np.stack(m_split["prob_array"])
        y_bin_s  = label_binarize(y_s, classes=np.arange(len(alt_order)))

        y_pred   = prob_s.argmax(axis=1)

        records.append({
            "Model": model,
            "Split": split,
            "Hit"  : accuracy_score(y_s, y_pred),
            "F1"   : f1_score     (y_s, y_pred, average="macro"),
            "AUPRC": average_precision_score(y_bin_s, prob_s, average="macro"),
        })

# -------------------------------------------------------------
# 5. Pretty print
# -------------------------------------------------------------
metrics_df = (
    pd.DataFrame(records)
      .sort_values(["Model","Split"])
      .reset_index(drop=True)
)

pd.options.display.float_format = "{:.4f}".format
print("\n====  MULTI-CLASS MODEL COMPARISON  ====\n")
print(metrics_df.to_string(index=False))

# save if you like
# metrics_df.to_csv("model_metrics_multiclass.csv", index=False)
# metrics_df.to_latex("model_metrics_multiclass.tex", index=False, float_format="%.4f")
