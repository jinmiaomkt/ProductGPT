#!/usr/bin/env python
"""
Summarise every *.json in a directory:
  • keeps scalar metrics (val_*  +  test_*)
  • auto-extracts hyper-params encoded in the file-name
  • handles both flattened keys (e.g. "val_all_hit_rate") and nested dicts
  • exports a single spreadsheet
"""

import json
import re
import numbers
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# 1.  Folder with your result files
# ------------------------------------------------------------------
ROOT = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedFull')

# --------------------------------------------FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0----------------------
# 2.  Map short tokens → canonical column names (from filename)
# ------------------------------------------------------------------
REMAP = {
    "h"       : "hidden_size",
    "bs"      : "batch_size",
    "dmodel"  : "d_model",
    "ff"      : "d_ff",
    "n"       : "N",
    "heads"   : "num_heads",
    "head"    : "num_heads",
    "weight"  : "weight",
    "gamma"   : "gamma",
    "lr"      : "lr",
}

# matches things like "dmodel32", "lr0.0001", "bs64", "nb_features16", etc.
TOKEN_RE = re.compile(r"^([a-zA-Z_]+)([0-9e.+\-]+)$", re.I)

def cast(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def tokens_from_name(stem):
    """
    Parse tokens from a filename stem, e.g.:
      FullProductGPT_indexbased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_weight2
    and map keys according to REMAP. Any token not in REMAP is lowercased as-is.
    """
    out = {}
    for token in stem.split("_"):
        m = TOKEN_RE.match(token)
        if not m:
            continue
        key_raw, raw_val = m.groups()
        key = REMAP.get(key_raw.lower(), key_raw.lower())
        if key not in out:
            out[key] = cast(raw_val)
    return out

# ------------------------------------------------------------------
# 3.  Walk every JSON file and build rows
# ------------------------------------------------------------------
rows = []

for fp in ROOT.glob("*.json"):
    with fp.open() as f:
        stats = json.load(f)

    row = {"file": fp.name}

    for k, v in stats.items():
        # 1. If it's a top-level scalar, keep it (e.g. val_loss, test_ppl, val_all_hit_rate, …)
        if isinstance(v, numbers.Number):
            row[k] = v

        # 2. If it's a nested dict (old format), unpack it
        elif isinstance(v, dict):
            parts = k.split("_", 1)
            prefix = parts[0]            # "val" or "test"
            subgroup = parts[1] if len(parts) > 1 else ""

            for subk, subv in v.items():
                if not isinstance(subv, numbers.Number):
                    continue

                # translate "hit" → "hit_rate", "f1" → "f1_score", else keep subk
                if subk == "hit":
                    metric = "hit_rate"
                elif subk == "f1":
                    metric = "f1_score"
                else:
                    metric = subk  # e.g. "auprc" or "rev_mae"

                # if subgroup=="all", collapse "val_all_hit_rate" → "val_hit_rate"
                if subgroup == "all":
                    colname = f"{prefix}_{metric}"
                else:
                    colname = f"{prefix}_{subgroup}_{metric}"

                row[colname] = subv

        # 3. Keep any string that ends with "_path" (e.g. best_checkpoint_path)
        elif isinstance(v, str) and k.endswith("_path"):
            row[k] = v

        # 4. Otherwise ignore non-numeric, non-dict fields

    # 4. Extract hyperparameters from the filename stem
    row.update(tokens_from_name(fp.stem))
    rows.append(row)

# ------------------------------------------------------------------
# 4.  Build DataFrame → Excel
# ------------------------------------------------------------------
df = pd.DataFrame(rows)

# desired column order (match flattened-JSON keys)
front = [
    "file",
    # validation scalars
    "val_loss", "val_ppl",
    # validation: overall
    "val_all_hit_rate", "val_all_f1_score", "val_all_auprc", "val_all_rev_mae",
    # validation: stop
    "val_stop_hit_rate", "val_stop_f1_score", "val_stop_auprc", "val_stop_rev_mae",
    # validation: after
    "val_after_hit_rate", "val_after_f1_score", "val_after_auprc", "val_after_rev_mae",
    # validation: transition
    "val_transition_hit_rate", "val_transition_f1_score", "val_transition_auprc", "val_transition_rev_mae",
    # test scalars
    "test_loss", "test_ppl",
    # test: overall
    "test_all_hit_rate", "test_all_f1_score", "test_all_auprc", "test_all_rev_mae",
    # test: stop
    "test_stop_hit_rate", "test_stop_f1_score", "test_stop_auprc", "test_stop_rev_mae",
    # test: after
    "test_after_hit_rate", "test_after_f1_score", "test_after_auprc", "test_after_rev_mae",
    # test: transition
    "test_transition_hit_rate", "test_transition_f1_score", "test_transition_auprc", "test_transition_rev_mae",
]

# then hyperparameters, if they exist in df.columns
front += [
    c for c in (
        "hidden_size",
        "d_model",
        "d_ff",
        "N",
        "num_heads",
        "lr",
        "batch_size",
        "weight",
        "gamma",
    )
    if c in df.columns
]

# ensure front columns appear first
ordered_cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
df = df[ordered_cols]

out = ROOT / "metrics_summary.xlsx"
df.to_excel(out, index=False)
print(f"✔  {len(df)} files summarised → {out}")
