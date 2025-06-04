#!/usr/bin/env python
"""
Summarise every *.json in a directory:
  • keeps scalar metrics (val_*  +  test_*)
  • auto-extracts hyper-params encoded in the file-name
  • unpacks nested val_all / val_cur_stop / val_after_stop / val_transition / test_all / test_cur_stop / …
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
ROOT = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/DecisionOnly')

# ------------------------------------------------------------------
# 2.  Map short tokens → canonical column names
# ------------------------------------------------------------------
REMAP = {
    "h"       : "hidden_size",
    "bs"      : "batch_size",
    "dmodel"  : "d_model",
    "ff"      : "d_ff",
    "n"       : "N",            # encoder layers
    "heads"   : "num_heads",
    "head"    : "num_heads",
    "weight"  : "weight",
    "gamma"   : "gamma",
    "lr"      : "lr",
}

# matches things like "dmodel64", "lr0.0001", "bs32e", etc.
TOKEN_RE = re.compile(r"^([a-zA-Z]+)([0-9e.+\-]+)$", re.I)

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
    Parse tokens from a filename stem of the form:
      DecisionOnly_performer_nb_features16_dmodel64_ff64_N6_heads16_lr0.0001_weight2
    and map keys according to REMAP. Any token that doesn't match REMAP
    is lowered and used as-is.
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
        # 1. Keep any top-level scalar (e.g., val_loss, test_ppl, …)
        if isinstance(v, numbers.Number):
            row[k] = v

        # 2. Unpack any nested dictionary (val_all, val_cur_stop, val_after_stop, val_transition,
        #    test_all, test_cur_stop, etc.)
        elif isinstance(v, dict):
            # Split prefix and subgroup (e.g., "val_all" → ["val","all"])
            parts = k.split("_", 1)
            prefix = parts[0]  # either "val" or "test"
            subgroup = parts[1] if len(parts) > 1 else ""

            for subk, subv in v.items():
                if not isinstance(subv, numbers.Number):
                    continue

                # Map "hit" → "hit_rate", "f1" → "f1_score", else keep subk as-is (e.g. "auprc")
                if subk == "hit":
                    metric = "hit_rate"
                elif subk == "f1":
                    metric = "f1_score"
                else:
                    metric = subk  # e.g., "auprc", "rev_mae", etc.

                # If subgroup == "all", collapse "val_all_hit_rate" → "val_hit_rate"
                if subgroup == "all":
                    colname = f"{prefix}_{metric}"
                else:
                    # e.g. "val_cur_stop_hit_rate", "test_transition_auprc", etc.
                    colname = f"{prefix}_{subgroup}_{metric}"

                row[colname] = subv

        # 3. Keep any string that ends with "_path" (e.g., best_checkpoint_path)
        elif isinstance(v, str) and k.endswith("_path"):
            row[k] = v

    # 4. Extract hyperparameters from the filename stem
    row.update(tokens_from_name(fp.stem))
    rows.append(row)

# ------------------------------------------------------------------
# 4.  DataFrame → Excel
# ------------------------------------------------------------------
df = pd.DataFrame(rows)

# desired column order
front = [
    "file",
    # validation scalars
    "val_loss", "val_ppl",
    # validation: overall
    "val_hit_rate", "val_f1_score", "val_auprc",
    # validation: cur_stop
    "val_cur_stop_hit_rate", "val_cur_stop_f1_score", "val_cur_stop_auprc",
    # validation: after_stop
    "val_after_stop_hit_rate", "val_after_stop_f1_score", "val_after_stop_auprc",
    # validation: transition
    "val_transition_hit_rate", "val_transition_f1_score", "val_transition_auprc",
    # test scalars
    "test_loss", "test_ppl",
    # test: overall
    "test_hit_rate", "test_f1_score", "test_auprc",
    # You can add test_cur_stop_* etc. here if desired
    # hyperparameters (only include if they exist in df.columns)
]

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

# Reorder so that 'front' columns appear first
ordered_cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
df = df[ordered_cols]

out = ROOT / "metrics_summary.xlsx"
df.to_excel(out, index=False)
print(f"✔  {len(df)} files summarised → {out}")
