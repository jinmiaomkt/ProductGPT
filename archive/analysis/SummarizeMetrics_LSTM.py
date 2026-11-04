#!/usr/bin/env python
"""
Summarise every *.json in a directory:
  • keeps scalar metrics (val_*  +  test_*)
  • auto-extracts hyper-params encoded in the file-name
  • unpacks nested val_all / val_cur_stop / val_after_stop / val_transition
    (and the corresponding test_*) blocks
  • collapses synonymous names so you never get “…_stop_stop_…”
  • exports a single Excel spreadsheet
"""

import json
import re
import numbers
from pathlib import Path
import pandas as pd

# ════════════════════════════════════════════════════════════════════════
# 1.  Folder with your result files
# ════════════════════════════════════════════════════════════════════════
ROOT = Path(
    "/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/"
    "TuningResult/GRU/new"
)

# ════════════════════════════════════════════════════════════════════════
# 2.  Map short tokens → canonical column names (for filename parsing)
# ════════════════════════════════════════════════════════════════════════
REMAP = {
    "h": "hidden_size",
    "bs": "batch_size",
    "dmodel": "d_model",
    "ff": "d_ff",
    "n": "N",
    "heads": "num_heads",
    "head": "num_heads",
    "weight": "weight",
    "gamma": "gamma",
    "lr": "lr",
}

TOKEN_RE = re.compile(r"^([a-zA-Z]+)([0-9e.+\-]+)$", re.I)


def cast(x: str):
    """Try int → float → str in that order."""
    for fn in (int, float):
        try:
            return fn(x)
        except ValueError:
            pass
    return x


def tokens_from_name(stem: str) -> dict:
    """
    Parse tokens from a filename stem, e.g.
      DecisionOnly_performer_nb_features16_dmodel64_ff64_N6_heads16_lr0.0001_weight2
    Returns a {key: value} dict with REMAP applied.
    """
    out = {}
    for token in stem.split("_"):
        m = TOKEN_RE.match(token)
        if not m:
            continue
        key_raw, val_raw = m.groups()
        key = REMAP.get(key_raw.lower(), key_raw.lower())
        # keep first occurrence only
        if key not in out:
            out[key] = cast(val_raw)
    return out


# ════════════════════════════════════════════════════════════════════════
# 3.  Walk every JSON file and build rows
# ════════════════════════════════════════════════════════════════════════
rows = []

for fp in ROOT.glob("*.json"):
    with fp.open() as f:
        stats = json.load(f)

    row = {"file": fp.name}

    for k, v in stats.items():
        # ── 1.  Top-level scalars ───────────────────────────────────────
        if isinstance(v, numbers.Number):
            row[k] = v

        # ── 2.  Nested dicts (val_all, val_cur_stop, …) ────────────────
        elif isinstance(v, dict):
            prefix, *rest = k.split("_", 1)
            subgroup = rest[0] if rest else ""

            for subk, subv in v.items():
                if not isinstance(subv, numbers.Number):
                    continue

                # --- NEW: guard against double "stop_stop_" -------------
                if subk.startswith("stop_"):
                    subk = subk[len("stop_") :]

                # hit → hit_rate, f1 → f1_score
                metric = (
                    "hit_rate"
                    if subk == "hit"
                    else "f1_score"
                    if subk == "f1"
                    else subk
                )

                # collapse “…_all_metric” → “…_metric”
                if subgroup == "all":
                    colname = f"{prefix}_{metric}"
                else:
                    colname = f"{prefix}_{subgroup}_{metric}"

                row[colname] = subv

        # ── 3.  Keep any *_path string (checkpoints) ────────────────────
        elif isinstance(v, str) and k.endswith("_path"):
            row[k] = v

    # ── 4.  Hyper-params from filename ─────────────────────────────────
    row.update(tokens_from_name(fp.stem))
    rows.append(row)

# ════════════════════════════════════════════════════════════════════════
# 4.  Build DataFrame and canonicalise column names
# ════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(rows)

def canon(col: str) -> str:
    """
    • val_all_*    -> val_*
    • test_all_*   -> test_*
    • val_stop_*   -> val_cur_stop_*
    • test_stop_*  -> test_cur_stop_*
    • val_after_*  -> val_after_stop_*   (same for test_)
    • remove any accidental '_stop_stop_'
    """
    col = re.sub(r"^(val|test)_all_", r"\1_", col)
    col = re.sub(r"^(val|test)_stop_", r"\1_cur_stop_", col)
    col = re.sub(r"^(val|test)_after_", r"\1_after_stop_", col)
    return col.replace("_stop_stop_", "_stop_")

df.rename(columns=canon, inplace=True)
df = df.loc[:, ~df.columns.duplicated()]  # keep first when duplicates

# ════════════════════════════════════════════════════════════════════════
# 5.  Optional: order columns (add or remove as needed)
# ════════════════════════════════════════════════════════════════════════
front = [
    "file",
    "val_loss",
    "val_ppl",
    "val_hit_rate",
    "val_f1_score",
    "val_auprc",
    "val_cur_stop_hit_rate",
    "val_cur_stop_f1_score",
    "val_cur_stop_auprc",
    "val_after_stop_hit_rate",
    "val_after_stop_f1_score",
    "val_after_stop_auprc",
    "val_transition_hit_rate",
    "val_transition_f1_score",
    "val_transition_auprc",
    "test_loss",
    "test_ppl",
    "test_hit_rate",
    "test_f1_score",
    "test_auprc",
    # hyper-parameters (include only if present)
    "hidden_size",
    "d_model",
    "d_ff",
    "N",
    "num_heads",
    "lr",
    "batch_size",
    "weight",
    "gamma",
]

ordered_cols = [c for c in front if c in df.columns] + [
    c for c in df.columns if c not in front
]
df = df[ordered_cols]

# ════════════════════════════════════════════════════════════════════════
# 6.  Save spreadsheet
# ════════════════════════════════════════════════════════════════════════
out_file = ROOT / "metrics_summary.xlsx"
df.to_excel(out_file, index=False)
print(f"✔  {len(df)} files summarised → {out_file}")
