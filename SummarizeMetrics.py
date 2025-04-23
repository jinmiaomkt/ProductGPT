#!/usr/bin/env python
"""
Aggregate all `metrics_*.json` files in one directory into a single
spreadsheet (confusion matrices are skipped).  Hyper-params that are
*baked into the file name* are extracted automatically.
"""

import json, re
from pathlib import Path
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# 1.  Folder that holds all the JSON files
# ──────────────────────────────────────────────────────────────────
root = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedFull")

# ──────────────────────────────────────────────────────────────────
# 2.  Helper: turn tokens like “h128”, “lr0.0001”, “bs2”
#     into {"hidden_size":128, "lr":0.0001, "batch_size":2}
# ──────────────────────────────────────────────────────────────────
TOKEN_MAP = {                       # pattern → canonical column name
    r"^h(\d+)$"            : "hidden_size",
    r"^bs(\d+)$"           : "batch_size",
    r"^dmodel(\d+)$"       : "d_model",
    r"^dff(\d+)$"          : "d_ff",
    r"^heads?(\d+)$"       : "num_heads",
    r"^n(\d+)$"            : "N",            # encoder layers
    r"^weight(\d+)$"       : "weight",
    r"^lr([0-9e.\-]+)$"    : "lr",
    r"^gamma([0-9e.\-]+)$" : "gamma",
}

def parse_tokens(stem: str) -> dict:
    """
    Split a file-stem like 'metrics_h128_lr0.0001_bs2' into tokens,
    map each token to a column name, and return the dict.
    """
    out = {}
    # drop the 'metrics_' prefix then split on '_' 
    for token in stem.removeprefix("metrics_").split("_"):
        for pattern, col in TOKEN_MAP.items():
            m = re.match(pattern, token, flags=re.I)
            if m:
                # numeric values become float/int when possible
                val = m.group(1)
                try:            val = int(val)
                except ValueError:
                    try:        val = float(val)
                    except ValueError:
                        pass
                out[col] = val
                break
    return out

# ──────────────────────────────────────────────────────────────────
# 3.  Collect rows
# ──────────────────────────────────────────────────────────────────
rows = []

for fp in root.glob("metrics_*.json"):
    with fp.open() as f:
        stats = json.load(f)

    # ---- scalar metrics from inside the JSON ----
    row = {
        "file":            fp.name,
        "val_loss":        stats["val_loss"],
        "val_ppl":         stats["val_ppl"],
        "val_hit_rate":    stats.get("val_hit_rate"),
        "val_macro_f1":    stats.get("val_f1_score"),
        "val_auprc":       stats.get("val_auprc"),
        "checkpoint":      stats.get("checkpoint") or stats.get("best_checkpoint_path"),
    }

    # ---- hyper-params explicitly stored inside the JSON ----
    for k in ("hidden_size", "d_model", "d_ff", "N",
              "num_heads", "lr", "batch_size", "weight", "gamma"):
        if k in stats and k not in row:
            row[k] = stats[k]

    # ---- hyper-params derived from the file-name ----
    row.update(parse_tokens(fp.stem))

    rows.append(row)

# ──────────────────────────────────────────────────────────────────
# 4.  Build DataFrame & export
# ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# Optional: order the columns (put filename & metrics first)
first_cols = ["file", "val_loss", "val_ppl",
              "val_hit_rate", "val_macro_f1", "val_auprc", "checkpoint"]
ordered = first_cols + [c for c in df.columns if c not in first_cols]
df = df[ordered]

# Sort by whatever keys you prefer:
df.sort_values(["hidden_size", "lr", "batch_size"], inplace=True, na_position="last")

out_path = root / "metrics_summary.xlsx"
df.to_excel(out_path, index=False)          # or df.to_csv(...)

print(f"✔  Wrote {len(df)} rows → {out_path}")
