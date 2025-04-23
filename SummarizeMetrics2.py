#!/usr/bin/env python
"""
Summarise every *.json in a directory:
  • keeps only scalar metrics (val_*  +  test_*)
  • auto-extracts hyper-params encoded in the file-name
  • exports a single spreadsheet
"""

import json, re, numbers
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# 1.  Folder with your result files
# ------------------------------------------------------------------
ROOT = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/IndexBasedFull")

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

TOKEN_RE = re.compile(r"([a-zA-Z]+)([0-9e.+\-]+)$", re.I)

def cast(x: str):
    """best-effort numeric coercion"""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x                    # fall back to raw string

def tokens_from_name(stem: str) -> dict:
    """
    'FeatureBased_FullProductGPT_dmodel16_ff64_N2_heads8_lr1e-05_weight4' →
        {'d_model':16, 'd_ff':64, 'N':2, 'num_heads':8, 'lr':1e-05, 'weight':4}
    """
    out = {}
    for token in stem.split("_"):
        m = TOKEN_RE.match(token)
        if not m:
            continue                    # skip words like 'FeatureBased'
        key, raw_val = m.groups()
        key = REMAP.get(key.lower(), key.lower())   # rename if needed
        if key not in out:              # first hit wins
            out[key] = cast(raw_val)
    return out

# ------------------------------------------------------------------
# 3.  Walk every JSON file
# ------------------------------------------------------------------
rows = []

for fp in ROOT.glob("*.json"):
    with fp.open() as f:
        stats = json.load(f)

    # ---- scalar metrics available inside the JSON ----
    row = {"file": fp.name}
    for k, v in stats.items():
        if isinstance(v, numbers.Number):              # any scalar
            row[k] = v
        elif k.endswith("_path"):                       # keep path strings
            row[k] = v

    # ---- hyper-params derived from filename tokens ----
    row.update(tokens_from_name(fp.stem))

    rows.append(row)

# ------------------------------------------------------------------
# 4.  DataFrame  →  Excel
# ------------------------------------------------------------------
df = pd.DataFrame(rows)

# a handy column order (edit as you like)
front = ["file",
         "val_loss", "val_ppl", "val_hit_rate", "val_f1_score", "val_auprc",
         "test_loss", "test_ppl", "test_hit_rate", "test_f1_score", "test_auprc"]
front += [c for c in ("hidden_size", "d_model", "d_ff", "N",
                      "num_heads", "lr", "batch_size", "weight", "gamma")
          if c in df.columns]
df = df[[c for c in front if c in df.columns] +  # desired leading cols
        [c for c in df.columns if c not in front]]  # everything else

out = ROOT / "metrics_summary.xlsx"
df.to_excel(out, index=False)
print(f"✔  {len(df)} files summarised → {out}")
