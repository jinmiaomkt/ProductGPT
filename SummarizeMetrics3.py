#!/usr/bin/env python
"""
Summarise every *.json in a directory:
  • keeps scalar metrics (val_*  +  test_*)
  • auto-extracts hyper-params encoded in the file-name
  • unpacks nested val_all / val_transition / etc.
  • exports a single spreadsheet
"""

import json, re, numbers
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# 1.  Folder with your result files
# ------------------------------------------------------------------
ROOT = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/DecisionOnly/metrics")

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
    try: return int(x)
    except ValueError:
        try: return float(x)
        except ValueError:
            return x

def tokens_from_name(stem: str) -> dict:
    out = {}
    for token in stem.split("_"):
        m = TOKEN_RE.match(token)
        if not m:
            continue
        key, raw_val = m.groups()
        key = REMAP.get(key.lower(), key.lower())
        if key not in out:
            out[key] = cast(raw_val)
    return out

# ------------------------------------------------------------------
# 3.  Walk every JSON file
# ------------------------------------------------------------------
rows = []

for fp in ROOT.glob("*.json"):
    with fp.open() as f:
        stats = json.load(f)

    row = {"file": fp.name}

    for k, v in stats.items():
        if isinstance(v, numbers.Number):
            row[k] = v
        elif isinstance(v, dict) and k.startswith("val_"):
            # e.g., val_all → val_hit_rate, val_f1_score, ...
            for subk, subv in v.items():
                if isinstance(subv, numbers.Number):
                    colname = f"{k}_{subk}".replace("hit", "hit_rate").replace("f1", "f1_score")
                    row[colname] = subv
        elif isinstance(v, str) and k.endswith("_path"):
            row[k] = v

    row.update(tokens_from_name(fp.stem))
    rows.append(row)

# ------------------------------------------------------------------
# 4.  DataFrame → Excel
# ------------------------------------------------------------------
df = pd.DataFrame(rows)

# desired column order
front = ["file",
         "val_loss", "val_ppl",
         "val_all_hit_rate", "val_all_f1_score", "val_all_auprc",
         "val_cur_stop_hit_rate", "val_cur_stop_f1_score", "val_cur_stop_auprc",
         "val_after_stop_hit_rate", "val_after_stop_f1_score", "val_after_stop_auprc",
         "val_transition_hit_rate", "val_transition_f1_score", "val_transition_auprc",
         "test_loss", "test_ppl", "test_hit_rate", "test_f1_score", "test_auprc"]

front += [c for c in ("hidden_size", "d_model", "d_ff", "N",
                      "num_heads", "lr", "batch_size", "weight", "gamma")
          if c in df.columns]

# ensure front columns are shown first
df = df[[c for c in front if c in df.columns] +
        [c for c in df.columns if c not in front]]

out = ROOT / "metrics_summary.xlsx"
df.to_excel(out, index=False)
print(f"✔  {len(df)} files summarised → {out}")
