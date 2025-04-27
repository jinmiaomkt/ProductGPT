#!/usr/bin/env python
"""
Aggregate all *.json files in one directory into a single Excel or CSV
spreadsheet (confusion matrices are skipped).

The script extracts hyper-parameters that are *baked into the file name*
(e.g. “…_lr0.0001_bs2.json”) as well as metrics stored inside the JSON.

Examples
--------
# default (Excel) output
python summarise_metrics.py --root "/path/to/folder"

# CSV output, custom glob pattern
python summarise_metrics.py --root "/path" --csv --pattern "metrics_*.json"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ────────────────────────────────────────────────────────────────
# 1.  CLI arguments
# ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Summarise metric JSON files into a spreadsheet.",
    )
    ap.add_argument("--root", required=True, help="Folder containing JSON files")
    ap.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern to match filenames (relative to --root)",
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help="Write CSV instead of XLSX",
    )
    return ap.parse_args()


# ────────────────────────────────────────────────────────────────
# 2.  Helper: map filename tokens → canonical column names
# ────────────────────────────────────────────────────────────────
TOKEN_MAP: dict[str, str] = {
    r"^h(\d+)$": "hidden_size",
    r"^bs(\d+)$": "batch_size",
    r"^dmodel(\d+)$": "d_model",
    r"^dff(\d+)$": "d_ff",
    r"^heads?(\d+)$": "num_heads",
    r"^n(\d+)$": "N",  # encoder layers
    r"^weight(\d+)$": "weight",
    r"^lr([0-9e.\-]+)$": "lr",
    r"^gamma([0-9e.\-]+)$": "gamma",
}


def parse_tokens(stem: str) -> dict[str, Any]:
    """
    Split a file-stem like 'DecisionOnly_dmodel32_ff32_lr0.0001_bs2'
    into tokens, map each token to a column name, and return the dict.
    """
    out: dict[str, Any] = {}
    # drop an optional 'metrics_' prefix then split on '_' 
    for token in stem.removeprefix("metrics_").split("_"):
        for pattern, col in TOKEN_MAP.items():
            if (m := re.match(pattern, token, flags=re.I)):
                val = m.group(1)
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                out[col] = val
                break
    return out


# ────────────────────────────────────────────────────────────────
# 3.  Main aggregation logic
# ────────────────────────────────────────────────────────────────
def summarise(root: Path, pattern: str = "*.json") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for fp in sorted(root.glob(pattern)):
        if fp.suffix.lower() != ".json":
            continue

        try:
            stats = json.loads(fp.read_text())
        except Exception as e:  # pragma: no cover
            print(f"[warn] Skipping {fp.name}: cannot read JSON ({e})", file=sys.stderr)
            continue

        # ---- scalar metrics from inside the JSON ----
        row: dict[str, Any] = {
            "file": fp.name,
            "val_loss": stats.get("val_loss"),
            "val_ppl": stats.get("val_ppl"),
            "val_hit_rate": stats.get("val_hit_rate"),
            "val_macro_f1": stats.get("val_f1_score"),
            "val_auprc": stats.get("val_auprc"),
            "checkpoint": stats.get("checkpoint") or stats.get("best_checkpoint_path"),
        }

        # ---- hyper-params explicitly stored inside the JSON ----
        for k in (
            "hidden_size",
            "d_model",
            "d_ff",
            "N",
            "num_heads",
            "lr",
            "batch_size",
            "weight",
            "gamma",
        ):
            if k in stats and row.get(k) is None:
                row[k] = stats[k]

        # ---- hyper-params derived from the filename ----
        row.update(parse_tokens(fp.stem))

        rows.append(row)

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────
# 4.  Entrypoint
# ────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        sys.exit(f"[error] {root} is not a directory")

    df = summarise(root, args.pattern)

    if df.empty:
        sys.exit(
            f"[error] No JSON files matched '{args.pattern}' in {root}\n"
            "Check the path or filename pattern."
        )

    # reorder columns – metrics first
    first_cols = [
        "file",
        "val_loss",
        "val_ppl",
        "val_hit_rate",
        "val_macro_f1",
        "val_auprc",
        "checkpoint",
    ]
    ordered = first_cols + [c for c in df.columns if c not in first_cols]
    df = df[ordered]

    # sensible sort keys if present
    sort_keys = [c for c in ("hidden_size", "d_model", "lr", "batch_size") if c in df]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True, na_position="last")

    # choose output format
    out_path = root / "metrics_summary.csv" if args.csv else root / "metrics_summary.xlsx"
    if args.csv:
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)

    print(f"✔  wrote {len(df):,} rows × {len(df.columns)} cols → {out_path}")


if __name__ == "__main__":
    main()
