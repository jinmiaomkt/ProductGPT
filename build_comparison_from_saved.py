#!/usr/bin/env python3
"""
build_comparison_from_saved.py

Reads the per-model CSVs already saved in /tmp/unified_eval_flash/
and builds cross-model comparison tables without re-running inference.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/tmp/unified_eval_flash")
MODELS = [d.name for d in BASE.iterdir() if d.is_dir() and (d / "multiclass_metrics_long.csv").exists()]
MODELS.sort()

SPLIT = "test"

print(f"Found models: {MODELS}")
print(f"Comparing on: {SPLIT}")

# ═══════════════════════════════════════════════════════════
# 1. Multiclass comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MULTICLASS METRICS (test)")
print("=" * 80)

multi_frames = []
for name in MODELS:
    df = pd.read_csv(BASE / name / "multiclass_metrics_long.csv")
    df = df[df["Split"] == SPLIT].copy()
    df["Model"] = name
    multi_frames.append(df)

multi = pd.concat(multi_frames, ignore_index=True)

for group in ["Calibration", "HoldoutA", "HoldoutB"]:
    sub = multi[multi["Group"] == group]
    if sub.empty:
        continue
    print(f"\n--- {group} ---")
    cols = ["Model", "MacroOvR_AUC", "MacroAUPRC", "MacroF1", "MCC", "LogLoss", "Top1Acc", "Top2Acc"]
    cols = [c for c in cols if c in sub.columns]
    print(sub[cols].to_string(index=False))

# ═══════════════════════════════════════════════════════════
# 2. Binary AUC comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("BINARY TASK AUC (test)")
print("=" * 80)

bin_frames = []
for name in MODELS:
    df = pd.read_csv(BASE / name / "binary_metrics_long.csv")
    df = df[df["Split"] == SPLIT].copy()
    df["Model"] = name
    bin_frames.append(df)

binary = pd.concat(bin_frames, ignore_index=True)
paper_tasks = ["BuyOne", "BuyTen", "BuyRegular", "BuyFigure", "BuyWeapon", "BuyNone"]

for group in ["Calibration", "HoldoutA", "HoldoutB"]:
    sub = binary[(binary["Group"] == group) & (binary["Task"].isin(paper_tasks))]
    if sub.empty:
        continue
    print(f"\n--- {group} ---")
    pivot = sub.pivot_table(index="Task", columns="Model", values="AUC").round(4)
    pivot = pivot.reindex([t for t in paper_tasks if t in pivot.index])
    print(pivot.to_string())

# ═══════════════════════════════════════════════════════════
# 3. Per-class AUC comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PER-CLASS AUC_OvR (test)")
print("=" * 80)

pc_frames = []
for name in MODELS:
    df = pd.read_csv(BASE / name / "perclass_metrics_long.csv")
    df = df[df["Split"] == SPLIT].copy()
    df["Model"] = name
    pc_frames.append(df)

perclass = pd.concat(pc_frames, ignore_index=True)

for group in ["Calibration", "HoldoutA", "HoldoutB"]:
    sub = perclass[perclass["Group"] == group]
    if sub.empty:
        continue
    print(f"\n--- {group} ---")
    pivot = sub.pivot_table(index="Class", columns="Model", values="AUC_OvR").round(4)
    print(pivot.to_string())

# ═══════════════════════════════════════════════════════════
# 4. Per-class F1 comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PER-CLASS F1 (test)")
print("=" * 80)

for group in ["Calibration", "HoldoutA", "HoldoutB"]:
    sub = perclass[perclass["Group"] == group]
    if sub.empty:
        continue
    print(f"\n--- {group} ---")
    pivot = sub.pivot_table(index="Class", columns="Model", values="F1").round(4)
    print(pivot.to_string())

# ═══════════════════════════════════════════════════════════
# 5. Summary: average across groups
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY: AVERAGE ACROSS ALL GROUPS (test)")
print("=" * 80)

summary_rows = []
for name in MODELS:
    df = pd.read_csv(BASE / name / "multiclass_metrics_long.csv")
    df = df[df["Split"] == SPLIT]
    row = {"Model": name}
    for metric in ["MacroOvR_AUC", "MacroAUPRC", "MacroF1", "MCC", "LogLoss", "Top1Acc", "Top2Acc"]:
        if metric in df.columns:
            row[metric] = round(df[metric].mean(), 4)
    
    # Average binary AUC
    bdf = pd.read_csv(BASE / name / "binary_metrics_long.csv")
    bdf = bdf[(bdf["Split"] == SPLIT) & (bdf["Task"].isin(paper_tasks))]
    row["AvgBinaryAUC"] = round(bdf["AUC"].mean(), 4) if not bdf.empty else np.nan
    
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# ═══════════════════════════════════════════════════════════
# 6. Save all tables
# ═══════════════════════════════════════════════════════════
out = BASE / "comparison_tables"
out.mkdir(exist_ok=True)
multi.to_csv(out / "multiclass_all.csv", index=False)
binary.to_csv(out / "binary_all.csv", index=False)
perclass.to_csv(out / "perclass_all.csv", index=False)
summary.to_csv(out / "summary.csv", index=False)
print(f"\nTables saved to {out}/")