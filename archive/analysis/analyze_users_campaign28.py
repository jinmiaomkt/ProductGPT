#!/usr/bin/env python3
"""
analyze_users_campaign28.py

User-level analysis of Campaign 28 sweep results.

Reads raw JSONL files from either a local directory or S3, then produces:
  1. user_avg_across_configs.csv     — per-user averages across all lto28 configs
  2. user_sensitivity_to_lto28.csv  — how much each user's behaviour varies by config
  3. user_wide_by_lto28.csv         — wide table: one row per user, cols per config
  4. user_decision_shares.csv       — per-user decision-type distribution
  5. user_avg_clustered.csv         — users grouped into behavioural archetypes (k-means)

Usage
─────
# From local raw directory (after aws s3 sync):
python3 analyze_users_campaign28.py \\
    --raw_dir /tmp/c28_raw \\
    --out_dir /tmp/c28_analysis

# Directly from S3:
python3 analyze_users_campaign28.py \\
    --s3_bucket productgptbucket \\
    --s3_prefix outputs/campaign28/c28_v1_20260403_004318/raw \\
    --out_dir /tmp/c28_analysis

# Upload outputs to S3 as well:
python3 analyze_users_campaign28.py \\
    --s3_bucket productgptbucket \\
    --s3_prefix outputs/campaign28/c28_v1_20260403_004318/raw \\
    --out_dir /tmp/c28_analysis \\
    --upload_s3_prefix outputs/campaign28/c28_v1_20260403_004318/analysis

# Control clustering:
python3 analyze_users_campaign28.py \\
    --raw_dir /tmp/c28_raw \\
    --out_dir /tmp/c28_analysis \\
    --n_clusters 5
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DECISION_LABELS = {
    1: "Buy1_Reg",  2: "Buy10_Reg",
    3: "Buy1_FigA", 4: "Buy10_FigA",
    5: "Buy1_FigB", 6: "Buy10_FigB",
    7: "Buy1_Wep",  8: "Buy10_Wep",
    9: "NotBuy",
}

BANNER_MAP = {
    1: "regular", 2: "regular",
    3: "figA",    4: "figA",
    5: "figB",    6: "figB",
    7: "weapon",  8: "weapon",
    9: "notbuy",
}

BANNERS = ["regular", "figA", "figB", "weapon", "notbuy"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_lines(text: str) -> List[Dict]:
    """Parse valid (non-error) lines from a JSONL string."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "error" not in obj:
                rows.append(obj)
        except json.JSONDecodeError:
            pass
    return rows


def load_from_local(raw_dir: Path) -> List[Dict]:
    """Load all raw JSONL files from a local directory tree."""
    records = []
    lto28_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    print(f"[LOAD] {len(lto28_dirs)} lto28 directories in {raw_dir}")
    for lto28_dir in lto28_dirs:
        uid_files = sorted(lto28_dir.glob("*.jsonl"))
        for uid_file in uid_files:
            runs = load_jsonl_lines(uid_file.read_text())
            records.append({
                "lto28_name": lto28_dir.name,
                "uid":        uid_file.stem,
                "runs":       runs,
            })
    print(f"[LOAD] {len(records)} (uid, lto28) pairs loaded.")
    return records


def load_from_s3(bucket: str, prefix: str) -> List[Dict]:
    """Load all raw JSONL files from S3."""
    import boto3
    s3 = boto3.client("s3")
    prefix = prefix.rstrip("/") + "/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    records = []
    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".jsonl"):
                keys.append(key)

    print(f"[LOAD] {len(keys)} JSONL files found in s3://{bucket}/{prefix}")
    for key in sorted(keys):
        parts = key[len(prefix):].split("/")
        if len(parts) != 2:
            continue
        lto28_name, filename = parts
        uid = filename.replace(".jsonl", "")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        runs = load_jsonl_lines(body)
        records.append({
            "lto28_name": lto28_name,
            "uid":        uid,
            "runs":       runs,
        })
    print(f"[LOAD] {len(records)} (uid, lto28) pairs loaded.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(record: Dict) -> Optional[Dict]:
    """Extract per-(uid, lto28) features from a list of seed runs."""
    uid       = record["uid"]
    lto28     = record["lto28_name"]
    runs      = record["runs"]

    if not runs:
        return None

    all_decs   = [d for r in runs for d in r.get("Campaign28_Decisions", [])]
    total      = len(all_decs)
    stop_steps = [r["stop_step"] for r in runs
                  if r.get("stopped") and r.get("stop_step") is not None]
    n_decs     = [r.get("n_decisions", len(r.get("Campaign28_Decisions", [])))
                  for r in runs]

    row = {
        "uid":              uid,
        "lto28":            lto28,
        "n_seeds":          len(runs),
        "stop_rate":        sum(r.get("stopped", False) for r in runs) / len(runs),
        "mean_stop_step":   float(np.mean(stop_steps)) if stop_steps else np.nan,
        "std_stop_step":    float(np.std(stop_steps))  if stop_steps else np.nan,
        "mean_n_decisions": float(np.mean(n_decs)),
        "std_n_decisions":  float(np.std(n_decs)),
        "total_decisions":  total,
    }

    # Banner-level shares
    banner_counts = Counter(BANNER_MAP.get(d, "?") for d in all_decs)
    for b in BANNERS:
        row[f"share_{b}"] = banner_counts.get(b, 0) / total if total else 0.0

    # Decision-level shares
    dec_counts = Counter(all_decs)
    for did, label in DECISION_LABELS.items():
        row[f"share_{label}"] = dec_counts.get(did, 0) / total if total else 0.0

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tables
# ─────────────────────────────────────────────────────────────────────────────

def build_user_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user averages across all lto28 configs."""
    agg_cols = {
        "n_configs":        ("lto28",             "nunique"),
        "mean_stop_rate":   ("stop_rate",          "mean"),
        "mean_stop_step":   ("mean_stop_step",     "mean"),
        "std_stop_step":    ("std_stop_step",      "mean"),
        "mean_n_decisions": ("mean_n_decisions",   "mean"),
    }
    for b in BANNERS:
        agg_cols[f"share_{b}"] = (f"share_{b}", "mean")
    for did, label in DECISION_LABELS.items():
        agg_cols[f"share_{label}"] = (f"share_{label}", "mean")

    return (df.groupby("uid")
              .agg(**agg_cols)
              .reset_index()
              .sort_values("mean_stop_step"))


def build_user_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    How much does each user's stop_step vary across lto28 configs?
    High std = user is sensitive to which banners are offered.
    Low std  = user behaves similarly regardless of banner lineup.
    """
    return (df.groupby("uid")
              .agg(
                  mean_stop_step      = ("mean_stop_step", "mean"),
                  sensitivity_std     = ("mean_stop_step", "std"),
                  mean_n_decisions    = ("mean_n_decisions", "mean"),
                  mean_share_figA     = ("share_figA",    "mean"),
                  mean_share_figB     = ("share_figB",    "mean"),
                  mean_share_weapon   = ("share_weapon",  "mean"),
                  mean_share_regular  = ("share_regular", "mean"),
                  mean_share_notbuy   = ("share_notbuy",  "mean"),
              )
              .reset_index()
              .sort_values("sensitivity_std", ascending=False))


def build_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table: one row per user, separate columns for each lto28 config.
    Useful for direct cross-config comparison per user.
    """
    pivot_stop   = df.pivot_table(index="uid", columns="lto28",
                                   values="mean_stop_step",  aggfunc="mean")
    pivot_ndecs  = df.pivot_table(index="uid", columns="lto28",
                                   values="mean_n_decisions", aggfunc="mean")
    pivot_figA   = df.pivot_table(index="uid", columns="lto28",
                                   values="share_figA",       aggfunc="mean")
    pivot_figB   = df.pivot_table(index="uid", columns="lto28",
                                   values="share_figB",       aggfunc="mean")
    pivot_weapon = df.pivot_table(index="uid", columns="lto28",
                                   values="share_weapon",     aggfunc="mean")
    pivot_stop.columns   = [f"stop_step__{c}"   for c in pivot_stop.columns]
    pivot_ndecs.columns  = [f"n_decs__{c}"      for c in pivot_ndecs.columns]
    pivot_figA.columns   = [f"figA__{c}"        for c in pivot_figA.columns]
    pivot_figB.columns   = [f"figB__{c}"        for c in pivot_figB.columns]
    pivot_weapon.columns = [f"weapon__{c}"      for c in pivot_weapon.columns]

    wide = pd.concat([pivot_stop, pivot_ndecs, pivot_figA,
                      pivot_figB, pivot_weapon], axis=1).reset_index()
    # Add range (max - min stop_step across configs) as a sensitivity proxy
    stop_cols = [c for c in wide.columns if c.startswith("stop_step__")]
    if stop_cols:
        wide["stop_step_range"] = wide[stop_cols].max(axis=1) - wide[stop_cols].min(axis=1)
    return wide.sort_values("stop_step_range", ascending=False)


def build_decision_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user decision-type distribution, averaged across configs."""
    dec_cols = [f"share_{label}" for label in DECISION_LABELS.values()]
    return (df.groupby("uid")[dec_cols]
              .mean()
              .reset_index()
              .sort_values("share_NotBuy", ascending=False))


def cluster_users(user_avg: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """K-means clustering of users by banner preference + stop step."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    except ImportError:
        print("[CLUSTER] sklearn not available — skipping clustering.")
        return user_avg

    feat_cols = ["share_figA", "share_figB", "share_weapon",
                 "share_regular", "share_notbuy", "mean_stop_step"]
    available = [c for c in feat_cols if c in user_avg.columns]
    X = user_avg[available].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_avg = user_avg.copy()
    user_avg["cluster"] = km.fit_predict(X_scaled)

    print(f"\n=== USER CLUSTERS (k={n_clusters}) ===")
    summary = user_avg.groupby("cluster")[available + ["mean_stop_step"]].mean()
    summary["n_users"] = user_avg.groupby("cluster").size()
    print(summary.to_string(float_format="%.3f"))

    return user_avg


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, out_dir: Path, filename: str,
         s3_client=None, s3_bucket: str = "", s3_prefix: str = "") -> Path:
    """Save a DataFrame to CSV locally and optionally upload to S3."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df)} rows)")

    if s3_client and s3_bucket and s3_prefix:
        key = f"{s3_prefix.rstrip('/')}/{filename}"
        body = path.read_bytes()
        s3_client.put_object(Bucket=s3_bucket, Key=key, Body=body)
        print(f"  -> s3://{s3_bucket}/{key}")

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="User-level Campaign 28 analysis")

    # Input — choose one
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--raw_dir",
                             help="Local path to raw/ directory from the sweep.")
    input_group.add_argument("--s3_bucket",
                             help="S3 bucket for input (requires --s3_prefix).")

    parser.add_argument("--s3_prefix", default="",
                        help="S3 key prefix pointing to the raw/ directory, e.g. "
                             "outputs/campaign28/c28_v1_20260403_004318/raw")

    # Output
    parser.add_argument("--out_dir", default="/tmp/c28_analysis",
                        help="Local directory for output CSVs.")
    parser.add_argument("--upload_s3_prefix", default="",
                        help="If set, also upload output CSVs to this S3 prefix.")

    # Options
    parser.add_argument("--n_clusters", type=int, default=4,
                        help="Number of k-means clusters for user segmentation.")
    parser.add_argument("--min_seeds", type=int, default=1,
                        help="Minimum number of valid seeds required to include a pair.")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # ── S3 client for uploads ─────────────────────────────────────────────────
    upload_s3 = None
    if args.upload_s3_prefix:
        import boto3
        upload_s3 = boto3.client("s3")
        upload_bucket = args.s3_bucket or ""
        # If input is local, s3_bucket may not be set for uploads
        if not upload_bucket:
            raise ValueError("--upload_s3_prefix requires --s3_bucket to be set.")
    else:
        upload_bucket = ""

    s3_upload_kwargs = dict(
        s3_client=upload_s3,
        s3_bucket=upload_bucket,
        s3_prefix=args.upload_s3_prefix,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.raw_dir:
        records = load_from_local(Path(args.raw_dir))
    else:
        records = load_from_s3(args.s3_bucket, args.s3_prefix)

    # ── Feature extraction ────────────────────────────────────────────────────
    feat_rows = []
    skipped   = 0
    for rec in records:
        if len(rec["runs"]) < args.min_seeds:
            skipped += 1
            continue
        row = extract_features(rec)
        if row:
            feat_rows.append(row)

    if skipped:
        print(f"[INFO] Skipped {skipped} pairs with fewer than {args.min_seeds} valid seeds.")

    df = pd.DataFrame(feat_rows)
    print(f"\n[BUILD] Feature DataFrame: {df.shape[0]} rows, "
          f"{df['uid'].nunique()} users, {df['lto28'].nunique()} configs")

    # ── Build and save tables ─────────────────────────────────────────────────
    print("\n[ANALYSIS] Building tables...")

    user_avg = build_user_avg(df)
    print(f"\n=== USER AVERAGES (top 20 by stop_step) ===")
    print(user_avg.head(20).to_string(index=False, float_format="%.3f"))
    save(user_avg, out_dir, "user_avg_across_configs.csv", **s3_upload_kwargs)

    sensitivity = build_user_sensitivity(df)
    print(f"\n=== SENSITIVITY TO lto28 (top 20, most sensitive first) ===")
    print(sensitivity.head(20).to_string(index=False, float_format="%.3f"))
    save(sensitivity, out_dir, "user_sensitivity_to_lto28.csv", **s3_upload_kwargs)

    wide = build_wide_table(df)
    print(f"\n=== WIDE TABLE (top 10 most config-sensitive users) ===")
    stop_cols = [c for c in wide.columns if c.startswith("stop_step__")]
    print(wide[["uid", "stop_step_range"] + stop_cols].head(10)
            .to_string(index=False, float_format="%.1f"))
    save(wide, out_dir, "user_wide_by_lto28.csv", **s3_upload_kwargs)

    dec_shares = build_decision_shares(df)
    print(f"\n=== TOP 20 USERS BY NotBuy SHARE ===")
    print(dec_shares.head(20).to_string(index=False, float_format="%.3f"))
    save(dec_shares, out_dir, "user_decision_shares.csv", **s3_upload_kwargs)

    # Raw per-(uid, lto28) features — useful for custom analysis
    save(df, out_dir, "features_per_uid_lto28.csv", **s3_upload_kwargs)

    # ── Clustering ────────────────────────────────────────────────────────────
    user_avg_clustered = cluster_users(user_avg, args.n_clusters)
    if "cluster" in user_avg_clustered.columns:
        save(user_avg_clustered, out_dir, "user_avg_clustered.csv", **s3_upload_kwargs)

    print(f"\n[DONE] All outputs written to {out_dir}/")


if __name__ == "__main__":
    main()