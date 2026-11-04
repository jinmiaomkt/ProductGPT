"""
analysis_campaign28.py

Starter analysis script for Campaign 28 sweep results.

Covers:
  1. Load summary CSVs from one or more sweeps OR directly from inference JSONL
  2. Per-user decision distribution across lto28 configs
  3. Stop rate and stop step analysis
  4. Cross-campaign comparison: c28 trajectory vs c1-c27 (from AggregateInput history)

Usage (from sweep directory):
  python3 analysis_campaign28.py \\
    --sweep_dir /home/ec2-user/outputs/c28_sweep \\
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --out_dir /home/ec2-user/analysis/c28

Usage (directly from JSONL):
  python3 analysis_campaign28.py \\
    --jsonl /home/ec2-user/outputs/campaign28_calibrated.jsonl \\
    --lto28_name "30_0_54_51" \\
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --out_dir /home/ec2-user/analysis/c28
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Decision metadata for readable labels
DECISION_LABELS = {
    1: "Buy1_Reg",   2: "Buy10_Reg",
    3: "Buy1_FigA",  4: "Buy10_FigA",
    5: "Buy1_FigB",  6: "Buy10_FigB",
    7: "Buy1_Wep",   8: "Buy10_Wep",
    9: "NotBuy",
}
AI_RATE = 15


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load sweep results
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep(sweep_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load all_runs.csv, stop_stats.csv, and config.json from a sweep directory.
    """
    p = Path(sweep_dir)
    runs_df  = pd.read_csv(p / "summary" / "all_runs.csv")
    stats_df = pd.read_csv(p / "summary" / "stop_stats.csv")
    config_path = p / "config.json"
    config = json.load(open(config_path)) if config_path.exists() else {}

    # Parse JSON columns
    runs_df["decisions_list"] = runs_df["decisions"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    stats_df["dec_counts_dict"] = stats_df["dec_counts"].apply(
        lambda x: {int(k): v for k, v in json.loads(x).items()} if isinstance(x, str) else {}
    )

    print(f"[LOAD] {len(runs_df)} runs, {runs_df['uid'].nunique()} users, "
          f"{runs_df['lto28_name'].nunique()} lto28 configs")
    return runs_df, stats_df, config


def load_jsonl(jsonl_path: str, lto28_name: str = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load directly from inference JSONL and build the same DataFrames
    that load_sweep would return.
    """
    records = []
    with open(jsonl_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    if lto28_name is None:
        lto28_name = Path(jsonl_path).stem

    # --- Build all_runs (one row per uid × run) ---
    run_rows = []
    for rec in records:
        uid = rec.get("uid", "")
        run_id = rec.get("run", 0)
        seed = rec.get("run_seed", run_id)
        decisions = rec.get("Campaign28_Decisions", [])
        stopped = rec.get("stopped", False)
        stop_step = rec.get("stop_step")

        run_rows.append({
            "uid": uid,
            "lto28_name": lto28_name,
            "run": run_id,
            "seed": seed,
            "stopped": stopped,
            "stop_step": stop_step if stop_step is not None else len(decisions),
            "n_decisions": len(decisions),
            "decisions": json.dumps(decisions),
            "decisions_list": decisions,
            "calibrated": rec.get("calibrated", False),
        })

    runs_df = pd.DataFrame(run_rows)

    # --- Build stop_stats (one row per uid × lto28_name) ---
    stats_rows = []
    for (uid, lto_name), grp in runs_df.groupby(["uid", "lto28_name"]):
        all_decs = []
        for dlist in grp["decisions_list"]:
            all_decs.extend(dlist)
        dec_counts = dict(Counter(all_decs))

        stats_rows.append({
            "uid": uid,
            "lto28_name": lto_name,
            "n_runs": len(grp),
            "stop_rate": grp["stopped"].mean(),
            "mean_stop_step": grp["stop_step"].mean(),
            "mean_n_decisions": grp["n_decisions"].mean(),
            "dec_counts": json.dumps(dec_counts),
            "dec_counts_dict": dec_counts,
        })

    stats_df = pd.DataFrame(stats_rows)

    config = {
        "lto28_name": lto28_name,
        "source_jsonl": str(jsonl_path),
        "n_users": int(runs_df["uid"].nunique()),
    }

    print(f"[LOAD-JSONL] {len(runs_df)} runs, {runs_df['uid'].nunique()} users, "
          f"lto28_name={lto28_name}")
    return runs_df, stats_df, config


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stop rate analysis
# ─────────────────────────────────────────────────────────────────────────────

def stop_rate_by_config(stats_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stats_df
        .groupby("lto28_name")
        .agg(
            n_users=("uid", "nunique"),
            mean_stop_rate=("stop_rate", "mean"),
            std_stop_rate=("stop_rate", "std"),
            mean_stop_step=("mean_stop_step", "mean"),
            mean_n_decisions=("mean_n_decisions", "mean"),
        )
        .reset_index()
        .sort_values("mean_stop_rate", ascending=False)
    )


def stop_rate_by_user(stats_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stats_df
        .groupby("uid")
        .agg(
            n_configs=("lto28_name", "nunique"),
            mean_stop_rate=("stop_rate", "mean"),
            mean_stop_step=("mean_stop_step", "mean"),
            mean_n_decisions=("mean_n_decisions", "mean"),
        )
        .reset_index()
        .sort_values("mean_stop_rate", ascending=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decision distribution analysis
# ─────────────────────────────────────────────────────────────────────────────

def decision_shares(runs_df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    records = []
    for key, grp in runs_df.groupby(group_by):
        all_decs = []
        for dlist in grp["decisions_list"]:
            all_decs.extend(dlist)
        total = len(all_decs)
        if total == 0:
            continue
        counts = Counter(all_decs)
        row = dict(zip(group_by, key if isinstance(key, tuple) else [key]))
        row["total_decisions"] = total
        for did, label in DECISION_LABELS.items():
            row[f"share_{label}"] = counts.get(did, 0) / total
        records.append(row)
    return pd.DataFrame(records)


def banner_spend(runs_df: pd.DataFrame) -> pd.DataFrame:
    banner_map = {
        1: "regular", 2: "regular",
        3: "figure_a", 4: "figure_a",
        5: "figure_b", 6: "figure_b",
        7: "weapon", 8: "weapon",
        9: "not_buy",
    }
    records = []
    for (uid, lto28_name), grp in runs_df.groupby(["uid", "lto28_name"]):
        all_decs = []
        for dlist in grp["decisions_list"]:
            all_decs.extend(dlist)
        total = len(all_decs)
        if total == 0:
            continue
        banner_counts = Counter(banner_map.get(d, "unknown") for d in all_decs)
        row = {"uid": uid, "lto28_name": lto28_name, "total_decisions": total}
        for b in ["regular", "figure_a", "figure_b", "weapon", "not_buy"]:
            row[f"share_{b}"] = banner_counts.get(b, 0) / total
        records.append(row)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-campaign comparison: c28 vs c1-c27 history
# ─────────────────────────────────────────────────────────────────────────────

def extract_historical_decisions(row: dict, ai_rate: int = AI_RATE) -> list[int]:
    from infer_new_campaign_calibrated import parse_int_sequence
    tokens = parse_int_sequence(row["AggregateInput"], na_to=0)
    n_blocks = len(tokens) // ai_rate
    return [int(tokens[i * ai_rate + 14]) for i in range(n_blocks)]


def build_historical_decision_df(data_path: str) -> pd.DataFrame:
    import gzip
    from infer_new_campaign_calibrated import _iter_rows
    records = []
    for row in _iter_rows(data_path):
        uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")
        decs = extract_historical_decisions(row)
        total = len(decs)
        if total == 0:
            continue
        counts = Counter(decs)
        rec = {"uid": uid, "n_historical_decisions": total}
        for did, label in DECISION_LABELS.items():
            rec[f"hist_share_{label}"] = counts.get(did, 0) / total
        records.append(rec)
    return pd.DataFrame(records)


def compare_c28_vs_history(
    runs_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    lto28_filter: str = None,
) -> pd.DataFrame:
    if lto28_filter:
        c28 = runs_df[runs_df["lto28_name"] == lto28_filter].copy()
    else:
        c28 = runs_df.copy()

    c28_shares = decision_shares(c28, group_by=["uid"])
    merged = c28_shares.merge(hist_df, on="uid", how="inner")

    for did, label in DECISION_LABELS.items():
        c28_col  = f"share_{label}"
        hist_col = f"hist_share_{label}"
        if c28_col in merged.columns and hist_col in merged.columns:
            merged[f"delta_{label}"] = merged[c28_col] - merged[hist_col]

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 5. Seed variance
# ─────────────────────────────────────────────────────────────────────────────

def seed_variance_by_user_config(runs_df: pd.DataFrame) -> pd.DataFrame:
    df = (
        runs_df
        .groupby(["uid", "lto28_name"])
        .agg(
            n_seeds=("seed", "count"),
            stop_rate=("stopped", "mean"),
            mean_stop_step=("stop_step", "mean"),
            std_stop_step=("stop_step", "std"),
            mean_n_decisions=("n_decisions", "mean"),
            std_n_decisions=("n_decisions", "std"),
        )
        .reset_index()
    )
    df["cv_stop_step"] = df["std_stop_step"] / (df["mean_stop_step"] + 1e-8)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    # Two input modes: sweep_dir OR jsonl (mutually exclusive)
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--sweep_dir",
                           help="Sweep directory containing summary/all_runs.csv etc.")
    input_grp.add_argument("--jsonl",
                           help="Inference JSONL file (direct load, no sweep dir needed)")

    parser.add_argument("--lto28_name", default=None,
                        help="Label for lto28 config (used with --jsonl; auto-derived if not set)")
    parser.add_argument("--data", default=None,
                        help="Original data file (for historical comparison)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lto28", default=None,
                        help="Restrict c28-vs-history comparison to this lto28_name")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep — from directory or directly from JSONL
    if args.sweep_dir:
        runs_df, stats_df, config = load_sweep(args.sweep_dir)
    else:
        runs_df, stats_df, config = load_jsonl(args.jsonl, lto28_name=args.lto28_name)

    # 1. Stop rate tables
    stop_by_config = stop_rate_by_config(stats_df)
    stop_by_config.to_csv(out_dir / "stop_rate_by_config.csv", index=False)
    print("\n[STOP RATE BY CONFIG]")
    print(stop_by_config.to_string(index=False))

    stop_by_user = stop_rate_by_user(stats_df)
    stop_by_user.to_csv(out_dir / "stop_rate_by_user.csv", index=False)

    # 2. Decision distribution
    dec_by_user_config = decision_shares(runs_df, group_by=["uid", "lto28_name"])
    dec_by_user_config.to_csv(out_dir / "decision_shares_by_user_config.csv", index=False)

    banner_df = banner_spend(runs_df)
    banner_df.to_csv(out_dir / "banner_spend.csv", index=False)

    # 3. Seed variance
    variance_df = seed_variance_by_user_config(runs_df)
    variance_df.to_csv(out_dir / "seed_variance.csv", index=False)

    # 4. Cross-campaign comparison (requires --data)
    if args.data:
        print("\n[BUILDING HISTORICAL BASELINE] This may take a moment...")
        hist_df = build_historical_decision_df(args.data)
        hist_df.to_csv(out_dir / "historical_decision_shares.csv", index=False)

        compare_df = compare_c28_vs_history(runs_df, hist_df, lto28_filter=args.lto28)
        compare_df.to_csv(out_dir / "c28_vs_history_deltas.csv", index=False)

        delta_cols = [c for c in compare_df.columns if c.startswith("delta_")]
        print("\n[C28 vs HISTORY — mean delta per decision type]")
        print(compare_df[delta_cols].mean().sort_values(ascending=False).to_string())

    print(f"\n[ANALYSIS] All outputs written to {out_dir}")


if __name__ == "__main__":
    main()