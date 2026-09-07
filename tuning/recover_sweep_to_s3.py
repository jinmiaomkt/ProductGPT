#!/usr/bin/env python3
"""
recover_sweep_to_s3.py

Uploads a partial local sweep directory to S3, rebuilds the manifest
from whatever raw JSONL files exist, and regenerates the summary CSVs.
After running this, use --skip_done when re-running the sweep to pick up
only the pairs that didn't finish.

Usage:
    python3 recover_sweep_to_s3.py \
        --local_dir /home/ec2-user/outputs/c28_v1_20260403_004318 \
        --s3_bucket productgptbucket \
        --s3_prefix outputs/campaign28/c28_v1_20260403_004318

What it does:
    1. Scans raw/{lto28_name}/{uid}.jsonl for every file that exists locally
    2. Counts valid (non-error) lines in each file
    3. Uploads each raw file to S3
    4. Rebuilds manifest.json marking each pair as 'done' or 'partial'
    5. Rebuilds all_runs.csv and stop_stats.csv from the raw files
    6. Uploads config.json if present
    7. Prints a resume command you can copy-paste
"""

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def s3_put(s3, bucket: str, key: str, body: str):
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
    print(f"  -> s3://{bucket}/{key}  ({len(body):,} bytes)")


def load_jsonl(path: Path) -> List[Dict]:
    """Load all valid JSON lines from a file; skip error lines."""
    rows = []
    with open(path) as f:
        for line in f:
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


# ── Summary CSV helpers (copied from sweep script) ────────────────────────────

ALL_RUNS_COLS = [
    "uid", "lto28_name",
    "lto28_figA", "lto28_figB", "lto28_wep1", "lto28_wep2",
    "run", "seed",
    "stopped", "stop_step", "n_decisions",
    "decisions", "final_states", "calibrated", "ts",
]

STOP_STATS_COLS = [
    "uid", "lto28_name",
    "lto28_figA", "lto28_figB", "lto28_wep1", "lto28_wep2",
    "n_runs", "stop_rate",
    "mean_stop_step", "std_stop_step",
    "mean_n_decisions", "std_n_decisions",
    "dec_counts",
]

DECISION_LABELS = {
    1: "Buy1_Reg",  2: "Buy10_Reg",
    3: "Buy1_FigA", 4: "Buy10_FigA",
    5: "Buy1_FigB", 6: "Buy10_FigB",
    7: "Buy1_Wep",  8: "Buy10_Wep",
    9: "NotBuy",
}


def build_summary_rows(uid: str, lto28_name: str, lto28: List[int], runs: List[Dict]):
    """Build all_runs rows and one stop_stats row from a list of run dicts."""
    all_runs_rows = []
    for r in runs:
        all_runs_rows.append({
            "uid": uid,
            "lto28_name": lto28_name,
            "lto28_figA": lto28[0], "lto28_figB": lto28[1],
            "lto28_wep1": lto28[2], "lto28_wep2": lto28[3],
            "run": r.get("run", ""),
            "seed": r.get("seed", ""),
            "stopped": int(r.get("stopped", False)),
            "stop_step": r.get("stop_step", ""),
            "n_decisions": r.get("n_decisions", len(r.get("Campaign28_Decisions", []))),
            "decisions": json.dumps(r.get("Campaign28_Decisions", [])),
            "final_states": json.dumps(r.get("final_states", {})),
            "calibrated": int(r.get("calibrated", False)),
            "ts": r.get("ts", ""),
        })

    stopped_runs = [r for r in runs if r.get("stopped", False)]
    stop_steps = [r["stop_step"] for r in stopped_runs
                  if r.get("stop_step") is not None]
    all_n_decs = [r.get("n_decisions", len(r.get("Campaign28_Decisions", [])))
                  for r in runs]
    dec_counts: Counter = Counter()
    for r in runs:
        for d in r.get("Campaign28_Decisions", []):
            dec_counts[str(d)] += 1

    stop_stats_row = {
        "uid": uid,
        "lto28_name": lto28_name,
        "lto28_figA": lto28[0], "lto28_figB": lto28[1],
        "lto28_wep1": lto28[2], "lto28_wep2": lto28[3],
        "n_runs": len(runs),
        "stop_rate": len(stopped_runs) / len(runs) if runs else 0.0,
        "mean_stop_step": float(np.mean(stop_steps)) if stop_steps else "",
        "std_stop_step":  float(np.std(stop_steps))  if stop_steps else "",
        "mean_n_decisions": float(np.mean(all_n_decs)) if all_n_decs else "",
        "std_n_decisions":  float(np.std(all_n_decs))  if all_n_decs else "",
        "dec_counts": json.dumps(dict(dec_counts)),
    }

    return all_runs_rows, stop_stats_row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True,
                        help="Path to the partial sweep directory on disk, e.g. "
                             "/home/ec2-user/outputs/c28_v1_20260403_004318")
    parser.add_argument("--s3_bucket", required=True,
                        help="S3 bucket name, e.g. productgptbucket")
    parser.add_argument("--s3_prefix", required=True,
                        help="S3 key prefix for this sweep, e.g. "
                             "outputs/campaign28/c28_v1_20260403_004318")
    parser.add_argument("--lto28_configs", default=None,
                        help="Path to lto28_configs.json (to infer lto28 tokens from name). "
                             "If omitted, lto28 token columns will be empty in CSVs.")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    raw_dir   = local_dir / "raw"
    s3        = boto3.client("s3")
    prefix    = args.s3_prefix.strip("/")

    # ── Load lto28 config map ─────────────────────────────────────────────────
    lto28_map: Dict[str, List[int]] = {}
    if args.lto28_configs:
        with open(args.lto28_configs) as f:
            for cfg in json.load(f):
                lto28_map[cfg["name"]] = cfg["lto28"]

    # ── Upload config.json ────────────────────────────────────────────────────
    config_path = local_dir / "config.json"
    if config_path.exists():
        print("\n[CONFIG]")
        s3_put(s3, args.s3_bucket, f"{prefix}/config.json", config_path.read_text())

    # ── Scan raw directory ────────────────────────────────────────────────────
    if not raw_dir.exists():
        print(f"[ERROR] raw directory not found: {raw_dir}")
        return

    manifest_records: Dict[str, Any] = {}
    all_runs_buf  = io.StringIO()
    stop_stats_buf = io.StringIO()
    all_runs_writer   = csv.DictWriter(all_runs_buf,   fieldnames=ALL_RUNS_COLS,   extrasaction="ignore")
    stop_stats_writer = csv.DictWriter(stop_stats_buf, fieldnames=STOP_STATS_COLS, extrasaction="ignore")
    all_runs_writer.writeheader()
    stop_stats_writer.writeheader()

    total_files   = 0
    total_runs    = 0
    done_pairs    = 0
    partial_pairs = 0

    print("\n[RAW FILES]")
    for lto28_dir in sorted(raw_dir.iterdir()):
        if not lto28_dir.is_dir():
            continue
        lto28_name = lto28_dir.name
        lto28 = lto28_map.get(lto28_name, [0, 0, 0, 0])

        for uid_file in sorted(lto28_dir.glob("*.jsonl")):
            uid = uid_file.stem
            runs = load_jsonl(uid_file)
            n_valid = len(runs)
            total_files += 1
            total_runs  += n_valid

            # Upload raw file
            raw_body = uid_file.read_text()
            s3_put(s3, args.s3_bucket,
                   f"{prefix}/raw/{lto28_name}/{uid}.jsonl", raw_body)

            # Manifest entry
            mkey = f"{uid}|{lto28_name}"
            if n_valid > 0:
                # Determine if complete: check if we have a full complement of seeds
                seeds_seen = sorted(set(r.get("seed", r.get("run", -1)) for r in runs))
                status = "done" if n_valid >= 50 else "partial"
                manifest_records[mkey] = {
                    "uid": uid, "lto28_name": lto28_name,
                    "status": status,
                    "started_at": None, "finished_at": None,
                    "n_runs": n_valid, "error": None,
                    "recovered": True,
                }
                if status == "done":
                    done_pairs += 1
                else:
                    partial_pairs += 1
                print(f"     {uid}  {lto28_name}  {n_valid} runs  [{status}]")
            else:
                manifest_records[mkey] = {
                    "uid": uid, "lto28_name": lto28_name,
                    "status": "error", "n_runs": 0,
                    "error": "no valid lines found", "recovered": True,
                }
                print(f"     {uid}  {lto28_name}  0 valid lines  [error]")

            # Summary rows
            if runs:
                ar_rows, ss_row = build_summary_rows(uid, lto28_name, lto28, runs)
                for row in ar_rows:
                    all_runs_writer.writerow(row)
                stop_stats_writer.writerow(ss_row)

    # ── Upload manifest ───────────────────────────────────────────────────────
    print("\n[MANIFEST]")
    s3_put(s3, args.s3_bucket, f"{prefix}/manifest.json",
           json.dumps(manifest_records, indent=2))

    # ── Upload summary CSVs ───────────────────────────────────────────────────
    print("\n[SUMMARY CSVs]")
    s3_put(s3, args.s3_bucket, f"{prefix}/summary/all_runs.csv",
           all_runs_buf.getvalue())
    s3_put(s3, args.s3_bucket, f"{prefix}/summary/stop_stats.csv",
           stop_stats_buf.getvalue())

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"""
{'='*60}
[RECOVERY COMPLETE]
  Files uploaded   : {total_files}
  Total runs       : {total_runs}
  Done pairs       : {done_pairs}
  Partial pairs    : {partial_pairs}
  S3 location      : s3://{args.s3_bucket}/{prefix}/
{'='*60}

To resume the sweep (skipping done pairs), run:

  MODEL_STEM="featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0"
  LOCAL_CKPT="/tmp/FullProductGPT_${{MODEL_STEM}}.pt"
  LOCAL_CAL="/tmp/calibrator_${{MODEL_STEM}}.pt"

  python3 run_campaign28_sweep.py \\
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --ckpt "${{LOCAL_CKPT}}" \\
    --feat_xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \\
    --lto28_configs lto28_configs.json \\
    --sweep_name c28_v1_resumed \\
    --s3_bucket {args.s3_bucket} \\
    --s3_prefix {args.s3_prefix} \\
    --n_seeds 50 --seed_base 42 \\
    --calibrator_ckpt "${{LOCAL_CAL}}" \\
    --calibrator_type platt \\
    --skip_done \\
    --quiet

NOTE: --skip_done reads the manifest from S3 and skips any pair
      marked 'done' (50 runs). Partial pairs will be re-run from
      scratch (seeds 42..91). If you want to top them up instead
      of re-run, set --seed_base to the next unseen seed.
""")


if __name__ == "__main__":
    main()