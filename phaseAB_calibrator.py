#!/usr/bin/env python3
"""
match_calibrators_to_ray.py

Ranks Ray Tune trials by val_nll, then cross-references with
calibrator files available in S3 to find the best-performing
checkpoint that has a calibrator.

Usage:
  python3 match_calibrators_to_ray.py
"""

from __future__ import annotations
import re, json
from pathlib import Path
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
EXP_DIR    = Path("/home/ec2-user/ProductGPT/ray_results/ProductGPT_RayTune")
METRIC     = "val_nll"
S3_CKPT_PREFIX = "s3://productgptbucket/FullProductGPT/performer/FeatureBased/checkpoints"

# Paste the `aws s3 ls --recursive | grep calibrator` output here,
# or point to a file containing it
CALIBRATOR_LISTING = """
2026-03-17 11:03:44       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel64_ff128_N4_heads4_lr0.0006212999742344905_w1_fold0.pt
2026-03-18 20:06:08       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel64_ff128_N6_heads4_lr0.00044222308716434253_w1_fold0.pt
2026-03-13 07:40:53       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel64_ff192_N4_heads4_lr0.00026406604683149813_w1_fold0.pt
2026-03-13 14:32:05       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel64_ff192_N4_heads4_lr0.0006129264018638836_w1_fold0.pt
2026-03-13 04:13:50       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel64_ff192_N5_heads4_lr0.00010590636735547144_w1_fold0.pt
2026-03-19 11:19:27       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0003476727013674719_w1_fold0.pt
2026-03-19 04:52:10       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0003520972290332634_w1_fold0.pt
2026-03-15 00:30:53       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0003702758431708816_w1_fold0.pt
2026-03-14 19:31:54       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0004077987306785015_w1_fold0.pt
2026-03-16 05:22:06       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0004262215672980239_w1_fold0.pt
2026-03-15 17:43:37       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0005127992419246565_w1_fold0.pt
2026-03-14 16:04:59       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0006206162966215636_w1_fold0.pt
2026-03-15 13:14:43       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N4_heads4_lr0.0009902167184652566_w1_fold0.pt
2026-03-16 21:25:01       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures16_dmodel96_ff192_N5_heads4_lr0.00014929906131598054_w1_fold0.pt
2026-03-08 02:36:36       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures32_dmodel64_ff192_N2_heads4_lr0.0004758087093016644_w1_fold0.pt
2026-03-18 03:04:17       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures32_dmodel64_ff192_N4_heads4_lr0.0008610994824274476_w1_fold0.pt
2026-03-13 00:48:06       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures32_dmodel64_ff192_N5_heads4_lr0.00012890928474222637_w1_fold0.pt
2026-03-12 21:54:33       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures32_dmodel64_ff192_N5_heads4_lr0.00014267311961672685_w1_fold0.pt
2026-03-16 11:27:50       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures32_dmodel96_ff192_N5_heads4_lr0.0005517835239624701_w1_fold0.pt
2026-03-18 09:21:31       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures48_dmodel64_ff128_N5_heads4_lr0.0007118131405499274_w1_fold0.pt
2026-03-10 20:57:23       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures48_dmodel64_ff192_N3_heads2_lr0.0004906455656729149_w1_fold0.pt
2026-03-11 01:01:43       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures48_dmodel64_ff192_N3_heads2_lr0.0004908756772262734_w1_fold0.pt
2026-03-11 07:16:53       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures48_dmodel64_ff192_N3_heads2_lr0.0005869404496504987_w1_fold0.pt
2026-03-11 12:34:35       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures48_dmodel64_ff192_N4_heads2_lr0.0009940136300588104_w1_fold0.pt
2026-03-08 23:14:12       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel48_ff144_N2_heads2_lr0.0004621979999026454_w1_fold0.pt
2026-03-09 00:27:01       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel48_ff144_N3_heads2_lr0.0004531414274774716_w1_fold0.pt
2026-03-09 03:04:56       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel48_ff144_N3_heads2_lr0.0004575209164514576_w1_fold0.pt
2026-03-09 04:42:54       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel48_ff144_N3_heads2_lr0.0004811563187548085_w1_fold0.pt
2026-03-09 06:29:19       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel48_ff144_N3_heads2_lr0.0005627166937148845_w1_fold0.pt
2026-03-09 17:25:36       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.00041557556806354694_w1_fold0.pt
2026-03-09 18:34:28       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.00042241441839904645_w1_fold0.pt
2026-03-09 12:53:48       2642 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.00042459228768596723_w1_fold0.pt
2026-03-10 11:27:07       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.0004991254642799674_w1_fold0.pt
2026-03-12 07:59:29       2630 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt
2026-03-10 14:19:40       2636 FullProductGPT/performer/FeatureBased/checkpoints/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.0006730630810434509_w1_fold0.pt
""".strip()

CAL_RE = re.compile(
    r"performerfeatures(?P<nb>\d+)_dmodel(?P<dm>\d+)_ff(?P<ff>\d+)"
    r"_N(?P<N>\d+)_heads(?P<h>\d+)_lr(?P<lr>[\d.eE\-+]+)_w(?P<w>\d+)_fold(?P<fold>\d+)\.pt$"
)

# ── Parse calibrator listing ──────────────────────────────────────────────────
def parse_calibrators(listing: str) -> pd.DataFrame:
    rows = []
    for line in listing.splitlines():
        line = line.strip()
        if not line:
            continue
        # last token is the S3 key
        s3_key = line.split()[-1]
        fname  = Path(s3_key).name
        m = CAL_RE.search(fname)
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "cal_s3_key":  s3_key,
            "cal_fname":   fname,
            "nb_features": int(d["nb"]),
            "d_model":     int(d["dm"]),
            "d_ff":        int(d["ff"]),
            "N":           int(d["N"]),
            "num_heads":   int(d["h"]),
            "lr_cal":      float(d["lr"]),
            "weight":      int(d["w"]),
            "fold":        int(d["fold"]),
        })
    return pd.DataFrame(rows)

# ── Ray trial loading (mirrors rankA logic) ───────────────────────────────────
def load_params(trial_dir: Path) -> dict:
    p = trial_dir / "params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

def load_best_metric(trial_dir: Path, metric: str):
    for fname, loader in [("result.json", _from_result_json), ("progress.csv", _from_csv)]:
        path = trial_dir / fname
        if path.exists():
            best = loader(path, metric)
            if best is not None:
                return best
    return None

def _from_result_json(path: Path, metric: str):
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if metric not in df.columns:
        return None
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    return df.loc[df[metric].idxmin()].to_dict() if not df.empty else None

def _from_csv(path: Path, metric: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if metric not in df.columns or df.empty:
        return None
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    return df.loc[df[metric].idxmin()].to_dict() if not df.empty else None

def find_trial_dirs(exp_dir: Path):
    candidates = set()
    for pattern in ("params.json", "result.json", "progress.csv"):
        for p in exp_dir.rglob(pattern):
            candidates.add(p.parent)
    return sorted(candidates)

def extract_hps_from_cfg(cfg: dict) -> dict:
    """
    Map Ray config keys → checkpoint filename fields.
    dm_heads is typically a tuple (d_model, num_heads) stored as a list in JSON.
    """
    hps = {}

    dm_heads = cfg.get("dm_heads")
    if isinstance(dm_heads, (list, tuple)) and len(dm_heads) == 2:
        hps["d_model"]   = int(dm_heads[0])
        hps["num_heads"] = int(dm_heads[1])
    elif isinstance(dm_heads, str):
        # e.g. "(96, 4)"
        nums = re.findall(r"\d+", dm_heads)
        if len(nums) == 2:
            hps["d_model"], hps["num_heads"] = int(nums[0]), int(nums[1])

    if "nb_features" in cfg:
        hps["nb_features"] = int(cfg["nb_features"])
    if "N" in cfg:
        hps["N"] = int(cfg["N"])
    if "lr" in cfg:
        hps["lr_trial"] = float(cfg["lr"])
    if "dff_mult" in cfg and "d_model" in hps:
        hps["d_ff"] = int(cfg["dff_mult"] * hps["d_model"])

    return hps

# ── HP matching key ───────────────────────────────────────────────────────────
def hp_key(nb, dm, ff, N, heads):
    return (int(nb), int(dm), int(ff), int(N), int(heads))

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Parse calibrators
    cal_df = parse_calibrators(CALIBRATOR_LISTING)
    cal_df["_key"] = cal_df.apply(
        lambda r: hp_key(r.nb_features, r.d_model, r.d_ff, r.N, r.num_heads), axis=1
    )
    print(f"Parsed {len(cal_df)} calibrator files from S3 listing.")

    # 2. Load Ray trials
    trial_dirs = find_trial_dirs(EXP_DIR)
    print(f"Found {len(trial_dirs)} trial directories under {EXP_DIR}")

    trial_rows = []
    for td in trial_dirs:
        cfg  = load_params(td)
        best = load_best_metric(td, METRIC)
        if best is None:
            continue
        hps = extract_hps_from_cfg(cfg)
        row = {
            "trial_name": td.name,
            "trial_dir":  str(td),
            "val_nll":    float(best.get(METRIC, float("inf"))),
            "val_hit":    best.get("val_hit"),
            "val_auprc_macro": best.get("val_auprc_macro"),
            "epoch_at_best":   best.get("epoch"),
            **hps,
        }
        trial_rows.append(row)

    trials_df = (
        pd.DataFrame(trial_rows)
        .sort_values("val_nll", ascending=True)
        .reset_index(drop=True)
    )
    print(f"Loaded metrics for {len(trials_df)} trials.")

    # 3. Build trial HP key (need d_ff; if missing, try common values)
    def trial_key(r):
        if all(k in r and pd.notna(r[k]) for k in ("nb_features","d_model","d_ff","N","num_heads")):
            return hp_key(r.nb_features, r.d_model, r.d_ff, r.N, r.num_heads)
        return None

    trials_df["_key"] = trials_df.apply(trial_key, axis=1)

    # 4. Cross-reference: keep only trials that have a calibrator
    cal_keys = set(cal_df["_key"])

    # Group calibrators by HP key → pick the one with the closest lr
    cal_by_key = cal_df.groupby("_key")

    matched_rows = []
    for _, trial in trials_df.iterrows():
        k = trial["_key"]
        if k is None or k not in cal_keys:
            continue

        # Among calibrators with same HPs, pick closest lr
        group = cal_by_key.get_group(k).copy()
        lr_trial = trial.get("lr_trial", None)
        if lr_trial is not None:
            group["lr_diff"] = (group["lr_cal"] - lr_trial).abs()
            best_cal = group.loc[group["lr_diff"].idxmin()]
        else:
            best_cal = group.iloc[0]

        ckpt_stem = best_cal["cal_fname"].replace("calibrator_", "FullProductGPT_")
        ckpt_s3   = f"{S3_CKPT_PREFIX}/{ckpt_stem}"
        cal_s3    = f"s3://productgptbucket/{best_cal['cal_s3_key']}"

        matched_rows.append({
            **trial.to_dict(),
            "cal_s3":   cal_s3,
            "ckpt_s3":  ckpt_s3,
            "lr_cal":   float(best_cal["lr_cal"]),
            "lr_diff":  abs((lr_trial or 0) - float(best_cal["lr_cal"])),
        })

    matched_df = (
        pd.DataFrame(matched_rows)
        .sort_values("val_nll", ascending=True)
        .reset_index(drop=True)
    )

    show_cols = [c for c in [
        "val_nll", "val_hit", "val_auprc_macro", "epoch_at_best",
        "nb_features", "d_model", "d_ff", "N", "num_heads",
        "lr_trial", "lr_cal", "lr_diff",
        "cal_s3", "ckpt_s3",
    ] if c in matched_df.columns]

    print(f"\n===== TOP 10 TRIALS WITH CALIBRATOR (ranked by {METRIC}) =====")
    print(matched_df[show_cols].head(10).to_string(index=False))

    if matched_df.empty:
        print("\n[WARN] No matches found. Check that dm_heads parses correctly from params.json.")
        print("       Run this to inspect a sample trial config:")
        print(f"       cat '{trial_dirs[0]}/params.json'")
        return

    best = matched_df.iloc[0]
    print("\n===== BEST CALIBRATED CHECKPOINT =====")
    print(f"  val_nll       : {best['val_nll']}")
    print(f"  HPs           : nb={best.get('nb_features')} dm={best.get('d_model')} "
          f"ff={best.get('d_ff')} N={best.get('N')} heads={best.get('num_heads')}")
    print(f"  Checkpoint S3 : {best['ckpt_s3']}")
    print(f"  Calibrator S3 : {best['cal_s3']}")
    print(f"  lr (trial/cal): {best.get('lr_trial')} / {best['lr_cal']}  (diff={best['lr_diff']:.2e})")

    print("\n===== SUGGESTED EVAL COMMAND =====")
    print(f"""
aws s3 cp '{best['ckpt_s3']}' /tmp/best_ckpt.pt
aws s3 cp '{best['cal_s3']}' /tmp/$(basename '{best['cal_s3']}')

python3 predict_productgpt_and_eval.py \\
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
  --ckpt /tmp/best_ckpt.pt \\
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \\
  --s3 s3://productgptbucket/evals/best_calibrated_$(date +%F_%H%M%S)/ \\
  --calibration calibrator \\
  --fold-id 0
""")

    out_csv = Path("/tmp/calibrated_ranked_trials.csv")
    matched_df[show_cols].to_csv(out_csv, index=False)
    print(f"Full ranking saved to: {out_csv}")

if __name__ == "__main__":
    main()