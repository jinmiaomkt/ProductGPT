#!/usr/bin/env python3
"""
run_campaign28_sweep.py

Disciplined sweep orchestrator for Campaign 28 inference.

Outer loop : lto28 configs  (loaded from JSON)
Middle loop: users          (all or --n_users)
Inner loop : seeds          (--n_seeds, starting at --seed_base)

The model and calibrator are loaded ONCE and reused across all (user, lto28) pairs,
so the sweep is fast even with many configurations.

Output layout
─────────────
{out_root}/{sweep_name}_{YYYYMMDD_HHMMSS}/
  config.json                        ← full experiment config snapshot
  manifest.json                      ← per-(uid, lto28) run status (written incrementally)
  raw/
    {lto28_name}/
      {uid}.jsonl                    ← one line per seed (n_seeds lines)
  summary/
    all_runs.csv                     ← flat: uid, lto28_name, run, seed, stopped, stop_step, decisions, ...
    stop_stats.csv                   ← aggregated per (uid, lto28): stop_rate, mean/std stop_step, decision dist

Usage examples
──────────────
# Minimal: all users, 50 seeds, configs from JSON
python3 run_campaign28_sweep.py \\
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --ckpt /tmp/FullProductGPT_...pt \\
    --feat_xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \\
    --lto28_configs configs/lto28_configs.json \\
    --sweep_name c28_sweep_v1 \\
    --out_root /home/ec2-user/outputs \\
    --n_seeds 50 \\
    --seed_base 42

# Quick smoke-test: 1 user, 5 seeds, 2 configs
python3 run_campaign28_sweep.py \\
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --ckpt /tmp/FullProductGPT_...pt \\
    --feat_xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \\
    --lto28_configs configs/lto28_configs.json \\
    --sweep_name smoke_test \\
    --out_root /home/ec2-user/outputs \\
    --n_seeds 5 \\
    --seed_base 42 \\
    --n_users 1 \\
    --quiet
"""

import argparse
import copy
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── imports from the calibrated inference module ──────────────────────────────
from infer_new_campaign_calibrated import (
    AI_RATE,
    DECISION_IDS,
    EOS_PROD_ID_LOCAL,
    GenValidationCfg,
    IdentityCalibrator,
    TraceCfg,
    _iter_rows,
    clean_state_dict,
    generate_campaign28_step2_simulated_outcomes,
    infer_ckpt_shapes,
    load_calibrator,
    load_feature_tensor,
    maybe_append_missing_terminal_block,
    parse_int_sequence,
    validate_seq_campaign28,
)
from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer


# ─────────────────────────────────────────────────────────────────────────────
# LTO28 config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_lto28_configs(path: str) -> List[Dict[str, Any]]:
    """
    Load lto28 configurations from a JSON file.

    Expected format (list of dicts):
        [
          {"name": "figA30_wep54_51", "lto28": [30, 0, 54, 51]},
          {"name": "figA22_figB30_wep45_50", "lto28": [22, 30, 45, 50]},
          ...
        ]

    Each entry must have:
      - "name"  : str — short identifier used as directory name and in CSVs
      - "lto28" : list[int] of length 4 — [figA, figB, wep1, wep2]

    Optional per-config overrides (merged with CLI defaults if present):
      - "calibrator_ckpt" : str
      - "temperature"     : float
    """
    with open(path) as f:
        configs = json.load(f)

    for i, cfg in enumerate(configs):
        assert "name" in cfg,  f"lto28 config [{i}] missing 'name'"
        assert "lto28" in cfg, f"lto28 config [{i}] missing 'lto28'"
        assert len(cfg["lto28"]) == 4, \
            f"lto28 config [{i}] '{cfg['name']}': expected 4 tokens, got {cfg['lto28']}"
        # sanitise name (no spaces, no slashes)
        cfg["name"] = cfg["name"].replace(" ", "_").replace("/", "-")

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

class Manifest:
    """Incrementally-written JSON manifest tracking every (uid, lto28) pair."""

    def __init__(self, path: Path):
        self.path = path
        self._records: Dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                self._records = json.load(f)

    def _key(self, uid: str, lto28_name: str) -> str:
        return f"{uid}|{lto28_name}"

    def mark_started(self, uid: str, lto28_name: str):
        key = self._key(uid, lto28_name)
        self._records[key] = {
            "uid": uid, "lto28_name": lto28_name,
            "status": "running", "started_at": datetime.utcnow().isoformat(),
            "finished_at": None, "n_runs": 0, "error": None,
        }
        self._flush()

    def mark_done(self, uid: str, lto28_name: str, n_runs: int):
        key = self._key(uid, lto28_name)
        self._records[key]["status"] = "done"
        self._records[key]["finished_at"] = datetime.utcnow().isoformat()
        self._records[key]["n_runs"] = n_runs
        self._flush()

    def mark_error(self, uid: str, lto28_name: str, error: str):
        key = self._key(uid, lto28_name)
        self._records[key]["status"] = "error"
        self._records[key]["finished_at"] = datetime.utcnow().isoformat()
        self._records[key]["error"] = error
        self._flush()

    def is_done(self, uid: str, lto28_name: str) -> bool:
        return self._records.get(self._key(uid, lto28_name), {}).get("status") == "done"

    def summary(self) -> Dict[str, int]:
        counts: Counter = Counter(v["status"] for v in self._records.values())
        return dict(counts)

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self._records, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV writers
# ─────────────────────────────────────────────────────────────────────────────

ALL_RUNS_COLS = [
    "uid", "lto28_name",
    "lto28_figA", "lto28_figB", "lto28_wep1", "lto28_wep2",
    "run", "seed",
    "stopped", "stop_step", "n_decisions",
    "decisions",          # JSON array string — easy to parse later
    "final_states",       # JSON object string
    "calibrated",
    "ts",                 # unix timestamp of this run
]

STOP_STATS_COLS = [
    "uid", "lto28_name",
    "lto28_figA", "lto28_figB", "lto28_wep1", "lto28_wep2",
    "n_runs",
    "stop_rate",
    "mean_stop_step", "std_stop_step",
    "mean_n_decisions", "std_n_decisions",
    "dec_counts",         # JSON: {dec_id: count} across all runs
]


def open_csv(path: Path, cols: List[str]):
    """Open (or append to) a CSV file, writing header only if new."""
    is_new = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer


def _stop_step_safe(r: Dict) -> Optional[int]:
    ss = r.get("stop_step")
    if ss is None or (isinstance(ss, float) and np.isnan(ss)):
        return None
    return int(ss)


def compute_stop_stats(
    uid: str,
    lto28_cfg: Dict,
    runs: List[Dict],
) -> Dict[str, Any]:
    """Aggregate statistics across all seeds for one (uid, lto28)."""
    lto28 = lto28_cfg["lto28"]
    stopped_runs = [r for r in runs if r.get("stopped", False)]
    stop_steps = [_stop_step_safe(r) for r in stopped_runs if _stop_step_safe(r) is not None]
    all_n_decs = [len(r.get("Campaign28_Decisions", [])) for r in runs]

    dec_counts: Counter = Counter()
    for r in runs:
        for d in r.get("Campaign28_Decisions", []):
            dec_counts[str(d)] += 1

    return {
        "uid": uid,
        "lto28_name": lto28_cfg["name"],
        "lto28_figA": lto28[0], "lto28_figB": lto28[1],
        "lto28_wep1": lto28[2], "lto28_wep2": lto28[3],
        "n_runs": len(runs),
        "stop_rate": len(stopped_runs) / len(runs) if runs else 0.0,
        "mean_stop_step": float(np.mean(stop_steps)) if stop_steps else None,
        "std_stop_step":  float(np.std(stop_steps))  if stop_steps else None,
        "mean_n_decisions": float(np.mean(all_n_decs)) if all_n_decs else None,
        "std_n_decisions":  float(np.std(all_n_decs))  if all_n_decs else None,
        "dec_counts": json.dumps(dict(dec_counts)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (shared across all runs)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
    "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow",
    "WeaponTypeMagic", "WeaponTypePolearm",
    "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire",
    "EthnicityThunder", "EthnicityWind",
    "GenderFemale", "GenderMale",
    "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
    "type_figure",
    "MinimumAttack", "MaximumAttack",
    "MinSpecialEffect", "MaxSpecialEffect",
    "SpecialEffectEfficiency", "SpecialEffectExpertise", "SpecialEffectAttack",
    "SpecialEffectSuper", "SpecialEffectRatio", "SpecialEffectPhysical",
    "SpecialEffectLife",
    "LTO",
]

SPECIAL_IDS = [0, 10, 11, 12, 57, 58, 59]


def load_model(ckpt_path: str, feat_xlsx: str, device: torch.device):
    """Load model checkpoint once."""
    cfg = get_config()
    feature_tensor = load_feature_tensor(
        Path(feat_xlsx), FEATURE_COLS, cfg["vocab_size_src"]
    )

    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        raw_sd = state["model_state_dict"]
    elif "module" in state:
        raw_sd = state["module"]
    else:
        raw_sd = state
    state_dict = clean_state_dict(raw_sd)

    n_layers, max_seq_len = infer_ckpt_shapes(
        state_dict,
        fallback_layers=cfg["N"],
        fallback_pe=cfg.get("seq_len_tgt", 1024),
    )

    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        d_model=cfg["d_model"],
        n_layers=n_layers,
        n_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        nb_features=cfg["nb_features"],
        max_seq_len=max_seq_len,
        kernel_type=cfg["kernel_type"],
        feature_tensor=feature_tensor,
        special_token_ids=SPECIAL_IDS,
    ).to(device).eval()

    model.load_state_dict(state_dict, strict=True)
    print(f"[MODEL] Loaded from {ckpt_path}")
    return model, max_seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Per-(user, lto28) runner
# ─────────────────────────────────────────────────────────────────────────────

def run_user_lto28(
    *,
    model,
    history_tokens: List[int],
    uid: str,
    lto28_cfg: Dict,
    n_seeds: int,
    seed_base: int,
    device: torch.device,
    calibrator,
    args,
    max_seq_len: int,
    raw_path: Path,
    quiet: bool,
) -> List[Dict]:
    """
    Run n_seeds simulations for one (user, lto28) pair.
    Writes results to raw_path (one line per seed).
    Returns list of result dicts for summary aggregation.
    """
    lto28 = lto28_cfg["lto28"]
    val_cfg = GenValidationCfg(
        require_lto4_match=True,
        strict_buy10_full=True,
        buy10_missing_is_error=True,
    )
    trace_cfg = TraceCfg(enabled=False)

    results = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "w") as fout:
        for r in range(n_seeds):
            seed = seed_base + r
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            try:
                out = generate_campaign28_step2_simulated_outcomes(
                    model=model,
                    history_tokens=list(history_tokens),  # fresh copy each run
                    lto28_tokens=lto28,
                    device=device,
                    init_prev_dec=None,
                    max_steps28=args.max_steps28,
                    stop_decision=9,
                    temperature=args.temperature,
                    greedy=args.greedy,
                    rng=rng,
                    use_epitomized=args.use_epitomized,
                    epitomized_target=args.epitomized_target,
                    max_ctx_tokens=max_seq_len,
                    trace=trace_cfg,
                    uid=uid,
                    run_id=r,
                    run_seed=seed,
                    calibrator=calibrator,
                )

                # Validate silently (suppress print in bulk mode)
                _ = validate_seq_campaign28(
                    seq_campaign28=out["seq_campaign28"],
                    expected_lto4=lto28,
                    cfg=val_cfg,
                )

                payload = {
                    "uid": uid,
                    "lto28_name": lto28_cfg["name"],
                    "lto28": lto28,
                    "run": r,
                    "seed": seed,
                    "step": 2,
                    "stopped": out["stopped"],
                    "stop_step": out["stop_step"],
                    "n_decisions": len(out["decisions28"]),
                    "Campaign28_Decisions": out["decisions28"],
                    "final_states": out["final_states"],
                    "calibrated": not isinstance(calibrator, IdentityCalibrator),
                    "ts": time.time(),
                }
                line = json.dumps(payload)
                fout.write(line + "\n")
                results.append(payload)
                if not quiet:
                    print(line, flush=True)

            except Exception as exc:
                err_payload = {
                    "uid": uid, "lto28_name": lto28_cfg["name"],
                    "run": r, "seed": seed, "error": str(exc),
                }
                fout.write(json.dumps(err_payload) + "\n")
                print(f"  [WARN] uid={uid} lto28={lto28_cfg['name']} run={r} error: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Campaign 28 sweep orchestrator")

    # Data / model
    parser.add_argument("--data", required=True, help="Input JSONL/JSON data file")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint .pt")
    parser.add_argument("--feat_xlsx", required=True, help="Feature Excel file")

    # Calibrator
    parser.add_argument("--calibrator_ckpt", default=None)
    parser.add_argument("--calibrator_type", default="auto",
                        choices=["auto", "temperature", "platt", "vector"])

    # Sweep configuration
    parser.add_argument("--lto28_configs", required=True,
                        help="JSON file listing lto28 configurations to sweep")
    parser.add_argument("--sweep_name", default="sweep",
                        help="Short name for this experiment (used in output dir name)")
    parser.add_argument("--n_seeds", type=int, default=50,
                        help="Number of seeds per (user, lto28) pair")
    parser.add_argument("--seed_base", type=int, default=42,
                        help="First seed; subsequent seeds are seed_base + r")
    parser.add_argument("--n_users", type=int, default=0,
                        help="Max users to process (0 = all)")

    # Output
    parser.add_argument("--out_root", default="/home/ec2-user/outputs",
                        help="Root output directory")
    parser.add_argument("--skip_done", action="store_true",
                        help="Skip (uid, lto28) pairs already marked done in manifest")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-run JSON line printing")

    # Generation knobs
    parser.add_argument("--max_steps28", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--use_epitomized", action="store_true")
    parser.add_argument("--epitomized_target", type=int, default=0)

    args = parser.parse_args()

    # ── output directory setup ────────────────────────────────────────────────
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_root) / f"{args.sweep_name}_{ts_str}"
    raw_dir     = sweep_dir / "raw"
    summary_dir = sweep_dir / "summary"
    for d in [sweep_dir, raw_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── load lto28 configs ────────────────────────────────────────────────────
    lto28_configs = load_lto28_configs(args.lto28_configs)
    print(f"[SWEEP] {len(lto28_configs)} lto28 configuration(s):")
    for cfg in lto28_configs:
        print(f"  {cfg['name']:30s}  lto28={cfg['lto28']}")

    # ── save experiment config snapshot ──────────────────────────────────────
    config_snapshot = {
        "sweep_name": args.sweep_name,
        "started_at": datetime.utcnow().isoformat(),
        "data": args.data,
        "ckpt": args.ckpt,
        "feat_xlsx": args.feat_xlsx,
        "calibrator_ckpt": args.calibrator_ckpt,
        "calibrator_type": args.calibrator_type,
        "lto28_configs": lto28_configs,
        "n_seeds": args.n_seeds,
        "seed_base": args.seed_base,
        "n_users": args.n_users,
        "max_steps28": args.max_steps28,
        "temperature": args.temperature,
        "greedy": args.greedy,
        "use_epitomized": args.use_epitomized,
        "epitomized_target": args.epitomized_target,
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)
    print(f"[SWEEP] Output directory: {sweep_dir}")

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest = Manifest(sweep_dir / "manifest.json")

    # ── device + model (loaded once) ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SWEEP] Device: {device}")

    model, max_seq_len = load_model(args.ckpt, args.feat_xlsx, device)

    # ── calibrator (loaded once) ──────────────────────────────────────────────
    if args.calibrator_ckpt:
        calibrator = load_calibrator(args.calibrator_ckpt, args.calibrator_type, device)
        print(f"[SWEEP] Calibrator: {calibrator}")
    else:
        calibrator = IdentityCalibrator()
        print("[SWEEP] No calibrator — using raw logits.")

    # ── open summary CSVs ─────────────────────────────────────────────────────
    all_runs_fh,   all_runs_writer   = open_csv(summary_dir / "all_runs.csv",   ALL_RUNS_COLS)
    stop_stats_fh, stop_stats_writer = open_csv(summary_dir / "stop_stats.csv", STOP_STATS_COLS)

    # ── main sweep ────────────────────────────────────────────────────────────
    processed_users = 0
    total_pairs = 0
    t0_sweep = time.time()

    for row in _iter_rows(args.data):
        uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")

        # Parse history
        history_tokens = parse_int_sequence(row["AggregateInput"], na_to=0)
        decs = parse_int_sequence(row.get("Decision", [])) if "Decision" in row else []
        history_tokens, appended = maybe_append_missing_terminal_block(
            history_tokens, decs, terminal_prod_tok=EOS_PROD_ID_LOCAL
        )
        if appended:
            print(f"[FIX] uid={uid}: appended terminal block")
        if len(history_tokens) % AI_RATE != 0:
            print(f"[SKIP] uid={uid}: history length {len(history_tokens)} not divisible by {AI_RATE}")
            continue

        print(f"\n[USER] uid={uid}  history_blocks={len(history_tokens)//AI_RATE}")

        for lto28_cfg in lto28_configs:
            lto28_name = lto28_cfg["name"]

            if args.skip_done and manifest.is_done(uid, lto28_name):
                print(f"  [SKIP] uid={uid} lto28={lto28_name} already done.")
                continue

            raw_path = raw_dir / lto28_name / f"{uid}.jsonl"
            manifest.mark_started(uid, lto28_name)
            t0 = time.time()

            try:
                results = run_user_lto28(
                    model=model,
                    history_tokens=history_tokens,
                    uid=uid,
                    lto28_cfg=lto28_cfg,
                    n_seeds=args.n_seeds,
                    seed_base=args.seed_base,
                    device=device,
                    calibrator=calibrator,
                    args=args,
                    max_seq_len=max_seq_len,
                    raw_path=raw_path,
                    quiet=args.quiet,
                )
                manifest.mark_done(uid, lto28_name, n_runs=len(results))

                # ── write all_runs.csv rows ────────────────────────────────
                lto28 = lto28_cfg["lto28"]
                for r in results:
                    all_runs_writer.writerow({
                        "uid": uid,
                        "lto28_name": lto28_name,
                        "lto28_figA": lto28[0], "lto28_figB": lto28[1],
                        "lto28_wep1": lto28[2], "lto28_wep2": lto28[3],
                        "run": r["run"],
                        "seed": r["seed"],
                        "stopped": int(r.get("stopped", False)),
                        "stop_step": r.get("stop_step"),
                        "n_decisions": r.get("n_decisions", 0),
                        "decisions": json.dumps(r.get("Campaign28_Decisions", [])),
                        "final_states": json.dumps(r.get("final_states", {})),
                        "calibrated": int(r.get("calibrated", False)),
                        "ts": r.get("ts", ""),
                    })
                all_runs_fh.flush()

                # ── write stop_stats.csv row ───────────────────────────────
                stats = compute_stop_stats(uid, lto28_cfg, results)
                stop_stats_writer.writerow(stats)
                stop_stats_fh.flush()

                elapsed = time.time() - t0
                print(f"  [DONE] uid={uid} lto28={lto28_name}  "
                      f"n_runs={len(results)} "
                      f"stop_rate={stats['stop_rate']:.2f} "
                      f"mean_stop_step={stats['mean_stop_step']} "
                      f"elapsed={elapsed:.1f}s")
                total_pairs += 1

            except Exception as exc:
                manifest.mark_error(uid, lto28_name, str(exc))
                print(f"  [ERROR] uid={uid} lto28={lto28_name}: {exc}")

        processed_users += 1
        if args.n_users and processed_users >= args.n_users:
            break

    # ── close CSVs ────────────────────────────────────────────────────────────
    all_runs_fh.close()
    stop_stats_fh.close()

    # ── final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0_sweep
    print(f"\n{'='*60}")
    print(f"[SWEEP COMPLETE]")
    print(f"  Users processed : {processed_users}")
    print(f"  (uid, lto28) pairs: {total_pairs}")
    print(f"  Seeds per pair  : {args.n_seeds}")
    print(f"  Total runs      : {total_pairs * args.n_seeds}")
    print(f"  Total elapsed   : {elapsed_total:.1f}s")
    print(f"  Manifest status : {manifest.summary()}")
    print(f"  Output dir      : {sweep_dir}")
    print(f"  Summary CSVs    : {summary_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()