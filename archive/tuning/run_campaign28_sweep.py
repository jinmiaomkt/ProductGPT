#!/usr/bin/env python3
"""
run_campaign28_sweep.py
Disciplined sweep orchestrator for Campaign 28 inference.
"""

import argparse
import csv
import io
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import torch

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
# S3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def s3_put(s3_client, bucket: str, key: str, body: str) -> None:
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))


def s3_get(s3_client, bucket: str, key: str) -> Optional[str]:
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read().decode("utf-8")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LTO28 config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_lto28_configs(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        configs = json.load(f)
    for i, cfg in enumerate(configs):
        assert "name"  in cfg, f"lto28 config [{i}] missing 'name'"
        assert "lto28" in cfg, f"lto28 config [{i}] missing 'lto28'"
        assert len(cfg["lto28"]) == 4, \
            f"lto28 config [{i}] '{cfg['name']}': expected 4 tokens, got {cfg['lto28']}"
        cfg["name"] = cfg["name"].replace(" ", "_").replace("/", "-")
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Manifest — works in both S3 and local mode
# ─────────────────────────────────────────────────────────────────────────────

class Manifest:
    def __init__(self, *, s3_client=None, bucket: str = "",
                 s3_key: str = "", local_path: Optional[Path] = None):
        self.s3         = s3_client
        self.bucket     = bucket
        self.s3_key     = s3_key
        self.local_path = local_path
        self._records: Dict[str, Any] = {}

        # Load existing manifest if present (for --skip_done resume)
        if s3_client and bucket and s3_key:
            existing = s3_get(s3_client, bucket, s3_key)
            if existing:
                self._records = json.loads(existing)
                print(f"[MANIFEST] Loaded {len(self._records)} records from S3.")
        elif local_path and local_path.exists():
            with open(local_path) as f:
                self._records = json.load(f)

    def _key(self, uid: str, lto28_name: str) -> str:
        return f"{uid}|{lto28_name}"

    def mark_started(self, uid: str, lto28_name: str):
        k = self._key(uid, lto28_name)
        self._records[k] = {
            "uid": uid, "lto28_name": lto28_name,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "n_runs": 0, "error": None,
        }
        self._flush()

    def mark_done(self, uid: str, lto28_name: str, n_runs: int):
        k = self._key(uid, lto28_name)
        self._records[k]["status"] = "done"
        self._records[k]["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._records[k]["n_runs"] = n_runs
        self._flush()

    def mark_error(self, uid: str, lto28_name: str, error: str):
        k = self._key(uid, lto28_name)
        self._records[k]["status"] = "error"
        self._records[k]["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._records[k]["error"] = error
        self._flush()

    def is_done(self, uid: str, lto28_name: str) -> bool:
        return self._records.get(self._key(uid, lto28_name), {}).get("status") == "done"

    def summary(self) -> Dict[str, int]:
        return dict(Counter(v["status"] for v in self._records.values()))

    def _flush(self):
        body = json.dumps(self._records, indent=2)
        if self.s3 and self.bucket and self.s3_key:
            s3_put(self.s3, self.bucket, self.s3_key, body)
        elif self.local_path:
            with open(self.local_path, "w") as f:
                f.write(body)


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

ALL_RUNS_COLS = [
    "uid", "lto28_name",
    "lto28_figA", "lto28_figB", "lto28_wep1", "lto28_wep2",
    "run", "seed", "stopped", "stop_step", "n_decisions",
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


def make_csv_writer(cols: List[str]):
    """In-memory CSV writer for S3 upload."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    return buf, writer


def open_local_csv(path: Path, cols: List[str]):
    """Local CSV writer, appends if file exists."""
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


def compute_stop_stats(uid: str, lto28_cfg: Dict, runs: List[Dict]) -> Dict[str, Any]:
    lto28        = lto28_cfg["lto28"]
    stopped_runs = [r for r in runs if r.get("stopped", False)]
    stop_steps   = [_stop_step_safe(r) for r in stopped_runs
                    if _stop_step_safe(r) is not None]
    all_n_decs   = [len(r.get("Campaign28_Decisions", [])) for r in runs]
    dec_counts: Counter = Counter()
    for r in runs:
        for d in r.get("Campaign28_Decisions", []):
            dec_counts[str(d)] += 1
    return {
        "uid": uid, "lto28_name": lto28_cfg["name"],
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
# Model loading
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
    "SpecialEffectLife", "LTO",
]

SPECIAL_IDS = [0, 10, 11, 12, 57, 58, 59]


def load_model(ckpt_path: str, feat_xlsx: str, device: torch.device):
    cfg = get_config()
    feature_tensor = load_feature_tensor(Path(feat_xlsx), FEATURE_COLS, cfg["vocab_size_src"])
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
    s3_client=None,
    s3_bucket: str = "",
    s3_prefix: str = "",
    local_raw_path: Optional[Path] = None,
    quiet: bool = False,
) -> List[Dict]:
    lto28     = lto28_cfg["lto28"]
    val_cfg   = GenValidationCfg(require_lto4_match=True,
                                  strict_buy10_full=True,
                                  buy10_missing_is_error=True)
    trace_cfg = TraceCfg(enabled=False)
    results   = []
    lines_buf = []

    for r in range(n_seeds):
        seed = seed_base + r
        rng  = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            out = generate_campaign28_step2_simulated_outcomes(
                model=model,
                history_tokens=list(history_tokens),
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
            lines_buf.append(line)
            results.append(payload)
            if not quiet:
                print(line, flush=True)

        except Exception as exc:
            err = {"uid": uid, "lto28_name": lto28_cfg["name"],
                   "run": r, "seed": seed, "error": str(exc)}
            lines_buf.append(json.dumps(err))
            print(f"  [WARN] uid={uid} lto28={lto28_cfg['name']} run={r} error: {exc}")

    # ── Write output ──────────────────────────────────────────────────────────
    body = "\n".join(lines_buf) + "\n"

    if s3_client and s3_bucket:
        key = f"{s3_prefix}/raw/{lto28_cfg['name']}/{uid}.jsonl"
        s3_put(s3_client, s3_bucket, key, body)
        if not quiet:
            print(f"  [S3] uploaded -> s3://{s3_bucket}/{key}")
    elif local_raw_path is not None:
        local_raw_path.parent.mkdir(parents=True, exist_ok=True)
        local_raw_path.write_text(body)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Campaign 28 sweep orchestrator")

    # Data / model
    parser.add_argument("--data",      required=True)
    parser.add_argument("--ckpt",      required=True)
    parser.add_argument("--feat_xlsx", required=True)

    # Calibrator
    parser.add_argument("--calibrator_ckpt", default=None)
    parser.add_argument("--calibrator_type", default="auto",
                        choices=["auto", "temperature", "platt", "vector"])

    # Sweep config
    parser.add_argument("--lto28_configs", required=True)
    parser.add_argument("--sweep_name",    default="sweep")
    parser.add_argument("--n_seeds",       type=int, default=50)
    parser.add_argument("--seed_base",     type=int, default=42)
    parser.add_argument("--n_users",       type=int, default=0)

    # S3 output (recommended)
    parser.add_argument("--s3_bucket", default="",
                        help="S3 bucket. If set, all outputs go to S3.")
    parser.add_argument("--s3_prefix", default="outputs",
                        help="S3 key prefix. For resume, pass the full existing sweep "
                             "prefix (with timestamp) so --skip_done finds the right manifest.")

    # Local output (fallback when --s3_bucket not set)
    parser.add_argument("--out_root", default="/home/ec2-user/outputs")

    # Behaviour
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--quiet",     action="store_true")

    # Generation knobs
    parser.add_argument("--max_steps28",       type=int,   default=500)
    parser.add_argument("--temperature",       type=float, default=1.0)
    parser.add_argument("--greedy",            action="store_true")
    parser.add_argument("--use_epitomized",    action="store_true")
    parser.add_argument("--epitomized_target", type=int,   default=0)

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    use_s3    = bool(args.s3_bucket)
    s3_client = boto3.client("s3") if use_s3 else None

    if use_s3:
        # Resume mode: caller passes the full existing prefix (with timestamp).
        # New sweep mode: append sweep_name + timestamp to the prefix.
        if args.skip_done and "/" in args.s3_prefix.strip("/"):
            s3_prefix = args.s3_prefix.strip("/")
        else:
            s3_prefix = f"{args.s3_prefix.strip('/')}/{args.sweep_name}_{ts_str}"
        sweep_label     = f"s3://{args.s3_bucket}/{s3_prefix}"
        local_sweep_dir = None
        raw_dir         = None
        summary_dir     = None
    else:
        s3_prefix       = ""
        local_sweep_dir = Path(args.out_root) / f"{args.sweep_name}_{ts_str}"
        raw_dir         = local_sweep_dir / "raw"
        summary_dir     = local_sweep_dir / "summary"
        for d in [local_sweep_dir, raw_dir, summary_dir]:
            d.mkdir(parents=True, exist_ok=True)
        sweep_label = str(local_sweep_dir)

    print(f"[SWEEP] Output: {sweep_label}")

    # ── Load configs ──────────────────────────────────────────────────────────
    lto28_configs = load_lto28_configs(args.lto28_configs)
    print(f"[SWEEP] {len(lto28_configs)} lto28 configuration(s):")
    for cfg in lto28_configs:
        print(f"  {cfg['name']:30s}  lto28={cfg['lto28']}")

    # ── Config snapshot ───────────────────────────────────────────────────────
    config_snapshot = {
        "sweep_name": args.sweep_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data": args.data, "ckpt": args.ckpt, "feat_xlsx": args.feat_xlsx,
        "calibrator_ckpt": args.calibrator_ckpt,
        "calibrator_type": args.calibrator_type,
        "lto28_configs": lto28_configs,
        "n_seeds": args.n_seeds, "seed_base": args.seed_base,
        "n_users": args.n_users, "max_steps28": args.max_steps28,
        "temperature": args.temperature, "greedy": args.greedy,
        "use_epitomized": args.use_epitomized,
        "epitomized_target": args.epitomized_target,
    }
    config_body = json.dumps(config_snapshot, indent=2)
    if use_s3:
        s3_put(s3_client, args.s3_bucket, f"{s3_prefix}/config.json", config_body)
    else:
        (local_sweep_dir / "config.json").write_text(config_body)

    # ── Manifest ──────────────────────────────────────────────────────────────
    if use_s3:
        manifest = Manifest(
            s3_client=s3_client,
            bucket=args.s3_bucket,
            s3_key=f"{s3_prefix}/manifest.json",
        )
    else:
        manifest = Manifest(local_path=local_sweep_dir / "manifest.json")

    # ── Device + model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SWEEP] Device: {device}")
    model, max_seq_len = load_model(args.ckpt, args.feat_xlsx, device)

    # ── Calibrator ────────────────────────────────────────────────────────────
    if args.calibrator_ckpt:
        calibrator = load_calibrator(args.calibrator_ckpt, args.calibrator_type, device)
        print(f"[SWEEP] Calibrator: {calibrator}")
    else:
        calibrator = IdentityCalibrator()
        print("[SWEEP] No calibrator — using raw logits.")

    # ── Summary CSV writers ───────────────────────────────────────────────────
    if use_s3:
        all_runs_fh,   all_runs_writer   = make_csv_writer(ALL_RUNS_COLS)
        stop_stats_fh, stop_stats_writer = make_csv_writer(STOP_STATS_COLS)
    else:
        all_runs_fh,   all_runs_writer   = open_local_csv(
            summary_dir / "all_runs.csv",   ALL_RUNS_COLS)
        stop_stats_fh, stop_stats_writer = open_local_csv(
            summary_dir / "stop_stats.csv", STOP_STATS_COLS)

    # ── Main sweep loop ───────────────────────────────────────────────────────
    processed_users = 0
    total_pairs     = 0
    t0_sweep        = time.time()

    for row in _iter_rows(args.data):
        uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")

        history_tokens = parse_int_sequence(row["AggregateInput"], na_to=0)
        decs = parse_int_sequence(row.get("Decision", [])) if "Decision" in row else []
        history_tokens, appended = maybe_append_missing_terminal_block(
            history_tokens, decs, terminal_prod_tok=EOS_PROD_ID_LOCAL
        )
        if appended:
            print(f"[FIX] uid={uid}: appended terminal block")
        if len(history_tokens) % AI_RATE != 0:
            print(f"[SKIP] uid={uid}: history length not divisible by {AI_RATE}")
            continue

        print(f"\n[USER] uid={uid}  history_blocks={len(history_tokens)//AI_RATE}")

        for lto28_cfg in lto28_configs:
            lto28_name = lto28_cfg["name"]

            if args.skip_done and manifest.is_done(uid, lto28_name):
                print(f"  [SKIP] uid={uid} lto28={lto28_name} already done.")
                continue

            local_raw_path = (raw_dir / lto28_name / f"{uid}.jsonl"
                              if not use_s3 else None)

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
                    s3_client=s3_client,
                    s3_bucket=args.s3_bucket,
                    s3_prefix=s3_prefix,
                    local_raw_path=local_raw_path,
                    quiet=args.quiet,
                )
                manifest.mark_done(uid, lto28_name, n_runs=len(results))

                lto28 = lto28_cfg["lto28"]
                for r in results:
                    all_runs_writer.writerow({
                        "uid": uid, "lto28_name": lto28_name,
                        "lto28_figA": lto28[0], "lto28_figB": lto28[1],
                        "lto28_wep1": lto28[2], "lto28_wep2": lto28[3],
                        "run": r["run"], "seed": r["seed"],
                        "stopped": int(r.get("stopped", False)),
                        "stop_step": r.get("stop_step"),
                        "n_decisions": r.get("n_decisions", 0),
                        "decisions": json.dumps(r.get("Campaign28_Decisions", [])),
                        "final_states": json.dumps(r.get("final_states", {})),
                        "calibrated": int(r.get("calibrated", False)),
                        "ts": r.get("ts", ""),
                    })
                if not use_s3:
                    all_runs_fh.flush()

                stats = compute_stop_stats(uid, lto28_cfg, results)
                stop_stats_writer.writerow(stats)
                if not use_s3:
                    stop_stats_fh.flush()

                elapsed = time.time() - t0
                print(f"  [DONE] uid={uid} lto28={lto28_name}  "
                      f"n_runs={len(results)}  "
                      f"stop_rate={stats['stop_rate']:.2f}  "
                      f"mean_stop_step={stats['mean_stop_step']}  "
                      f"elapsed={elapsed:.1f}s")
                total_pairs += 1

            except Exception as exc:
                manifest.mark_error(uid, lto28_name, str(exc))
                print(f"  [ERROR] uid={uid} lto28={lto28_name}: {exc}")

        processed_users += 1
        if args.n_users and processed_users >= args.n_users:
            break

    # ── Upload / close summary CSVs ───────────────────────────────────────────
    if use_s3:
        s3_put(s3_client, args.s3_bucket,
               f"{s3_prefix}/summary/all_runs.csv",   all_runs_fh.getvalue())
        s3_put(s3_client, args.s3_bucket,
               f"{s3_prefix}/summary/stop_stats.csv", stop_stats_fh.getvalue())
        print(f"[S3] Summary CSVs uploaded to s3://{args.s3_bucket}/{s3_prefix}/summary/")
    else:
        all_runs_fh.close()
        stop_stats_fh.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0_sweep
    print(f"\n{'='*60}")
    print(f"[SWEEP COMPLETE]")
    print(f"  Users processed   : {processed_users}")
    print(f"  (uid,lto28) pairs : {total_pairs}")
    print(f"  Seeds per pair    : {args.n_seeds}")
    print(f"  Total runs        : {total_pairs * args.n_seeds}")
    print(f"  Total elapsed     : {elapsed_total:.1f}s")
    print(f"  Manifest status   : {manifest.summary()}")
    print(f"  Output            : {sweep_label}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()