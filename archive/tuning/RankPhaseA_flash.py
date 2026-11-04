from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

# ── Update this to your Flash experiment directory ──
EXP_DIR = Path("/home/ec2-user/ProductGPT/ray_results/ProductGPT_Flash_RayTune")
METRIC = "val_nll"

def load_params(trial_dir: Path):
    p = trial_dir / "params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def load_best_from_result_json(path: Path, metric: str):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    if metric not in df.columns:
        return None, rows[-1]

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        return None, rows[-1]

    best_idx = df[metric].idxmin()
    return df.loc[best_idx].to_dict(), rows[-1]

def load_best_from_progress_csv(path: Path, metric: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None
    if df.empty:
        return None, None
    if metric not in df.columns:
        return None, df.iloc[-1].to_dict()

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        return None, None

    return df.loc[df[metric].idxmin()].to_dict(), df.iloc[-1].to_dict()

def find_trial_dirs(exp_dir: Path):
    candidates = set()
    for pattern in ["params.json", "result.json", "progress.csv"]:
        for p in exp_dir.rglob(pattern):
            candidates.add(p.parent)
    return sorted(candidates)

def main():
    if not EXP_DIR.exists():
        print(f"EXP_DIR not found: {EXP_DIR}")
        return

    trial_dirs = find_trial_dirs(EXP_DIR)
    print(f"Found {len(trial_dirs)} candidate trial directories under {EXP_DIR}")

    rows = []
    for td in trial_dirs:
        cfg = load_params(td)

        best_row, last_row = None, None

        rj = td / "result.json"
        pc = td / "progress.csv"

        if rj.exists():
            best_row, last_row = load_best_from_result_json(rj, METRIC)
        if best_row is None and pc.exists():
            best_row, last_row = load_best_from_progress_csv(pc, METRIC)
        if best_row is None:
            continue

        row = {
            "trial_dir": str(td),
            "trial_name": td.name,
            "val_nll": float(best_row.get("val_nll", float("nan"))),
            "epoch_at_best": best_row.get("epoch"),
            "val_hit": best_row.get("val_hit"),
            "val_f1_macro": best_row.get("val_f1_macro"),
            "val_auprc_macro": best_row.get("val_auprc_macro"),
            "last_epoch": (last_row or {}).get("epoch"),
        }

        for k in [
            "N", "dm_heads", "dropout", "lr", "tau", "gamma",
            "warmup_steps", "data_frac", "batch_size",
            "dff_mult", "label_smoothing",
        ]:
            if k in cfg:
                row[f"cfg_{k}"] = cfg[k]

        rows.append(row)

    if not rows:
        print("No usable trial metrics found.")
        return

    df = pd.DataFrame(rows).sort_values("val_nll", ascending=True).reset_index(drop=True)

    # ── Show top 10 ──
    print(f"\n===== TOP 10 TRIALS BY {METRIC} (lower is better) =====")
    show_cols = [c for c in [
        "val_nll", "epoch_at_best", "val_hit", "val_f1_macro", "val_auprc_macro",
        "cfg_N", "cfg_dm_heads", "cfg_dropout", "cfg_lr", "cfg_tau", "cfg_gamma",
        "cfg_warmup_steps", "cfg_dff_mult", "cfg_batch_size",
        "trial_name",
    ] if c in df.columns]
    print(df[show_cols].head(10).to_string(index=False))

    # ── Best trial details ──
    best = df.iloc[0]
    print("\n===== BEST TRIAL =====")
    for k in df.columns:
        print(f"  {k}: {best[k]}")

    # ── Print the Phase B command ──
    cfg = load_params(Path(best["trial_dir"]))
    if cfg:
        print("\n===== PHASE B: RETRAIN BEST CONFIG ON FULL DATA =====")
        print("Copy this config into your retrain script or run:")
        print()

        dm, heads = cfg.get("dm_heads", [64, 2])
        d_ff = int(dm * cfg.get("dff_mult", 3))
        print(f"  d_model={dm}, num_heads={heads}, d_ff={d_ff}")
        print(f"  N={cfg.get('N')}")
        print(f"  lr={cfg.get('lr')}")
        print(f"  dropout={cfg.get('dropout')}")
        print(f"  tau={cfg.get('tau')}")
        print(f"  gamma={cfg.get('gamma')}")
        print(f"  warmup_steps={cfg.get('warmup_steps')}")
        print(f"  batch_size={cfg.get('batch_size', 4)}")
        print(f"  label_smoothing={cfg.get('label_smoothing', 0.0)}")
        print()
        print("  data_frac=1.0, num_epochs=200, do_infer=True")

    # ── Save ranking ──
    out_csv = EXP_DIR / "flash_phaseA_ranked_trials.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full ranking to: {out_csv}")

if __name__ == "__main__":
    main()