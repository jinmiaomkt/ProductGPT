from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

EXP_DIR = Path("/home/ec2-user/ProductGPT/ray_results/ProductGPT_RayTune")
METRIC = "val_nll"

def load_params(trial_dir: Path):
    # Ray commonly stores params in params.json
    p = trial_dir / "params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    # Fallback: nested artifacts may store params elsewhere
    return {}

def load_best_from_result_json(path: Path, metric: str):
    """
    Ray result.json is usually JSON-lines (one record per report).
    Return (best_row_dict, last_row_dict) by min metric.
    """
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows.append(r)
            except Exception:
                continue
    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    if metric not in df.columns:
        return None, rows[-1]

    # numeric coercion
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        return None, rows[-1]

    best_idx = df[metric].idxmin()
    best_row = df.loc[best_idx].to_dict()
    last_row = rows[-1]
    return best_row, last_row

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

    best_idx = df[metric].idxmin()
    best_row = df.loc[best_idx].to_dict()
    last_row = df.iloc[-1].to_dict()
    return best_row, last_row

def find_trial_dirs(exp_dir: Path):
    """
    Trial dirs usually contain params.json and result.json/progress.csv.
    We scan recursively because you have multiple experiment_state snapshots.
    """
    candidates = set()

    for p in exp_dir.rglob("params.json"):
        candidates.add(p.parent)

    for p in exp_dir.rglob("result.json"):
        candidates.add(p.parent)

    for p in exp_dir.rglob("progress.csv"):
        candidates.add(p.parent)

    return sorted(candidates)

def is_phase_a_config(cfg: dict) -> bool:
    """
    Phase A in your tuning script is typically:
      do_infer=False, data_frac=0.05, augment_train=False
    We keep filter loose so older runs still pass.
    """
    if not cfg:
        return True  # if missing params.json, don't auto-drop
    # Strong filter if available:
    if "do_infer" in cfg and cfg["do_infer"] is not False:
        return False
    # Optional heuristics (uncomment if needed)
    # if "data_frac" in cfg and float(cfg["data_frac"]) != 0.05:
    #     return False
    # if "augment_train" in cfg and bool(cfg["augment_train"]) is not False:
    #     return False
    return True

def main():
    if not EXP_DIR.exists():
        print(f"EXP_DIR not found: {EXP_DIR}")
        return

    trial_dirs = find_trial_dirs(EXP_DIR)
    print(f"Found {len(trial_dirs)} candidate trial directories under {EXP_DIR}")

    rows = []
    for td in trial_dirs:
        cfg = load_params(td)
        if not is_phase_a_config(cfg):
            continue

        best_row = None
        last_row = None

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
            "val_nll": float(best_row.get("val_nll")),
            "epoch_at_best": best_row.get("epoch"),
            "val_hit": best_row.get("val_hit"),
            "val_f1_macro": best_row.get("val_f1_macro"),
            "val_auprc_macro": best_row.get("val_auprc_macro"),
            "last_epoch": (last_row or {}).get("epoch"),
        }

        # flatten selected config fields if present
        for k in [
            "N", "nb_features", "dm_heads", "dropout", "lr", "weight",
            "warmup_steps", "data_frac", "augment_train", "permute_repeat",
            "do_infer", "num_epochs", "dff_mult"
        ]:
            if k in cfg:
                row[f"cfg_{k}"] = cfg[k]

        rows.append(row)

    if not rows:
        print("No usable trial metrics found. (Checked both result.json and progress.csv)")
        return

    df = pd.DataFrame(rows).sort_values("val_nll", ascending=True).reset_index(drop=True)

    print("\n===== TOP 10 PHASE-A TRIALS BY val_nll =====")
    show_cols = [c for c in [
        "val_nll", "epoch_at_best", "val_hit", "val_f1_macro", "val_auprc_macro",
        "cfg_N", "cfg_nb_features", "cfg_dm_heads", "cfg_dropout", "cfg_lr",
        "cfg_weight", "cfg_warmup_steps", "cfg_data_frac", "cfg_augment_train",
        "trial_name"
    ] if c in df.columns]
    print(df[show_cols].head(10).to_string(index=False))

    best = df.iloc[0]
    print("\n===== BEST PHASE-A TRIAL =====")
    for k in df.columns:
        print(f"{k}: {best[k]}")

    out_csv = EXP_DIR / "phaseA_ranked_trials.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full ranking to: {out_csv}")

if __name__ == "__main__":
    main()