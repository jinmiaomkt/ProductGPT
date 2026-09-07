from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

# ──── Experiment directory ────
EXP_DIR = Path("/home/ec2-user/ProductGPT/ray_results/Mixture2_RayTune")
CKPT_DIR = Path("/home/ec2-user/output/checkpoints")
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
                r = json.loads(line)
                rows.append(r)
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
    candidates = set()
    for p in exp_dir.rglob("params.json"):
        candidates.add(p.parent)
    for p in exp_dir.rglob("result.json"):
        candidates.add(p.parent)
    for p in exp_dir.rglob("progress.csv"):
        candidates.add(p.parent)
    return sorted(candidates)

def is_phase_a_config(cfg: dict) -> bool:
    if not cfg:
        return True
    if "do_infer" in cfg and cfg["do_infer"] is not False:
        return False
    return True


# ──── NEW: Build the set of available calibrator UIDs ────
def get_available_calibrator_uids(ckpt_dir: Path) -> set[str]:
    """
    Scan checkpoint directory for calibrator_*.pt files.
    Return set of UID strings (everything between 'calibrator_' and '.pt').
    """
    uids = set()
    for f in ckpt_dir.glob("calibrator_mixture2_*.pt"):
        # calibrator_mixture2_performerfeatures48_dmodel96_ff288_N5_heads4_lr0.0008..._w1_fold0.pt
        uid = f.name.replace("calibrator_", "").replace(".pt", "")
        uids.add(uid)
    return uids


def cfg_to_uid(cfg: dict, fold_id: int = 0) -> str | None:
    """
    Reconstruct the UID string from a trial's config, matching the naming
    convention in train_model():
      mixture2_performerfeatures{nb}_dmodel{dm}_ff{ff}_N{N}_heads{h}_lr{lr}_w{w}_fold{fold}
    """
    try:
        dm_heads = cfg.get("dm_heads")
        if dm_heads is None:
            return None

        if isinstance(dm_heads, (list, tuple)):
            d_model, num_heads = dm_heads
        else:
            return None

        nb = cfg.get("nb_features")
        N = cfg.get("N")
        lr = cfg.get("lr")
        w = cfg.get("weight", 1)
        dff_mult = cfg.get("dff_mult", 2)
        d_ff = min(int(d_model * dff_mult), 512)

        if any(v is None for v in [nb, N, lr]):
            return None

        uid = (
            f"mixture2_performerfeatures{nb}"
            f"_dmodel{d_model}_ff{d_ff}"
            f"_N{N}_heads{num_heads}"
            f"_lr{lr}_w{w}_fold{fold_id}"
        )
        return uid
    except Exception:
        return None


def main():
    if not EXP_DIR.exists():
        print(f"EXP_DIR not found: {EXP_DIR}")
        return

    # ──── NEW: load available calibrators ────
    cal_uids = get_available_calibrator_uids(CKPT_DIR)
    print(f"Found {len(cal_uids)} calibrator files in {CKPT_DIR}")

    # Also check which calibrator UIDs have matching checkpoints (locally or on S3)
    cal_with_ckpt = set()
    for uid in cal_uids:
        ckpt_path = CKPT_DIR / f"FullProductGPT_{uid}.pt"
        if ckpt_path.exists():
            cal_with_ckpt.add(uid)
    print(f"  of which {len(cal_with_ckpt)} also have a local checkpoint")

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

        # ──── NEW: check calibrator availability ────
        uid = cfg_to_uid(cfg, fold_id=0)
        has_calibrator = uid is not None and uid in cal_uids
        has_ckpt_local = uid is not None and uid in cal_with_ckpt

        row = {
            "trial_dir": str(td),
            "trial_name": td.name,
            "val_nll": float(best_row.get("val_nll")),
            "epoch_at_best": best_row.get("epoch"),
            "val_hit": best_row.get("val_hit"),
            "val_f1_macro": best_row.get("val_f1_macro"),
            "val_auprc_macro": best_row.get("val_auprc_macro"),
            "last_epoch": (last_row or {}).get("epoch"),
            "uid": uid or "",
            "has_calibrator": has_calibrator,
            "has_ckpt_local": has_ckpt_local,
        }

        for k in [
            "N", "nb_features", "dm_heads", "dropout", "lr", "weight", "tau",
            "warmup_steps", "data_frac", "augment_train", "permute_repeat",
            "do_infer", "num_epochs", "dff_mult"
        ]:
            if k in cfg:
                row[f"cfg_{k}"] = cfg[k]

        rows.append(row)

    if not rows:
        print("No usable trial metrics found.")
        return

    df = pd.DataFrame(rows).sort_values("val_nll", ascending=True).reset_index(drop=True)

    recent_pattern = r"_2026-03-\d{2}_"
    df = df[df["trial_name"].str.contains(recent_pattern, regex=True, na=False)].copy()

    if df.empty:
        print("No trials matched the date filter. Showing ALL trials instead:")
        df = pd.DataFrame(rows).sort_values("val_nll", ascending=True).reset_index(drop=True)

    # ──── Show ALL trials ranked ────
    print(f"\n===== TOP 10 ALL TRIALS BY val_nll =====")
    show_cols = [c for c in [
        "val_nll", "val_hit", "val_f1_macro", "val_auprc_macro",
        "has_calibrator", "has_ckpt_local",
        "cfg_N", "cfg_nb_features", "cfg_dm_heads", "cfg_lr",
        "cfg_tau",
    ] if c in df.columns]
    print(df[show_cols].head(10).to_string(index=False))

    # ──── NEW: Show best trial WITH calibrator ────
    df_cal = df[df["has_calibrator"] == True].copy()

    print(f"\n===== TOP 10 TRIALS WITH CALIBRATOR BY val_nll =====")
    if df_cal.empty:
        print("  ** NO trials have a matching calibrator **")
        print("  You will need to retrain Phase B for the best trial.")
    else:
        print(df_cal[show_cols].head(10).to_string(index=False))

        best_cal = df_cal.iloc[0]
        print(f"\n===== BEST TRIAL WITH CALIBRATOR =====")
        print(f"  UID:          {best_cal['uid']}")
        print(f"  val_nll:      {best_cal['val_nll']:.4f}")
        print(f"  val_hit:      {best_cal.get('val_hit', '?')}")
        print(f"  val_f1_macro: {best_cal.get('val_f1_macro', '?')}")
        print(f"  val_auprc:    {best_cal.get('val_auprc_macro', '?')}")
        print(f"  ckpt local:   {best_cal['has_ckpt_local']}")
        print(f"\n  Checkpoint: FullProductGPT_{best_cal['uid']}.pt")
        print(f"  Calibrator: calibrator_{best_cal['uid']}.pt")

    # ──── Compare: best overall vs best with calibrator ────
    best_overall = df.iloc[0]
    print(f"\n===== COMPARISON =====")
    print(f"  Best overall:        val_nll={best_overall['val_nll']:.4f}  has_calibrator={best_overall['has_calibrator']}")
    if not df_cal.empty:
        best_cal = df_cal.iloc[0]
        print(f"  Best w/ calibrator:  val_nll={best_cal['val_nll']:.4f}")
        gap = best_cal["val_nll"] - best_overall["val_nll"]
        print(f"  Gap: {gap:+.4f} NLL")
        if gap > 0.02:
            print("  >> Gap is significant — consider retraining Phase B with calibrator upload fix.")
        else:
            print("  >> Gap is small — the calibrated model should be fine to use.")
    else:
        print("  Best w/ calibrator:  NONE — retrain Phase B needed.")

    # ──── Save ────
    out_csv = EXP_DIR / "phaseA_mixture_ranked_with_calibrator.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full ranking to: {out_csv}")


if __name__ == "__main__":
    main()