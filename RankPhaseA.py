import json
from pathlib import Path
import pandas as pd

EXP_DIRS = [
    Path("/home/ec2-user/ray_results/f_2026-02-15_21-37-23"),
    Path("/home/ec2-user/ray_results/f_2026-02-15_21-38-49"),
]

METRIC = "val_nll"   # change if your old run used a different key

def load_params(trial_dir: Path):
    p = trial_dir / "params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

rows = []
for exp in EXP_DIRS:
    if not exp.exists():
        continue
    for prog in exp.glob("**/progress.csv"):
        trial_dir = prog.parent
        try:
            df = pd.read_csv(prog, usecols=lambda c: c in {METRIC, "epoch", "training_iteration"})
            if METRIC not in df.columns:
                continue
            s = pd.to_numeric(df[METRIC], errors="coerce").dropna()
            if s.empty:
                continue
            best_i = s.idxmin()
            params = load_params(trial_dir)
            rows.append({
                "exp": str(exp),
                "trial_dir": str(trial_dir),
                "trial_name": trial_dir.name,
                "best_val_nll": float(s.loc[best_i]),
                "best_epoch": float(df.loc[best_i].get("epoch", float("nan"))),
                "training_iteration": float(df.loc[best_i].get("training_iteration", float("nan"))),
                "params": params,
            })
        except Exception:
            continue

if not rows:
    raise SystemExit("No usable progress.csv with metric found. Check EXP_DIRS and METRIC name.")

rows.sort(key=lambda r: r["best_val_nll"])
best = rows[0]

print("\n===== BEST PHASE-A TRIAL (from local progress.csv) =====")
print("Experiment:", best["exp"])
print("Trial dir :", best["trial_dir"])
print("Best val_nll:", best["best_val_nll"])
print("Best epoch  :", best["best_epoch"])
print("\nParams (params.json):")
for k, v in best["params"].items():
    print(f"  {k}: {v}")

print("\n===== TOP 10 =====")
for i, r in enumerate(rows[:10], 1):
    print(f"{i:2d}. {r['best_val_nll']:.6f} | {r['trial_name']}")