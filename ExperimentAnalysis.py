from pathlib import Path
from ray.tune import ExperimentAnalysis

# Change this to your actual experiment folder:
exp_dir = Path("/home/ec2-user/ProductGPT/ray_results/ProductGPT_RayTune")

ea = ExperimentAnalysis(str(exp_dir))

# Best trial by minimum val_nll across the full trial history
best_trial = ea.get_best_trial(metric="val_nll", mode="min", scope="all")

print("\n===== BEST PHASE-A TRIAL =====")
print("Trial ID:", best_trial.trial_id)
print("Logdir  :", best_trial.logdir)

print("\nBest config:")
for k, v in best_trial.config.items():
    print(f"  {k}: {v}")

print("\nLast reported metrics (snippet):")
for k in ["epoch", "val_nll", "val_hit", "val_f1_macro", "val_auprc_macro"]:
    print(f"  {k}: {best_trial.last_result.get(k)}")

# Optional: show top trials table
df = ea.dataframe(metric="val_nll", mode="min")
cols = [c for c in [
    "trial_id", "logdir",
    "val_nll", "val_hit", "val_f1_macro", "val_auprc_macro",
    "config/N", "config/nb_features", "config/dm_heads",
    "config/dropout", "config/lr", "config/weight",
    "config/warmup_steps", "config/data_frac", "config/augment_train"
] if c in df.columns]

print("\n===== TOP 10 TRIALS (by val_nll) =====")
print(df.sort_values("val_nll", ascending=True)[cols].head(10).to_string(index=False))