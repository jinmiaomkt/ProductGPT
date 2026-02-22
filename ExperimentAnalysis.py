from pathlib import Path
from ray.tune import ExperimentAnalysis

exp_dir = Path("/home/ec2-user/ProductGPT/ray_results/ProductGPT_RayTune")  # change if needed
ea = ExperimentAnalysis(str(exp_dir))

print("Trials loaded:", len(ea.trials))

# Show each trial status + whether val_nll exists
for t in ea.trials:
    r = getattr(t, "last_result", {}) or {}
    print(
        t.trial_id,
        "| status:", getattr(t, "status", "NA"),
        "| has_val_nll:", ("val_nll" in r),
        "| last keys sample:", list(r.keys())[:10]
    )

print("\nTrying dataframe() without metric...")
df0 = ea.dataframe()
print("Columns:", list(df0.columns))
print(df0.head(3).to_string(index=False))