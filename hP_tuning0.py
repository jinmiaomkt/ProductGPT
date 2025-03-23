import itertools
import json
import torch

# Import your existing config
from config0git import get_config, get_weights_file_path
from train0_per_git import train_model  # Suppose your training entrypoint is train_model()

# 1) Define the ranges for each hyperparameter
d_model_values = [32, 64, 128]
d_ff_values = [32, 64, 128]
N_values = [2, 4, 8]
num_heads_values = [2, 4, 8]

def hyperparam_sweep():
    # 2) Create an iterator of all combinations
    all_combinations = itertools.product(d_model_values, d_ff_values, N_values, num_heads_values)

    for (d_model, d_ff, N, num_heads) in all_combinations:
        # 3) Construct a config for the current combination
        config = get_config()
        config['d_model'] = d_model
        config['d_ff'] = d_ff
        config['N'] = N
        config['num_heads'] = num_heads

        # 4) Create a custom prefix so each run gets its own folder or file name
        # Something like "tmodel_dmodel64_dff128_N2_numheads1"
        unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}"
        config['model_basename'] = f"tmodel_{unique_id}_"

        # 5) Run your training script, which returns final metrics
        final_metrics = train_model(config)

        # 6) Save your final metrics (loss, accuracy, confusion matrix, etc.) into JSON
        #    so you can look them up later
        metrics = {
            "d_model": d_model,
            "d_ff": d_ff,
            "N": N,
            "num_heads": num_heads,
            "val_loss": final_metrics['val_loss'],
            "val_hit_rate": final_metrics['val_hit_rate'],
            "confusion_matrix": final_metrics['confusion_matrix'],
        }
        metrics_file = f"results_{unique_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 7) Optionally print or log the results
        print(f"Saved run {unique_id} to {metrics_file}")

if __name__ == "__main__":
    hyperparam_sweep()
