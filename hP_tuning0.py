import itertools
import json
import torch

# Local modules
from config0git import get_config, get_weights_file_path
from train0_per_git import train_model

# 1) Define the ranges for each hyperparameter
d_model_values    = [32, 64, 128]
d_ff_values       = [32, 64, 128]
N_values          = [2, 4, 8]
num_heads_values  = [2, 4, 8]
gamma_values      = [1.0, 2.0, 4.0]

def hyperparam_sweep():
    """
    Iterates over all hyperparameter combos, trains, and logs final metrics.
    """
    # 2) Create an iterator of all combinations
    all_combinations = itertools.product(d_model_values, d_ff_values, N_values, num_heads_values, gamma_values)

    for (d_model, d_ff, N, num_heads, gamma) in all_combinations:
        # 3) Construct a config for the current combination
        config = get_config()
        config['d_model']   = d_model
        config['d_ff']      = d_ff
        config['N']         = N
        config['num_heads'] = num_heads
        config['gamma'] = gamma

        # 4) Create a custom prefix so each run gets its own file name
        # Example: "tmodel_dmodel64_ff128_N2_heads1_"
        unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_gamma{gamma}"
        config['model_basename'] = f"MyProductGPT_{unique_id}_"

        # 5) Run your training function, which returns final metrics
        # (Make sure train_model actually returns the metrics you need below.)
        final_metrics = train_model(config)

        # 6) Save final metrics to JSON
        #    e.g., validation loss, confusion matrix, or anything else returned
        metrics = {
            "d_model": d_model,
            "d_ff": d_ff,
            "N": N,
            "num_heads": num_heads,
            "gamma": gamma,
            "val_loss": final_metrics.get('val_loss'),
            "val_hit_rate": final_metrics.get('val_hit_rate'),
            "confusion_matrix": final_metrics.get('confusion_matrix'),
            "best_checkpoint_path": final_metrics.get('best_checkpoint_path'),
        }

        # Write out a JSON file named like "results_dmodel32_ff64_N2_heads4.json"
        metrics_file = f"results_{unique_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 7) Print or log the results
        print(f"Completed run {unique_id}. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    hyperparam_sweep()
