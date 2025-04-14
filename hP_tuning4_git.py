# hyperparam_sweep.py

import itertools
import json
import torch

# Local modules
from config4git import get_config
from train4_decoderonly_git import train_model

# Define the hyperparameter ranges
# d_model_values = [32, 64, 128]
# d_ff_values    = [32, 64, 128]
d_model_values = [64]
d_ff_values    = [64]
N_values       = [2, 4, 8]
num_heads_values = [2, 4, 8]
gamma_values   = [1.0, 2.0, 4.0]
lr_values      = [0.0001, 0.00001, 0.000001]
weight_values  = [2, 4, 8, 16]

def hyperparam_sweep():
    all_combinations = itertools.product(d_model_values, d_ff_values, N_values, num_heads_values, gamma_values)

    for (d_model, d_ff, N, num_heads, gamma) in all_combinations:
        # 1) Get default config
        config = get_config()

        # 2) Override hyperparams
        config['d_model'] = d_model
        config['d_ff']    = d_ff
        config['N']       = N
        config['num_heads'] = num_heads
        config['gamma'] = gamma

        # 3) Unique name
        unique_id = f"dmodel{d_model}_ff{d_ff}_N{N}_heads{num_heads}_gamma{gamma}"
        config['model_basename'] = f"MyProductGPT_{unique_id}_"

        # 4) Train model
        final_metrics = train_model(config)

        # 5) Save final metrics for this combo
        metrics_out = {
            "d_model": d_model,
            "d_ff": d_ff,
            "N": N,
            "num_heads": num_heads,
            "gamma": gamma,
            "val_loss": final_metrics['val_loss'],
            "val_ppl": final_metrics['val_ppl'],
            "confusion_matrix": final_metrics['val_confusion_matrix'],
            "hit_rate": final_metrics['val_hit_rate'],
            "best_checkpoint_path": final_metrics['best_checkpoint_path']
        }
        metrics_file = f"results_{unique_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_out, f, indent=2)

        print(f"[Done] {unique_id} -> {metrics_file}")

if __name__ == "__main__":
    hyperparam_sweep()
