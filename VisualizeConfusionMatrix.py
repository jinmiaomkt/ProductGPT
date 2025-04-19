import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt   # required only for the heat‑map

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the results JSON
# ─────────────────────────────────────────────────────────────────────────────
json_path = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/metrics_FullProductGPT_FeatureBasec_d16_ff16_N4_h4_lr1e-05_w4.json")          # <= put your real path here
with json_path.open() as f:
    stats = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build a DataFrame for the confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
conf_mat = pd.DataFrame(
    stats["val_confusion_matrix"],
    index=[f"Pred {i}"  for i in range(1, 10)],
    columns=[f"True {i}" for i in range(1, 10)]
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Collect the other metrics you care about
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    "Validation loss"                   : stats["val_loss"],
    "Validation perplexity"             : stats["val_ppl"],
    "Validation hit‑rate (accuracy)"    : stats["val_hit_rate"],
    "Validation macro‑F1"               : stats["val_f1_score"],
    "Validation AUPRC"                  : stats["val_auprc"],
    "Best checkpoint path"              : stats["best_checkpoint_path"],
    # model‑size hyper‑params, if you want them here:
    # "d_model"                           : stats["d_model"],
    # "d_ff"                              : stats["d_ff"],
    # "Encoder layers (N)"                : stats["N"],
    # "Heads"                             : stats["num_heads"],
    # "Learning rate"                     : stats["lr"],
    # "Weight multiplier"                 : stats["weight"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pretty‑print everything
# ─────────────────────────────────────────────────────────────────────────────
print("\nValidation confusion matrix:\n")
print(conf_mat.to_string())
print("\nOther performance metrics:\n")

# align key/value pairs nicely
key_width = max(len(k) for k in metrics.keys())
for k, v in metrics.items():
    print(f"{k:<{key_width}} : {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Optional: quick heat‑map
# ─────────────────────────────────────────────────────────────────────────────
if False:                                # flip to True if you want the plot
    ax = plt.gca()
    im = ax.imshow(conf_mat.values)      # default colour map
    ax.set_xticks(range(9), conf_mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(9), conf_mat.index)
    ax.set_title("Validation Confusion Matrix")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, conf_mat.iat[i, j],
                    ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()


