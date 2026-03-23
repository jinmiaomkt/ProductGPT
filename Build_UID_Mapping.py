import torch
import pandas as pd

from config4 import get_config
from your_training_file import build_dataloaders   # replace with your actual filename

ckpt_path = "/path/to/your_model.pt"
out_csv = "/path/to/user_mixture_weights.csv"

cfg = get_config()

# IMPORTANT: set these to exactly the same values used in training
cfg["filepath"] = "/path/to/train_data.json"
cfg["mode"] = "train"
cfg["data_frac"] = 1.0          # or the exact fraction you used
cfg["subsample_seed"] = 33      # or the exact seed you used

# If you used these during training, set them too:
# cfg["uids_trainval"] = ...
# cfg["batch_size"] = ...
# cfg["permute_repeat"] = ...

train_dl, _, _, _ = build_dataloaders(cfg)
base_train_ds = train_dl.dataset.base if hasattr(train_dl.dataset, "base") else train_dl.dataset

index_to_uid = dict(base_train_ds.index_to_uid)

state = torch.load(ckpt_path, map_location="cpu")
sd = state["model_state_dict"]

mix_logits = sd["projection.output_head.user_mix_logits.weight"].detach().cpu()
mix_weights = torch.softmax(mix_logits, dim=-1)

rows = []
num_users, num_heads = mix_weights.shape

for user_index in range(num_users):
    row = {
        "user_index": user_index,
        "uid": index_to_uid.get(user_index, "[MISSING]"),
    }
    for h in range(num_heads):
        row[f"mix_head_{h}_logit"] = float(mix_logits[user_index, h])
        row[f"mix_head_{h}_weight"] = float(mix_weights[user_index, h])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(df.head())
print(f"saved to {out_csv}")