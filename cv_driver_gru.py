# cv_driver_gru.py  ────────────────────────────────────────────────
import ray, pandas as pd, boto3, json
from train_fold_gru import run_single_fold

ray.init(address="auto")          # one running Ray head, 1 GPU total

@ray.remote(num_gpus=1)
def _one(fold):        # exactly like ProductGPT wrapper
    return run_single_fold(fold)

# ---- schedule folds SEQUENTIALLY on a 1‑GPU node ---------------
metrics = []
for k in range(10):
    metrics.append(ray.get(_one.remote(k)))
    print(f"✓ fold{k} finished")

# ---- aggregate + upload ----------------------------------------
df = pd.DataFrame(metrics).set_index("fold")
df.to_csv("cv_gru_metrics.csv")

summary = df.select_dtypes(float).agg(["mean", "std"]).round(4)
summary.to_csv("cv_gru_summary.csv")
summary.to_json("cv_gru_summary.json", indent=2)

s3 = boto3.client("s3")
s3.upload_file("cv_gru_metrics.csv",  "productgptbucket",
               "CV_GRU/tables/cv_gru_metrics.csv")
s3.upload_file("cv_gru_summary.csv",  "productgptbucket",
               "CV_GRU/tables/cv_gru_summary.csv")
s3.upload_file("cv_gru_summary.json", "productgptbucket",
               "CV_GRU/tables/cv_gru_summary.json")
print("✓ GRU CV complete; summaries uploaded.")
