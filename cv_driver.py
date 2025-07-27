import ray, pandas as pd, boto3
from train_fold import run_single_fold

ray.init(address="auto")
@ray.remote(num_gpus=1)
def _one(fold): return run_single_fold(fold, "s3://productgptbucket/CV/folds.json")

results = ray.get([_one.remote(f) for f in range(10)])
pd.DataFrame(results).to_csv("cv_metrics.csv", index=False)
boto3.client("s3").upload_file("cv_metrics.csv",
                               "productgptbucket",
                               "CV/tables/cv_metrics.csv")
