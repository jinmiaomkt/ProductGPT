import ray, pandas as pd, boto3, io
from botocore.exceptions import ClientError
from train_fold import run_single_fold

S3_BUCKET = "productgptbucket"
S3_KEY    = "CV/tables/cv_metrics.csv"
SPEC_URI  = "s3://productgptbucket/CV/folds.json"
FOLDS     = list(range(2, 10))   # 2..9

# Try to connect to a cluster; otherwise start local
try:
    ray.init(address="auto", ignore_reinit_error=True)
except Exception:
    ray.init(ignore_reinit_error=True)
@ray.remote(num_gpus=1)

def _one(fold):
    res = run_single_fold(fold, SPEC_URI)
    # Ensure fold id is present in the record we save
    if isinstance(res, dict):
        res = {**res, "fold_id": fold}
    return res

# Launch jobs
results = ray.get([_one.remote(f) for f in FOLDS])
new_df  = pd.DataFrame(results)

# Append/merge with existing CSV on S3
s3 = boto3.client("s3")
try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    old_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    out_df = pd.concat([old_df, new_df], ignore_index=True)

    # Drop duplicates by fold_id if present (keep latest)
    if "fold_id" in out_df.columns:
        out_df = (out_df
                  .sort_values(by=["fold_id"])
                  .drop_duplicates(subset=["fold_id"], keep="last"))
except ClientError as e:
    if e.response.get("Error", {}).get("Code") == "NoSuchKey":
        out_df = new_df
    else:
        raise

# Upload merged CSV back to S3
buf = io.BytesIO()
out_df.to_csv(buf, index=False)
buf.seek(0)
s3.upload_fileobj(buf, S3_BUCKET, S3_KEY)
print(f"Uploaded updated metrics to s3://{S3_BUCKET}/{S3_KEY}")

ray.shutdown()