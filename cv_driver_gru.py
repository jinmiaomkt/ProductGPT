import ray, subprocess, json, pandas as pd, boto3, os, pathlib

TRAIN = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py" 
BUCKET= "productgptbucket"
PREFIX= "CV_GRU"

ray.init(address="auto")

@ray.remote(num_gpus=1)
def run_fold(k):
    subprocess.check_call(["python3", TRAIN,
                           "--model", "gru", "--fold", str(k),
                           "--bucket", BUCKET])
    # read the metrics json we just wrote
    name = f"gru_h128_lr0.0001_bs4_fold{k}.json"
    local = pathlib.Path(name)
    subprocess.check_call(["aws","s3","cp",
                           f"s3://{BUCKET}/{PREFIX}/metrics/{name}", str(local)])
    return json.loads(local.read_text())

metrics = ray.get([run_fold.remote(k) for k in range(10)])
df = pd.DataFrame(metrics).set_index("fold")
df.to_csv("cv_gru_metrics.csv")

# mean ± std
summary = df.select_dtypes(float).agg(["mean","std"]).round(4)
summary.to_csv("cv_gru_summary.csv")
summary.to_json("cv_gru_summary.json", indent=2)

s3 = boto3.client("s3")
s3.upload_file("cv_gru_metrics.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_metrics.csv")
s3.upload_file("cv_gru_summary.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_summary.csv")
s3.upload_file("cv_gru_summary.json", BUCKET, f"{PREFIX}/tables/cv_gru_summary.json")
print("[✓] GRU CV complete and summaries uploaded.")
