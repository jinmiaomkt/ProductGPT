import ray, subprocess, json, pandas as pd, boto3, pathlib, os

# ───────── CONFIG ────────────────────────────────────────────────────
TRAIN  = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py"
BUCKET = "productgptbucket"
PREFIX = "CV_GRU"

# single location for common paths
DATA_TRAIN = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
CKPT_DUMMY = "/tmp/ckpt_dummy.pt"                       # never read, but flag needed
OUT_DUMMY  = "/tmp/out_dummy.txt"                      # never read, but flag needed
HIDDEN     = "128"                                     # matches gru_h128_…
INPUT_DIM  = "15"
BATCH_SIZE = "4"

# ───────── RAY SETUP ─────────────────────────────────────────────────
ray.init(address="auto")

@ray.remote(num_gpus=1)
def run_fold(k: int):
    """Train and upload one fold; return its metrics dict."""
    cmd = [
        "python3", TRAIN,
        "--model",       "gru",
        "--fold",        str(k),
        "--bucket",      BUCKET,
        "--data",        DATA_TRAIN,     # ↓ four required legacy flags
        "--ckpt",        CKPT_DUMMY,
        "--hidden_size", HIDDEN,
        "--out",         OUT_DUMMY,
        "--input_dim",   INPUT_DIM,
        "--batch_size",  BATCH_SIZE,
    ]
    subprocess.check_call(cmd)

    # download the metrics JSON that the training script uploaded
    name   = f"gru_h{HIDDEN}_lr0.0001_bs{BATCH_SIZE}_fold{k}.json"
    local  = pathlib.Path(name)
    subprocess.check_call(
        ["aws", "s3", "cp",
         f"s3://{BUCKET}/{PREFIX}/metrics/{name}", str(local)]
    )
    return json.loads(local.read_text())

# ───────── SCHEDULE 10 FOLDS ────────────────────────────────────────
metrics = ray.get([run_fold.remote(k) for k in range(10)])

# ───────── AGGREGATE & UPLOAD TABLES ─────────────────────────────────
df = pd.DataFrame(metrics).set_index("fold")
df.to_csv("cv_gru_metrics.csv")

summary = df.select_dtypes(float).agg(["mean", "std"]).round(4)
summary.to_csv("cv_gru_summary.csv")
summary.to_json("cv_gru_summary.json", indent=2)

s3 = boto3.client("s3")
s3.upload_file("cv_gru_metrics.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_metrics.csv")
s3.upload_file("cv_gru_summary.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_summary.csv")
s3.upload_file("cv_gru_summary.json", BUCKET, f"{PREFIX}/tables/cv_gru_summary.json")
print("[✓] GRU CV complete and summaries uploaded.")
