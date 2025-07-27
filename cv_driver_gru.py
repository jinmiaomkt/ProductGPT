import ray, subprocess, json, pandas as pd, boto3, pathlib, os

# ───── constants for the GRU run ────────────────────────────────────
TRAIN       = "/home/ec2-user/ProductGPT/train_gru_lstm_new.py"
BUCKET      = "productgptbucket"
PREFIX      = "CV_GRU"

MODEL_NAME  = "gru"
HIDDEN      = "128"
LR_STR      = "0.0001"          # used only in filenames
BATCH_SIZE  = "4"
INPUT_DIM   = "15"

DATA_TRAIN  = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
CKPT_DUMMY  = "/tmp/ckpt_dummy.pt"
OUT_DUMMY   = "/tmp/out_dummy.txt"

ray.init(address="auto")

@ray.remote(num_gpus=1)
def run_fold(k: int):
    cmd = [
        "python3", TRAIN,
        "--model", MODEL_NAME, "--fold", str(k), "--bucket", BUCKET,
        "--data", DATA_TRAIN,
        "--ckpt", CKPT_DUMMY,
        "--hidden_size", HIDDEN,
        "--out", OUT_DUMMY,
        "--input_dim", INPUT_DIM,
        "--batch_size", BATCH_SIZE,
    ]
    subprocess.check_call(cmd)

    json_key = f"{MODEL_NAME}_h{HIDDEN}_lr{LR_STR}_bs{BATCH_SIZE}_fold{k}.json"
    local    = pathlib.Path(json_key)
    subprocess.check_call(["aws", "s3", "cp",
                           f"s3://{BUCKET}/{PREFIX}/metrics/{json_key}", str(local)])
    return json.loads(local.read_text())

metrics = ray.get([run_fold.remote(k) for k in range(10)])
df      = pd.DataFrame(metrics).set_index("fold")
df.to_csv("cv_gru_metrics.csv")

summary = df.select_dtypes(float).agg(["mean","std"]).round(4)
summary.to_csv("cv_gru_summary.csv")
summary.to_json("cv_gru_summary.json", indent=2)

s3 = boto3.client("s3")
s3.upload_file("cv_gru_metrics.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_metrics.csv")
s3.upload_file("cv_gru_summary.csv",  BUCKET, f"{PREFIX}/tables/cv_gru_summary.csv")
s3.upload_file("cv_gru_summary.json", BUCKET, f"{PREFIX}/tables/cv_gru_summary.json")
print("[✓] GRU 10‑fold CV finished and summaries uploaded.")
