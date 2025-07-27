import ray, pandas as pd, boto3, os, json

ray.init(address="auto")
TRAIN = "/home/ec2-user/ProductGPT/train_gru_lstm.py"   # full path

@ray.remote(num_gpus=1)

def _one(fold):
    cmd = f"python3 {TRAIN} --model gru --fold {fold}"
    status = os.system(cmd)
    if status != 0:
        raise RuntimeError(f"Training failed for fold {fold}")
    return json.loads(open("m.json").read())

rows = ray.get([_one.remote(f) for f in range(10)])

pd.DataFrame(rows).to_csv("cv_gru_metrics.csv", index=False)

boto3.client("s3").upload_file("cv_gru_metrics.csv",
      "productgptbucket", "CV_GRU/cv_gru_metrics.csv")