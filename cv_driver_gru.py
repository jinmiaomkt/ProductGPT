import ray, pandas as pd, boto3, os, json

ray.init(address="auto")
@ray.remote(num_gpus=1)

def _one(f): 
    os.system(f"python3 train_gru_lstm.py --model gru  --fold {f}")
    return json.loads(open("m.json").read())

rows = ray.get([_one.remote(f) for f in range(10)])

pd.DataFrame(rows).to_csv("cv_gru_metrics.csv", index=False)

boto3.client("s3").upload_file("cv_gru_metrics.csv",
      "productgptbucket", "CV_GRU/cv_gru_metrics.csv")