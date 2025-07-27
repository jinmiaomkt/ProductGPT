# train_fold.py -------------------------------------------------------
import json, boto3, os
from pathlib import Path

from train4_decoderonly_performer_feature_aws import train_model   # ← your existing code
from config4 import get_config, get_weights_file_path, latest_weights_file_path
from dataset4_productgpt import load_json_dataset      # ← already imported above

def run_single_fold(fold_id: int, fold_spec_uri: str):
    # 1 ) pull fold assignment
    if fold_spec_uri.startswith("s3://"):
        bucket, key = fold_spec_uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        spec = json.loads(body)
    else:
        spec = json.loads(Path(fold_spec_uri).read_text())

    # 2 ) restrict dataset
    cfg = get_config()
    cfg["fold_id"]      = fold_id
    cfg["run_name"]     = f"{cfg['exp_name']}_fold{fold_id}"
    cfg["uids_test"]    = [u for u,f in spec["assignment"].items() if f==fold_id]
    cfg["uids_trainval"]= [u for u in spec["assignment"] if u not in cfg["uids_test"]]

    # (optional) split train vs. val here if you want nested CV

    return train_model(cfg)                 # returns a metrics dict
