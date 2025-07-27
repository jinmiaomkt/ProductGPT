# train_fold.py -------------------------------------------------------
import json, boto3, os
from pathlib import Path

from train4_decoderonly_performer_feature_aws import train_model   # ← your existing code
from config4 import get_config, get_weights_file_path, latest_weights_file_path
from dataset4_productgpt import load_json_dataset      # ← already imported above

def run_single_fold(fold_id, spec_uri):
    spec = json.loads(boto3.client("s3").get_object(
        Bucket="productgptbucket", Key="CV/folds.json")["Body"].read())
    test_uids = [u for u,f in spec["assignment"].items() if f==fold_id]
    train_uids= [u for u in spec["assignment"] if u not in test_uids]

    cfg = get_config()
    cfg.update({"fold_id": fold_id,
                "uids_test": test_uids,
                
                "uids_trainval": train_uids,
                "mode": "train"})
    return train_model(cfg)
