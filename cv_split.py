# cv_split.py ---------------------------------------------------------
from sklearn.model_selection import GroupKFold
import json, uuid, boto3, pathlib

def make_folds(uids, n_splits=10, bucket=None, key_prefix="ProductGPT/folds/"):
    gkf = GroupKFold(n_splits)
    fold_map = {}
    for fold, (_, test_idx) in enumerate(gkf.split(uids, groups=uids)):
        for u in uids[test_idx]:
            fold_map[u] = fold

    obj = {"n_splits": n_splits, "assignment": fold_map}
    fname = f"folds_{uuid.uuid4().hex[:8]}.json"

    if bucket:                              # push to S3
        boto3.client("s3").put_object(
            Bucket=bucket, Key=f"{key_prefix}{fname}",
            Body=json.dumps(obj).encode()
        )
        return f"s3://{bucket}/{key_prefix}{fname}"
    else:                                   # local file
        path = pathlib.Path(fname).write_text(json.dumps(obj, indent=2))
        return str(path)
