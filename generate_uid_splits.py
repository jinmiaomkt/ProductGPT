#!/usr/bin/env python3
"""
generate_uid_splits.py
Run once to create /tmp/uids_val.txt and /tmp/uids_test.txt
"""
import json, random, boto3

s3 = boto3.client("s3")
body = s3.get_object(Bucket="productgptbucket", Key="folds/productgptfolds.json")["Body"].read()
spec = json.loads(body)

fold_id = 0
uids_test = [u for u, f in spec["assignment"].items() if f == fold_id]
uids_trainval = [u for u in spec["assignment"] if u not in uids_test]

rng = random.Random(33)
rng.shuffle(uids_trainval)
n_val = max(1, int(0.1 * len(uids_trainval)))
uids_val = uids_trainval[:n_val]

with open("/tmp/uids_val.txt", "w") as f:
    f.write("\n".join(uids_val))
with open("/tmp/uids_test.txt", "w") as f:
    f.write("\n".join(uids_test))

print(f"Val UIDs:  {len(uids_val)} → /tmp/uids_val.txt")
print(f"Test UIDs: {len(uids_test)} → /tmp/uids_test.txt")
print(f"Train UIDs: {len(uids_trainval) - len(uids_val)}")