# save this as cv_split_upload.py and run: python3 cv_split_upload.py
import json, boto3
from sklearn.model_selection import GroupKFold
from pathlib import Path

# Load UID list from your label file
label_path = Path("/home/ec2-user/data/clean_list_int_wide4_simple6.json")
records = json.loads(label_path.read_text())

# Extract UID strings
uids = [str(r["uid"][0] if isinstance(r["uid"], list) else r["uid"]) for r in records]

# Assign 10 folds using GroupKFold
gkf = GroupKFold(n_splits=10)
fold_map = {}
for fold, (_, test_idx) in enumerate(gkf.split(uids, groups=uids)):
    for u in [uids[i] for i in test_idx]:
        fold_map[u] = fold

# Create fold spec dictionary
spec = {
    "n_splits": 10,
    "assignment": fold_map,
}

# Save locally
local_json = "productgptfolds.json"
with open(local_json, "w") as f:
    json.dump(spec, f, indent=2)
print(f"[✓] Fold spec saved locally as {local_json}")

# Upload to S3 (you can change the bucket and key prefix)
bucket = "productgptbucket"
key    = "CV/folds.json"           # <- change this line
boto3.client("s3").upload_file(local_json, bucket, key)
print(f"[↑] Uploaded to s3://{bucket}/{key}")