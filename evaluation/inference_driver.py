# inference_driver.py  (run on an EC2 node with one GPU)

import json, boto3, torch, tqdm
from pathlib import Path
from train4_decoderonly_performer_feature_aws import build_model, load_feature_tensor
from dataset4_productgpt import load_json_dataset
from config4 import get_config
from sklearn.metrics import f1_score, average_precision_score

BUCKET   = "productgptbucket"
CK_KEY   = "FullProductGPT/performer/FeatureBased/checkpoints/FullProductGPT_featurebased_…_fold0.pt"
VAL_UIDS = set(json.loads(Path("val_uids.json").read_text()))
TEST_UIDS= set(json.loads(Path("test_uids.json").read_text()))

# 1) download checkpoint
s3 = boto3.client("s3")
local_ckpt = "/tmp/best_fold0.pt"
s3.download_file(BUCKET, CK_KEY, local_ckpt)

# 2) rebuild model
cfg = get_config()                     # same as training
model = build_model(cfg, load_feature_tensor(FEAT_FILE))
state = torch.load(local_ckpt, map_location="cuda")
model.load_state_dict(state["model_state_dict"])
model.eval()

# 3) data = ALL 30 campaigns
data = load_json_dataset(cfg["filepath"], keep_uids=None)
dl   = DataLoader(data, batch_size=64)

rows, val_logits, test_logits, val_labels, test_labels = [], [], [], [], []
with torch.no_grad():
    for batch in tqdm.tqdm(dl):
        x    = batch["aggregate_input"].cuda()
        uid  = batch["uid"]                 # <- include uid in Dataset.__getitem__
        lab  = batch["label"].numpy()
        prob = model(x).softmax(-1).cpu().numpy()

        # stream full‑file output
        for u,p in zip(uid, prob):
            rows.append({"uid": u, "probs": p.tolist()})

        # stash val / test for metrics
        for u,p,l in zip(uid, prob, lab):
            if u in VAL_UIDS:
                val_logits.append(p); val_labels.append(l)
            elif u in TEST_UIDS:
                test_logits.append(p); test_labels.append(l)

# 4) write all predictions
pred_path = Path("/tmp/full_predictions.jsonl")
with pred_path.open("w") as fp:
    for r in rows:
        fp.write(json.dumps(r) + "\n")
s3.upload_file(pred_path, BUCKET, "ProductGPT/predictions/full_predictions.jsonl")

# 5) compute metrics
def bucket_metrics(y, prob):
    y_bin = (y[:,None] == np.arange(1,10)).astype(int)
    return {
        "macro_f1":  f1_score(y, prob.argmax(1), average="macro"),
        "auprc":     average_precision_score(y_bin, prob[:,1:10], average="macro"),
    }

out = {
    "val":  bucket_metrics(np.array(val_labels),  np.stack(val_logits)),
    "test": bucket_metrics(np.array(test_labels), np.stack(test_logits)),
}

tmp_json = "/tmp/fold0_valtest_metrics.json"
Path(tmp_json).write_text(json.dumps(out, indent=2))
s3.upload_file(tmp_json, BUCKET, "ProductGPT/metrics/fold0_valtest_metrics.json")
