import os
import json
import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------------
# 0) Hyperparameters & paths
# ------------------------------------------------------------------
BATCH_SIZE     = 4
EPOCHS         = 20
INPUT_DIM      = 15
HIDDEN_SIZE    = 32
NUM_CLASSES    = 10       # classes 0..9 (0 = PAD/ignore)
CLASS_9_WEIGHT = 5.0

JSON_PATH      = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
S3_BUCKET      = "productgptbucket"
S3_PREFIX      = "LSTM"   # S3 folder

# ------------------------------------------------------------------
# 1) Dataset + collate
# ------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            records = json.load(f)

        self.x_seqs = []
        self.y_seqs = []
        for row in records:
            flat = [0 if t == "NA" else int(t)
                    for t in row["AggregateInput"][0].split()]
            T    = len(flat) // INPUT_DIM
            x    = torch.tensor(flat, dtype=torch.float32).view(T, INPUT_DIM)

            dec  = [0 if t == "NA" else int(t)
                    for t in row["Decision"][0].split()]
            valid = min(T, len(dec)) - 1
            y    = torch.tensor(dec[1:valid+1], dtype=torch.long)

            self.x_seqs.append(x[:valid])
            self.y_seqs.append(y)

    def __len__(self):
        return len(self.x_seqs)

    def __getitem__(self, idx):
        return self.x_seqs[idx], self.y_seqs[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return x_pad, y_pad

# ------------------------------------------------------------------
# 2) DataLoaders with 80:10:10 split
# ------------------------------------------------------------------
dataset = SequenceDataset(JSON_PATH)
n = len(dataset)
train_size = int(0.8 * n)
val_size   = int(0.1 * n)
test_size  = n - train_size - val_size

seed_value = 33
torch.manual_seed(seed_value)

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
# test_ds is created but not evaluated here

# ------------------------------------------------------------------
# 3) Model
# ------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_SIZE, batch_first=True)
        self.fc   = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)            # (B, T, HIDDEN_SIZE)
        return self.fc(out)              # (B, T, NUM_CLASSES)

# ------------------------------------------------------------------
# 4) Evaluation (only on val_loader)
# ------------------------------------------------------------------
def evaluate(loader, model, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_ppl  = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Evaluating"):
            x = x_batch.to(device)        # (B, T, 15)
            y = y_batch.to(device)        # (B, T)

            logits = model(x)             # (B, T, C)
            B, T, C = logits.size()

            flat_logits = logits.reshape(-1, C)   # (B*T, C)
            flat_labels = y.reshape(-1)           # (B*T,)

            loss = loss_fn(flat_logits, flat_labels)
            total_loss += loss.item()

            probs = F.softmax(flat_logits, dim=-1)
            true_p = probs[torch.arange(len(flat_labels)), flat_labels]
            ppl = torch.exp(-torch.log(true_p + 1e-9).mean()).item()
            total_ppl += ppl

            preds = probs.argmax(dim=-1).cpu().numpy()
            labs  = flat_labels.cpu().numpy()
            mask  = labs != 0                     # ignore PAD=0
            all_preds.append(preds[mask])
            all_labels.append(labs[mask])
            all_probs.append(probs.cpu().numpy()[mask, :])

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)

    avg_loss = total_loss / len(loader)
    avg_ppl  = total_ppl  / len(loader)

    cls_ids   = np.arange(1, NUM_CLASSES)
    conf_mat  = confusion_matrix(all_labels, all_preds, labels=cls_ids)
    hit_rate  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="macro")

    y_bin  = label_binarize(all_labels, classes=cls_ids)
    auprc  = average_precision_score(y_bin, all_probs[:,1:], average="macro")

    return avg_loss, conf_mat, avg_ppl, hit_rate, f1, auprc

# ------------------------------------------------------------------
# 5) Train & validate
# ------------------------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = LSTMClassifier().to(device)
weights   = torch.ones(NUM_CLASSES, device=device)
weights[9] = CLASS_9_WEIGHT
loss_fn   = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        x = x_batch.to(device)
        y = y_batch.to(device)

        logits = model(x).reshape(-1, NUM_CLASSES)
        labels = y.reshape(-1)

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    val_loss, val_conf_mat, val_ppl, val_hit, val_f1, val_auprc = evaluate(
        val_loader, model, device, loss_fn
    )

    print(f"\nEpoch {epoch}")
    print(f"Train Loss={avg_train_loss:.4f}")
    print(f"Val   Loss={val_loss:.4f}")
    print(f"Val   PPL={val_ppl:.4f}")
    print(f"Val   Hit Rate={val_hit:.4f}")
    print(f"Val   F1 Score={val_f1:.4f}")
    print(f"Val   Area Under Precision-Recall Curve={val_auprc:.4f}")
    print("Val Confusion Matrix:\n", val_conf_mat)

# ------------------------------------------------------------------
# 6) Save & upload
# ------------------------------------------------------------------
torch.save(model.state_dict(), "model.pt")
metrics = {
    "val_loss": val_loss,
    "val_ppl":  val_ppl,
    "val_hit":  val_hit,
    "val_f1":   val_f1,
    "val_auprc":val_auprc
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

s3 = boto3.client("s3")
for fn in ["model.pt", "metrics.json"]:
    s3.upload_file(fn, S3_BUCKET, f"{S3_PREFIX}/{fn}")
    print(f"Uploaded {fn} to s3://{S3_BUCKET}/{S3_PREFIX}/{fn}")
