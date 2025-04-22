# # import os
# # import json
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from sklearn.metrics import (
# #     confusion_matrix,
# #     accuracy_score,
# #     f1_score,
# #     average_precision_score
# # )
# # from sklearn.preprocessing import label_binarize

# # # ------------------------------------------------------------------
# # # 1) Load & parse your reorganized JSON
# # # ------------------------------------------------------------------
# # def load_json_dataset(filepath):
# #     with open(filepath, 'r') as f:
# #         return json.load(f)

# # def parse_space_separated_ints(string_list):
# #     """Convert ["1 2 3 NA 4 ..."] → [1,2,3,0,4,...]."""
# #     toks = string_list[0].split()
# #     return [0 if t == "NA" else int(t) for t in toks]

# # records = load_json_dataset("/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json")

# # X_list, y_list = [], []
# # for row in records:
# #     # each AggregateInput is 15 * T ints flattened
# #     flat = parse_space_separated_ints(row["AggregateInput"])
# #     T = len(flat) // 15
# #     agg_seq = np.array(flat, dtype="float32").reshape(T, 15)

# #     dec_seq = parse_space_separated_ints(row["Decision"])
# #     # we predict dec_seq[t+1] from agg_seq[t], so drop last one
# #     valid = min(T, len(dec_seq)) - 1
# #     X_list.append( agg_seq[:valid] )
# #     y_list.append( np.array(dec_seq[1 : valid+1], dtype="int32") )

# # # ------------------------------------------------------------------
# # # 2) Pad all sequences to the same length (post‑padding)
# # # ------------------------------------------------------------------
# # max_len = max(x.shape[0] for x in X_list)

# # X_padded = pad_sequences(
# #     X_list,
# #     maxlen=max_len,
# #     dtype="float32",
# #     padding="post",
# #     value=0.0,
# # )  # shape = (batch, max_len, 15)

# # y_padded = pad_sequences(
# #     y_list,
# #     maxlen=max_len,
# #     dtype="int32",
# #     padding="post",
# #     value=-1,
# # )  # shape = (batch, max_len)

# # # ------------------------------------------------------------------
# # # 3) Build & compile the LSTM model
# # # ------------------------------------------------------------------
# # num_classes = 9  # adjust to your actual # of decision labels

# # model = Sequential([
# #     Masking(mask_value=0.0, input_shape=(max_len, 15)),
# #     LSTM(32, return_sequences=True),
# #     TimeDistributed(Dense(num_classes, activation="softmax"))
# # ])

# # model.compile(
# #     loss="sparse_categorical_crossentropy",
# #     optimizer="adam",
# #     metrics=["accuracy"]
# # )

# # # ------------------------------------------------------------------
# # # 4) Train
# # # ------------------------------------------------------------------
# # model.fit(
# #     X_padded,
# #     y_padded,
# #     batch_size=2,
# #     epochs=5,
# # )

# # # ------------------------------------------------------------------
# # # 5) Evaluation function
# # # ------------------------------------------------------------------
# # def evaluate_model(model, X, y, pad_token=-1):
# #     """
# #     Returns:
# #       avg_loss, conf_mat, avg_ppl, accuracy, macro_f1, auprc
# #     """
# #     # 1) get full softmax probabilities: (batch, T, C)
# #     probs = model.predict(X, verbose=0)
# #     B, T, C = probs.shape

# #     # 2) flatten and mask out padding
# #     preds = probs.argmax(axis=-1).reshape(-1)      # shape = (B*T,)
# #     labels = y.reshape(-1)                         # shape = (B*T,)
# #     mask   = labels != pad_token                   # ignore padded steps

# #     preds = preds[mask]
# #     labels = labels[mask]
# #     probs = probs.reshape(-1, C)[mask]

# #     # 3) compute average loss & perplexity manually
# #     #    loss = -mean(log p_true)
# #     true_probs = probs[np.arange(len(labels)), labels]
# #     eps = 1e-9
# #     log_probs = np.log(true_probs + eps)
# #     avg_loss = - log_probs.mean()
# #     avg_ppl  = float(np.exp(avg_loss))

# #     # 4) classic metrics
# #     decision_ids = np.arange(C)  # or narrow to the classes you care about
# #     conf_mat = confusion_matrix(labels, preds, labels=decision_ids)
# #     acc       = accuracy_score(labels, preds)
# #     macro_f1  = f1_score(labels, preds, average="macro")

# #     # 5) average precision (AUPRC)
# #     #    need one-hot of labels
# #     y_bin = label_binarize(labels, classes=decision_ids)
# #     # and probs for each class
# #     auprc = average_precision_score(y_bin, probs, average="macro")

# #     return avg_loss, conf_mat, avg_ppl, acc, macro_f1, auprc

# # # ------------------------------------------------------------------
# # # 6) Run evaluation & print
# # # ------------------------------------------------------------------
# # avg_loss, conf_mat, ppl, acc, f1, auprc = evaluate_model(model, X_padded, y_padded)
# # print(f"Avg loss:      {avg_loss:.4f}")
# # print(f"Perplexity:    {ppl:.4f}")
# # print(f"Accuracy:      {acc:.4f}")
# # print(f"Macro F1:      {f1:.4f}")
# # print(f"Macro AUPRC:   {auprc:.4f}")
# # print("Confusion matrix:")
# # print(conf_mat)
# import os
# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Masking, LSTM, TimeDistributed, Dense
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     f1_score,
#     average_precision_score,
# )
# from sklearn.preprocessing import label_binarize

# # ------------------------------------------------------------------
# # Hyperparams & paths
# # ------------------------------------------------------------------
# BATCH_SIZE     = 2
# EPOCHS         = 20
# NUM_CLASSES    = 10    # now 0–9
# CLASS_9_WEIGHT = 5.0

# JSON_PATH = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"

# # ------------------------------------------------------------------
# # 1) Load & parse JSON
# # ------------------------------------------------------------------
# def load_json_dataset(fp):
#     with open(fp, "r") as f:
#         return json.load(f)

# def parse_space_separated_ints(sl):
#     toks = sl[0].split()
#     return [0 if t=="NA" else int(t) for t in toks]

# records = load_json_dataset(JSON_PATH)

# X_list, y_list = [], []
# for row in records:
#     flat = parse_space_separated_ints(row["AggregateInput"])
#     T = len(flat)//15
#     agg = np.array(flat, dtype="float32").reshape(T, 15)

#     dec = parse_space_separated_ints(row["Decision"])
#     valid = min(T, len(dec)) - 1
#     X_list.append( agg[:valid] )
#     # decisions[t] ∈ {1..9}; we'll use 0 as “pad”
#     y_list.append( np.array(dec[1:valid+1], dtype="int32") )

# # ------------------------------------------------------------------
# # 2) Post‑pad with 0
# # ------------------------------------------------------------------
# max_len = max(x.shape[0] for x in X_list)

# X = pad_sequences(X_list, maxlen=max_len, dtype="float32",
#                   padding="post", value=0.0)
# y = pad_sequences(y_list, maxlen=max_len, dtype="int32",
#                   padding="post", value=0)   # pad=0

# # build sample_weight: class 9 → heavier; pad (0) → zero
# sw = np.ones_like(y, dtype="float32")
# sw[y==9]  = CLASS_9_WEIGHT
# sw[y==0]  = 0.0

# # ------------------------------------------------------------------
# # 3) Build & compile
# # ------------------------------------------------------------------
# model = Sequential([
#     Masking(mask_value=0.0, input_shape=(max_len, 15)),
#     LSTM(32, return_sequences=True),
#     TimeDistributed(Dense(NUM_CLASSES, activation="softmax"))
# ])
# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# # ------------------------------------------------------------------
# # 4) Train
# # ------------------------------------------------------------------
# model.fit(
#     X, y,
#     sample_weight=sw,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     verbose=1,
# )

# # ------------------------------------------------------------------
# # 5) Evaluation (ignore class 0)
# # ------------------------------------------------------------------
# def evaluate_model(m, X, y, pad_token=0):
#     probs = m.predict(X, verbose=1)    # (B, T, 10)
#     B,T,C = probs.shape

#     preds  = probs.argmax(axis=-1).reshape(-1)
#     labels = y.reshape(-1)
#     mask   = labels != pad_token      # drop all 0s

#     preds  = preds[mask]
#     labels = labels[mask]
#     probs2 = probs.reshape(-1, C)[mask]

#     # avg loss & ppl
#     tp = probs2[np.arange(len(labels)), labels]
#     lp = np.log(tp + 1e-9)
#     loss = - lp.mean()
#     ppl  = float(np.exp(loss))

#     # metrics on classes 1–9
#     cls = np.arange(1, C)
#     cm     = confusion_matrix(labels, preds, labels=cls)
#     acc    = accuracy_score(labels, preds)
#     f1     = f1_score(labels, preds, average="macro")
    
#     ybin = label_binarize(labels, classes=cls)
#     auprc= average_precision_score(ybin, probs2[:,1:], average="macro")

#     return loss, cm, ppl, acc, f1, auprc

# l, cm, ppl, acc, f1, auprc = evaluate_model(model, X, y)
# print(f"Loss: {l:.4f}, PPL: {ppl:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}")
# print("Confusion matrix for classes 1–9:\n", cm)

import os
import json
import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
NUM_CLASSES    = 10        # classes 0..9 (0 will be the PAD/ignore class)
CLASS_9_WEIGHT = 5.0
STEP_SIZE      = INPUT_DIM # we predict one decision per block of 15 tokens

JSON_PATH      = "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json"
S3_BUCKET      = "your-s3-bucket"
S3_PREFIX      = "ProductGPT/LSTM"  # S3 folder

# ------------------------------------------------------------------
# 1) Dataset definition
# ------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            records = json.load(f)

        self.x_seqs = []
        self.y_seqs = []
        for row in records:
            flat   = [0 if t=="NA" else int(t)
                      for t in row["AggregateInput"][0].split()]
            T      = len(flat) // INPUT_DIM
            x_seq  = torch.tensor(flat, dtype=torch.float32) \
                          .view(T, INPUT_DIM)

            dec    = [0 if t=="NA" else int(t)
                      for t in row["Decision"][0].split()]
            valid  = min(T, len(dec)) - 1
            y_seq  = torch.tensor(dec[1:valid+1], dtype=torch.long)

            self.x_seqs.append(x_seq[:valid])
            self.y_seqs.append(y_seq)

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
# 2) DataLoaders
# ------------------------------------------------------------------
dataset = SequenceDataset(JSON_PATH)
# simple split 80/20
n_train = int(0.8 * len(dataset))
train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn)

# ------------------------------------------------------------------
# 3) Model definition
# ------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_SIZE, batch_first=True)
        self.fc   = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x: (B, T, INPUT_DIM)
        out, _ = self.lstm(x)
        # out: (B, T, HIDDEN_SIZE)
        return self.fc(out)  # -> (B, T, NUM_CLASSES)

# ------------------------------------------------------------------
# 4) Evaluation function
# ------------------------------------------------------------------
def evaluate(dataloader, model, device, loss_fn, stepsize):
    model.eval()
    total_loss = 0.0
    total_ppl  = 0.0

    all_preds  = []
    all_labels = []
    all_probs  = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Evaluating"):
            x_batch = x_batch.to(device)       # (B, T, 15)
            y_batch = y_batch.to(device)       # (B, T)

            logits = model(x_batch)            # (B, T, C)

            # pick positions where decision occurs
            pos = torch.arange(stepsize-1, logits.size(1), stepsize,
                               device=logits.device)
            dec_logits = logits[:, pos, :]     # (B, N, C)
            dec_labels = y_batch[:, pos]       # (B, N)

            B, N, C = dec_logits.shape
            flat_logits = dec_logits.reshape(-1, C)
            flat_labels = dec_labels.reshape(-1)

            loss = loss_fn(flat_logits, flat_labels)
            total_loss += loss.item()

            # perplexity
            probs = F.softmax(flat_logits, dim=-1)
            true_p = probs[torch.arange(len(flat_labels)), flat_labels]
            ppl = torch.exp(-torch.log(true_p + 1e-9).mean()).item()
            total_ppl += ppl

            preds = probs.argmax(dim=-1).cpu().numpy()
            labs  = flat_labels.cpu().numpy()

            # mask out PAD=0
            mask = labs != 0
            preds = preds[mask]
            labs  = labs[mask]
            probs = probs.cpu().numpy()[mask, :]

            all_preds.append(preds)
            all_labels.append(labs)
            all_probs.append(probs)

    # aggregate
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)

    avg_loss = total_loss / len(dataloader)
    avg_ppl  = total_ppl  / len(dataloader)

    cls_ids = np.arange(1, NUM_CLASSES)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=cls_ids)
    hit_rate = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")

    y_bin    = label_binarize(all_labels, classes=cls_ids)
    auprc    = average_precision_score(y_bin, all_probs[:,1:], average="macro")

    return avg_loss, conf_mat, avg_ppl, hit_rate, f1, auprc

# ------------------------------------------------------------------
# 5) Setup and train loop
# ------------------------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = LSTMClassifier().to(device)

class_weights = torch.ones(NUM_CLASSES, device=device)
class_weights[9] = CLASS_9_WEIGHT
loss_fn  = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
optim    = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        B, T, C = logits.size()
        flat_logits = logits.reshape(-1, C)
        flat_labels = y_batch.reshape(-1)

        loss = loss_fn(flat_logits, flat_labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Evaluate on val set
    val_loss, val_conf_mat, val_ppl, val_hit, val_f1, val_auprc = evaluate(
        val_loader, model, device, loss_fn, STEP_SIZE
    )
    print(f"\nEpoch {epoch}")
    print(f"Train Loss:           {avg_train_loss:.4f}")
    print(f"Val   Loss:           {val_loss:.4f}")
    print(f"Val   PPL:            {val_ppl:.4f}")
    print(f"Val   Hit Rate:       {val_hit:.4f}")
    print(f"Val   Macro F1:       {val_f1:.4f}")
    print(f"Val   AUPRC:          {val_auprc:.4f}")
    print("Val Confusion Matrix:\n", val_conf_mat)

# ------------------------------------------------------------------
# 6) Save model & metrics
# ------------------------------------------------------------------
torch.save(model.state_dict(), "model.pt")
metrics = {
    "val_loss":    val_loss,
    "val_ppl":     val_ppl,
    "val_hit":     val_hit,
    "val_f1":      val_f1,
    "val_auprc":   val_auprc
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ------------------------------------------------------------------
# 7) Upload to S3
# ------------------------------------------------------------------
s3 = boto3.client("s3")
for fn in ["model.pt", "metrics.json"]:
    dest = f"{S3_PREFIX}/{fn}"
    s3.upload_file(fn, S3_BUCKET, dest)
    print(f"Uploaded {fn} to s3://{S3_BUCKET}/{dest}")

