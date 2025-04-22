import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------------
# 1) Load & parse your reorganized JSON
# ------------------------------------------------------------------
def load_json_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_space_separated_ints(string_list):
    """Convert ["1 2 3 NA 4 ..."] → [1,2,3,0,4,...]."""
    toks = string_list[0].split()
    return [0 if t == "NA" else int(t) for t in toks]

records = load_json_dataset("/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json")

X_list, y_list = [], []
for row in records:
    # each AggregateInput is 15 * T ints flattened
    flat = parse_space_separated_ints(row["AggregateInput"])
    T = len(flat) // 15
    agg_seq = np.array(flat, dtype="float32").reshape(T, 15)

    dec_seq = parse_space_separated_ints(row["Decision"])
    # we predict dec_seq[t+1] from agg_seq[t], so drop last one
    valid = min(T, len(dec_seq)) - 1
    X_list.append( agg_seq[:valid] )
    y_list.append( np.array(dec_seq[1 : valid+1], dtype="int32") )

# ------------------------------------------------------------------
# 2) Pad all sequences to the same length (post‑padding)
# ------------------------------------------------------------------
max_len = max(x.shape[0] for x in X_list)

X_padded = pad_sequences(
    X_list,
    maxlen=max_len,
    dtype="float32",
    padding="post",
    value=0.0,
)  # shape = (batch, max_len, 15)

y_padded = pad_sequences(
    y_list,
    maxlen=max_len,
    dtype="int32",
    padding="post",
    value=-1,
)  # shape = (batch, max_len)

# ------------------------------------------------------------------
# 3) Build & compile the LSTM model
# ------------------------------------------------------------------
num_classes = 9  # adjust to your actual # of decision labels

model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_len, 15)),
    LSTM(32, return_sequences=True),
    TimeDistributed(Dense(num_classes, activation="softmax"))
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# ------------------------------------------------------------------
# 4) Train
# ------------------------------------------------------------------
model.fit(
    X_padded,
    y_padded,
    batch_size=2,
    epochs=5,
)

# ------------------------------------------------------------------
# 5) Evaluation function
# ------------------------------------------------------------------
def evaluate_model(model, X, y, pad_token=-1):
    """
    Returns:
      avg_loss, conf_mat, avg_ppl, accuracy, macro_f1, auprc
    """
    # 1) get full softmax probabilities: (batch, T, C)
    probs = model.predict(X, verbose=0)
    B, T, C = probs.shape

    # 2) flatten and mask out padding
    preds = probs.argmax(axis=-1).reshape(-1)      # shape = (B*T,)
    labels = y.reshape(-1)                         # shape = (B*T,)
    mask   = labels != pad_token                   # ignore padded steps

    preds = preds[mask]
    labels = labels[mask]
    probs = probs.reshape(-1, C)[mask]

    # 3) compute average loss & perplexity manually
    #    loss = -mean(log p_true)
    true_probs = probs[np.arange(len(labels)), labels]
    eps = 1e-9
    log_probs = np.log(true_probs + eps)
    avg_loss = - log_probs.mean()
    avg_ppl  = float(np.exp(avg_loss))

    # 4) classic metrics
    decision_ids = np.arange(C)  # or narrow to the classes you care about
    conf_mat = confusion_matrix(labels, preds, labels=decision_ids)
    acc       = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro")

    # 5) average precision (AUPRC)
    #    need one-hot of labels
    y_bin = label_binarize(labels, classes=decision_ids)
    # and probs for each class
    auprc = average_precision_score(y_bin, probs, average="macro")

    return avg_loss, conf_mat, avg_ppl, acc, macro_f1, auprc

# ------------------------------------------------------------------
# 6) Run evaluation & print
# ------------------------------------------------------------------
avg_loss, conf_mat, ppl, acc, f1, auprc = evaluate_model(model, X_padded, y_padded)
print(f"Avg loss:      {avg_loss:.4f}")
print(f"Perplexity:    {ppl:.4f}")
print(f"Accuracy:      {acc:.4f}")
print(f"Macro F1:      {f1:.4f}")
print(f"Macro AUPRC:   {auprc:.4f}")
print("Confusion matrix:")
print(conf_mat)
