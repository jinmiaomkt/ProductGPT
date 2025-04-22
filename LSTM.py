# # import numpy as np
# # import json
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Embedding, Masking
# # from tensorflow.keras.preprocessing.sequence import pad_sequences

# # # ------------------------------------------------------------------
# # # 1) Parse each user's row into numeric arrays
# # # ------------------------------------------------------------------
# # # Load JSON dataset
# # def load_json_dataset(filepath):
# #     with open(filepath, 'r') as f:
# #         return json.load(f)
    

# # def parse_space_separated_ints(string_list):
# #     """
# #     Splits the first element of the string_list on spaces and converts tokens to integers.
# #     If a token is "NA", it replaces it with 0 (or another placeholder).
# #     """
# #     long_string = string_list[0]
# #     tokens = long_string.split()
    
# #     cleaned_values = []
# #     for tok in tokens:
# #         if tok == "NA":
# #             cleaned_values.append(0)  # or use -1, or np.nan if using floats
# #         else:
# #             cleaned_values.append(int(tok))
# #     return cleaned_values

# # all_users_data = []  # Will store a list of dicts with numeric arrays
# # records = load_json_dataset("/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json")

# # for row in records:
# #     user_id = row["uid"][0]

# #     # Parse each field you need
# #     item_seq = parse_space_separated_ints(row["Item"])
# #     lto_seq  = parse_space_separated_ints(row["LTO"])
# #     dec_seq  = parse_space_separated_ints(row["Decision"])
# #     # Additional fields if you need them...

# #     all_users_data.append({
# #         "uid": user_id,
# #         "Item": item_seq,
# #         "LTO": lto_seq,
# #         "Decision": dec_seq
# #         # ...
# #     })

# # # ------------------------------------------------------------------
# # # 2) For each user, build (T, 14) inputs if "10 items + 4 LTO" per step
# # # ------------------------------------------------------------------
# # # The typical logic:
# # #   - For each time t, we want the last 10 item values plus the last 4 LTO values.
# # #   - This yields a 14-dimensional vector at time t.
# # #   - The label (decision) is `dec_seq[t]`.

# # window_item = 10
# # window_lto  = 4

# # X_list = []
# # y_list = []

# # for user_data in all_users_data:
# #     item_seq = user_data["Item"]  # e.g. length N
# #     lto_seq  = user_data["LTO"]   # e.g. length N
# #     dec_seq  = user_data["Decision"]
    
# #     length = min(len(item_seq), len(lto_seq), len(dec_seq))
# #     # We'll assume item_seq, lto_seq, dec_seq all have the same length, 
# #     # but if not, we take the min to be safe.

# #     # We'll collect all time steps from t=0..(length-1).
# #     # But for the first few time steps, we might not have 10 previous items, 
# #     # so we can either:
# #     #   (a) skip those initial steps, or
# #     #   (b) zero-pad them. 
# #     # For simplicity, let's skip the first `window_item-1` steps:
# #     start_t = max(window_item, window_lto)
    
# #     user_X = []
# #     user_y = []
    
# #     for t in range(start_t, length):
# #         # Build a feature vector of length 14:
# #         #   [ item_seq[t-10], ..., item_seq[t-1], item_seq[t], 
# #         #     lto_seq[t-4], ..., lto_seq[t-1], lto_seq[t] ]
# #         # 
# #         # Actually, that’s 11 items if we do t-10..t inclusive. 
# #         # So be precise: if you want EXACTLY 10 item values, 
# #         # maybe it’s item_seq[t-10 : t], i.e. 10 values ending right before t.
# #         # But let’s assume “the last 10 item values up to and including t-1”.
# #         # You can decide your exact indexing. Example:
        
# #         last_10_items = item_seq[t - window_item : t]
# #         last_4_lto    = lto_seq[t - window_lto : t]
        
# #         # In Python indexing, item_seq[t-window_item : t] 
# #         # will be exactly 10 elements if t >= 10.
        
# #         # Combine them into one list of length 14
# #         features_14 = last_10_items + last_4_lto  # shape = (14,)
        
# #         user_X.append(features_14)
# #         user_y.append(dec_seq[t])  # predict decision at time t
    
# #     user_X = np.array(user_X)  # shape = (T_user, 14)
# #     user_y = np.array(user_y)  # shape = (T_user,)

# #     X_list.append(user_X)
# #     y_list.append(user_y)

# # # Now X_list, y_list is a list of arrays, one for each user.
# # # If your sequence lengths differ across users, you can:
# # #   - pad them to the same length using e.g. keras.preprocessing.sequence.pad_sequences
# # #   - or keep them separate and do batch_size=1, or a generator that handles variable-length sequences.

# # # Figure out the max length
# # max_len = max(x.shape[0] for x in X_list)

# # # We can pad X_list (3D -> list of 2D) with zeros. Keras expects (batch, timesteps, features).
# # X_padded = pad_sequences(X_list, maxlen=max_len, dtype='float32', padding='pre', 
# #                          value=0.0)  # shape = (num_users, max_len, 14)

# # # We also pad y_list similarly (2D -> list of 1D).
# # y_padded = pad_sequences(y_list, maxlen=max_len, dtype='int32', padding='pre',
# #                          value=-1)    # shape = (num_users, max_len)

# # # Check shapes
# # print("X_padded shape:", X_padded.shape)  # (num_users, max_len, 14)
# # print("y_padded shape:", y_padded.shape)  # (num_users, max_len)

# # # ------------------------------------------------------------------
# # # 3) Now you can feed X_padded, y_padded into an LSTM
# # # ------------------------------------------------------------------

# # num_classes = 7  # Suppose decisions are in {0..6}, for example
# # # If your y are in {0..6}, you can one-hot them or use sparse_categorical_crossentropy

# # model = Sequential()

# # # (Optional) Masking layer if you used a special padding value that you want the LSTM to ignore
# # # For example, if you used 0.0 for X, or -1 for y. 
# # # If your features can’t be zero in normal data, you can do:
# # model.add(Masking(mask_value=0.0, input_shape=(max_len, 14)))

# # # LSTM returning sequences if you want a decision at *every* time step
# # model.add(LSTM(32, return_sequences=True))
# # # Then a final TimeDistributed Dense
# # model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

# # model.compile(loss='sparse_categorical_crossentropy', 
# #               optimizer='adam',
# #               metrics=['accuracy'])

# # # y_padded is shape (batch_size, max_len) with integer class IDs => OK for sparse_categorical_crossentropy
# # model.fit(X_padded, y_padded, 
# #           batch_size=2,  # small batch for example
# #           epochs=5)

# # # This is obviously a toy example; adapt to real data and your real shapes.

# import numpy as np
# import json
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------------------------------------------
# # Utilities
# # ------------------------------------------------------------------
# def load_json_dataset(filepath):
#     with open(filepath, 'r') as f:
#         return json.load(f)

# def parse_space_separated_ints(string_list):
#     """
#     string_list: e.g. row["AggregateInput"] or row["Decision"], 
#     each is a single-element list containing a space‑separated string.
#     """
#     tokens = string_list[0].split()
#     return [0 if tok=="NA" else int(tok) for tok in tokens]

# # ------------------------------------------------------------------
# # 1) Parse aggregated inputs + decisions
# # ------------------------------------------------------------------
# records = load_json_dataset("/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json")

# X_list, y_list = [], []
# for row in records:
#     # flatten all the 15*N ints, then reshape into (N, 15)
#     flat_agg = parse_space_separated_ints(row["AggregateInput"])
#     seq_len = len(flat_agg) // 15
#     agg_seq = np.array(flat_agg).reshape(seq_len, 15)

#     # decision sequence of length ≥ seq_len
#     dec_seq = parse_space_separated_ints(row["Decision"])

#     # we'll use agg_seq[t] to predict dec_seq[t+1], so drop last step
#     valid_steps = min(seq_len - 1, len(dec_seq) - 1)
#     X_list.append( agg_seq[:valid_steps] )
#     y_list.append( np.array(dec_seq[1:valid_steps+1]) )

# # ------------------------------------------------------------------
# # 2) Pad to common length
# # ------------------------------------------------------------------
# max_len = max(x.shape[0] for x in X_list)
# X_padded = pad_sequences(
#     X_list,
#     maxlen=max_len,
#     dtype="float32",
#     padding="pre",
#     value=0.0
# )  # shape = (num_users, max_len, 15)

# y_padded = pad_sequences(
#     y_list,
#     maxlen=max_len,
#     dtype="int32",
#     padding="pre",
#     value=-1
# )  # shape = (num_users, max_len)

# # ------------------------------------------------------------------
# # 3) Build & train LSTM
# # ------------------------------------------------------------------
# num_classes = 9  # adjust to your number of decision classes

# model = Sequential([
#     Masking(mask_value=0, input_shape=(max_len, 15)),
#     LSTM(32, return_sequences=True),
#     TimeDistributed(Dense(num_classes, activation="softmax"))
# ])

# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# model.fit(
#     X_padded,
#     y_padded,
#     batch_size=2,
#     epochs=5
# )

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
    average_precision_score,
    label_binarize,
)

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
