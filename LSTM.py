import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Embedding, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------
# 1) Parse each user's row into numeric arrays
# ------------------------------------------------------------------
# Load JSON dataset
def load_json_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    

def parse_space_separated_ints(string_list):
    """
    Splits the first element of the string_list on spaces and converts tokens to integers.
    If a token is "NA", it replaces it with 0 (or another placeholder).
    """
    long_string = string_list[0]
    tokens = long_string.split()
    
    cleaned_values = []
    for tok in tokens:
        if tok == "NA":
            cleaned_values.append(0)  # or use -1, or np.nan if using floats
        else:
            cleaned_values.append(int(tok))
    return cleaned_values

all_users_data = []  # Will store a list of dicts with numeric arrays
records = load_json_dataset("drive/MyDrive/ProductGPT/clean_list_int_wide12_simple3.json")

for row in records:
    user_id = row["uid"][0]

    # Parse each field you need
    item_seq = parse_space_separated_ints(row["Item"])
    lto_seq  = parse_space_separated_ints(row["LTO"])
    dec_seq  = parse_space_separated_ints(row["Decision"])
    # Additional fields if you need them...

    all_users_data.append({
        "uid": user_id,
        "Item": item_seq,
        "LTO": lto_seq,
        "Decision": dec_seq
        # ...
    })

# ------------------------------------------------------------------
# 2) For each user, build (T, 14) inputs if "10 items + 4 LTO" per step
# ------------------------------------------------------------------
# The typical logic:
#   - For each time t, we want the last 10 item values plus the last 4 LTO values.
#   - This yields a 14-dimensional vector at time t.
#   - The label (decision) is `dec_seq[t]`.

window_item = 10
window_lto  = 4

X_list = []
y_list = []

for user_data in all_users_data:
    item_seq = user_data["Item"]  # e.g. length N
    lto_seq  = user_data["LTO"]   # e.g. length N
    dec_seq  = user_data["Decision"]
    
    length = min(len(item_seq), len(lto_seq), len(dec_seq))
    # We'll assume item_seq, lto_seq, dec_seq all have the same length, 
    # but if not, we take the min to be safe.

    # We'll collect all time steps from t=0..(length-1).
    # But for the first few time steps, we might not have 10 previous items, 
    # so we can either:
    #   (a) skip those initial steps, or
    #   (b) zero-pad them. 
    # For simplicity, let's skip the first `window_item-1` steps:
    start_t = max(window_item, window_lto)
    
    user_X = []
    user_y = []
    
    for t in range(start_t, length):
        # Build a feature vector of length 14:
        #   [ item_seq[t-10], ..., item_seq[t-1], item_seq[t], 
        #     lto_seq[t-4], ..., lto_seq[t-1], lto_seq[t] ]
        # 
        # Actually, that’s 11 items if we do t-10..t inclusive. 
        # So be precise: if you want EXACTLY 10 item values, 
        # maybe it’s item_seq[t-10 : t], i.e. 10 values ending right before t.
        # But let’s assume “the last 10 item values up to and including t-1”.
        # You can decide your exact indexing. Example:
        
        last_10_items = item_seq[t - window_item : t]
        last_4_lto    = lto_seq[t - window_lto : t]
        
        # In Python indexing, item_seq[t-window_item : t] 
        # will be exactly 10 elements if t >= 10.
        
        # Combine them into one list of length 14
        features_14 = last_10_items + last_4_lto  # shape = (14,)
        
        user_X.append(features_14)
        user_y.append(dec_seq[t])  # predict decision at time t
    
    user_X = np.array(user_X)  # shape = (T_user, 14)
    user_y = np.array(user_y)  # shape = (T_user,)

    X_list.append(user_X)
    y_list.append(user_y)

# Now X_list, y_list is a list of arrays, one for each user.
# If your sequence lengths differ across users, you can:
#   - pad them to the same length using e.g. keras.preprocessing.sequence.pad_sequences
#   - or keep them separate and do batch_size=1, or a generator that handles variable-length sequences.

# Figure out the max length
max_len = max(x.shape[0] for x in X_list)

# We can pad X_list (3D -> list of 2D) with zeros. Keras expects (batch, timesteps, features).
X_padded = pad_sequences(X_list, maxlen=max_len, dtype='float32', padding='pre', 
                         value=0.0)  # shape = (num_users, max_len, 14)

# We also pad y_list similarly (2D -> list of 1D).
y_padded = pad_sequences(y_list, maxlen=max_len, dtype='int32', padding='pre',
                         value=-1)    # shape = (num_users, max_len)

# Check shapes
print("X_padded shape:", X_padded.shape)  # (num_users, max_len, 14)
print("y_padded shape:", y_padded.shape)  # (num_users, max_len)

# ------------------------------------------------------------------
# 3) Now you can feed X_padded, y_padded into an LSTM
# ------------------------------------------------------------------

num_classes = 7  # Suppose decisions are in {0..6}, for example
# If your y are in {0..6}, you can one-hot them or use sparse_categorical_crossentropy

model = Sequential()

# (Optional) Masking layer if you used a special padding value that you want the LSTM to ignore
# For example, if you used 0.0 for X, or -1 for y. 
# If your features can’t be zero in normal data, you can do:
model.add(Masking(mask_value=0.0, input_shape=(max_len, 14)))

# LSTM returning sequences if you want a decision at *every* time step
model.add(LSTM(32, return_sequences=True))
# Then a final TimeDistributed Dense
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# y_padded is shape (batch_size, max_len) with integer class IDs => OK for sparse_categorical_crossentropy
model.fit(X_padded, y_padded, 
          batch_size=2,  # small batch for example
          epochs=5)

# This is obviously a toy example; adapt to real data and your real shapes.