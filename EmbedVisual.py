# import torch
# from config4 import get_config
# from model4_per import build_transformer
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from tokenizers import Tokenizer

# # 1) Build the Transformer architecture
# config = get_config()
# model = build_transformer(
#     src_vocab_size=config["vocab_size_src"],
#     tgt_vocab_size=config["vocab_size_tgt"],
#     lto_vocab_size=config["vocab_size_lto"],
#     src_seq_len=config["seq_len_src"],
#     tgt_seq_len=config["seq_len_tgt"],
#     lto_seq_len=config["seq_len_lto"],
#     d_model=config["d_model"],
#     N=config["N"],
#     h=config["num_heads"],
#     dropout=config["dropout"],
#     kernel_type=config["kernel_type"],
#     d_ff=config["d_ff"]
# )

# # 2) Load your checkpoint
# checkpoint_path = "drive/MyDrive/ProductGPT_weights/MyProductGPT_dmodel32_ff32_N6_heads8_gamma2_best.pt"
# checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
# model.load_state_dict(checkpoint["model_state_dict"], strict=False)
# model.eval()

# # 3) Get the embedding weight matrix (shape: (vocab_size, d_model))
# emb_weights = model.src_embed.embedding.weight.detach().cpu().numpy()
# print("Embedding matrix shape:", emb_weights.shape)

# # 4) Load the tokenizer & build the id→token mapping
# tokenizer_src = Tokenizer.from_file("drive/MyDrive/ProductGPT_weights/source_tokenizer.json")
# id_to_token = {v: k for k, v in tokenizer_src.get_vocab().items()}

# # 5) Manually filter out unwanted items (by ID or token)
# excluded_tokens = {"[PAD]", "[UNK]", "[SOS]", "[EOS]", "NA", ""}
# excluded_ids = set()
# final_indices = []
# final_tokens = []

# for idx in range(emb_weights.shape[0]):
#     token = id_to_token.get(idx, None)
#     # Skip if token is None, or in excluded set, or idx == 0
#     if token is None:
#         continue
#     if token in excluded_tokens:
#         continue
#     if idx == 0:
#         continue
#     # If we get here, we keep this token
#     final_indices.append(idx)
#     final_tokens.append(token)

# # Create the filtered embedding matrix
# filtered_emb = emb_weights[final_indices]
# print(f"Original vocab size: {emb_weights.shape[0]}, Filtered size: {filtered_emb.shape[0]}")

# # Debug: Check if “NA” or "" are still present
# for i, tok in enumerate(final_tokens):
#     if tok in ["NA", ""]:
#         print("DEBUG: Found undesired token:", tok, "at index in final_tokens =", i)

# # 6) t-SNE on the *filtered* embeddings
# tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30)
# emb_2d = tsne.fit_transform(filtered_emb)  # shape: (num_filtered, 2)

# # 7) Plot the filtered embeddings
# plt.figure(figsize=(8, 8))
# plt.scatter(emb_2d[:, 0], emb_2d[:, 1])

# # Label the first ~50 points
# for i, token in enumerate(final_tokens):
#     if i < 50:
#         x, y = emb_2d[i]
#         plt.text(x, y, token)

# plt.title("2D t-SNE of Filtered Source Embeddings")
# plt.savefig("drive/MyDrive/ProductGPT/Plot/EmbedVisual_TSNE_filtered.png", dpi=150)
# # plt.show()  # If in an interactive environment

# # 7) Save t-SNE + embeddings to CSV
# df = pd.DataFrame({
#     "token": final_tokens,
#     "x": emb_2d[:, 0],
#     "y": emb_2d[:, 1],
#     "id": final_indices
# })

# # Add original embedding dimensions as d0, d1, ..., d{d_model-1}
# for i in range(filtered_emb.shape[1]):
#     df[f"d{i}"] = filtered_emb[:, i]

# # Save to CSV
# os.makedirs("drive/MyDrive/ProductGPT/Plot", exist_ok=True)
# df.to_csv("drive/MyDrive/ProductGPT/Plot/filtered_embeddings_tsne.csv", index=False)

# # 8) Plot and save the t-SNE visualization
# plt.figure(figsize=(8, 8))
# plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10)

import os
import torch
import pandas as pd

# 0. Configuration
CKPT_PATH = "/home/ec2-user/ProductGPT/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2.pt"
# make OUT_DIR absolute, next to your checkpoint
BASE_DIR  = os.path.dirname(CKPT_PATH)
OUT_DIR   = os.path.join(BASE_DIR, "embeddings_output")
FIRST_PROD_ID, LAST_PROD_ID = 13, 56

# 1. Print where we’re running
print("Working directory:", os.getcwd())
print("Writing output to:", OUT_DIR)

# 2. Ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# 3. Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# 4. Grab the state_dict
state_dict = ckpt.get("model_state_dict", ckpt)

# 5. (Optional) Print all keys to confirm where the embedding lives
# keys = list(state_dict.keys())
# print("Available state_dict keys:", keys)

# 6. Identify the embedding key
emb_key = next((k for k in state_dict if k.endswith("id_embed.weight")), None)
if emb_key is None:
    raise KeyError("Could not find a key ending with 'id_embed.weight' in the checkpoint")
print("Using embedding key:", emb_key)

# 7. Slice out rows 13–56
emb_matrix = state_dict[emb_key]  # shape: [vocab_size, d_model]
subset = emb_matrix[FIRST_PROD_ID:LAST_PROD_ID + 1].cpu().numpy()

# 8. Build DataFrame and save to CSV
df = pd.DataFrame(
    subset,
    index=range(FIRST_PROD_ID, LAST_PROD_ID + 1),
    columns=[f"dim_{i}" for i in range(subset.shape[1])]
)
df.index.name = "token_id"

csv_path = os.path.join(OUT_DIR, "token_embeddings_13_56.csv")
df.to_csv(csv_path)

print(f"✅ Saved embeddings for tokens {FIRST_PROD_ID}–{LAST_PROD_ID} "
      f"({subset.shape[0]}×{subset.shape[1]}) to:\n  {csv_path}")

# 9. List the files in OUT_DIR so you can confirm
print("Contents of", OUT_DIR, ":", os.listdir(OUT_DIR))
