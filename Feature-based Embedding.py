import pandas as pd
import numpy as np
import torch

df = pd.read_excel("SelectedFigureWeaponEmbeddingIndex.xlsx", sheet_name=0)

## NewProductIndex3 is not a feature
feature_cols = [
    "Rarity", 
    "MaxLife", 
    "MaxOffense", 
    "MaxDefense",
    "WeaponTypeOneHandSword", 
    "WeaponTypeTwoHandSword", 
    "WeaponTypeArrow", 
    "WeaponTypeMagic", 
    "WeaponTypePolearm",
    "EthnicityIce", 
    "EthnicityRock", 
    "EthnicityWater", 
    "EthnicityFire", 
    "EthnicityGrass", 
    "EthnicityThunder", 
    "EthnicityWind", 
    "EthnicityFemale", 
    "EthnicityMale",
    "CountryFengDan", 
    "CountryRuiYue", 
    "CountryDaoQi", 
    "CountryZhiDong", 
    "CountryMengDe", 
    "CountryXuMi",
    "type",  # if it's numeric or one-hot encoded; else skip if it's a string
    "MinimumAttack", 
    "MaximumAttack",
    "MinSpecialEffect", 
    "MaxSpecialEffect",
    "SpecialEffectEfficiency", 
    "SpecialEffectExpertise",
    "SpecialEffectAttack", 
    "SpecialEffectSuper",
    "SpecialEffectRatio", 
    "SpecialEffectPhysical", 
    "SpecialEffectLife", 
    # "NewProductIndex3",  # only if you actually want it as a feature
    "LTO" 
]

## Determine max_token_id and the dimension
max_token_id = 120
# This is the count of numeric columns (your "feature_dim"). 
# For example, if you have 38 numeric columns (NOT counting the ID).
feature_dim = 37

feature_array = np.zeros((max_token_id + 1, feature_dim), dtype=np.float32)

for idx in range(len(df)):
    # The product ID for this row (should be in [1..65])
    token_id = int(df["NewProductIndex3"].iloc[idx])  
    
    # Gather the numeric features from the row. 
    # If you only want the columns in `feature_cols` (excluding "NewProductIndex3"), do:
    feats = df.loc[idx, feature_cols].values  # shape (37,) => [Rarity, MaxLife, MaxOffense, ...]

    feature_array[token_id, :] = feats

# ## change
# feature_cols = [col for col in df.columns 
#                 if col not in ["NewProductIndex3", "SomeTextCol", ...]]

# for idx in range(len(df)):
#     token_id = int(df["NewProductIndex3"].iloc[idx]) 
#     feats = df.loc[idx, feature_cols].values
#     feature_array[token_id, :] = feats

feature_tensor = torch.from_numpy(feature_array)  # shape [110, 38]

# In your model initialization, keep features frozen
self.register_buffer("feature_table", feature_tensor)








## 1. Prepare a Feature Table for Product IDs
## 2. Create a Hybrid Embedding Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpecialPlusFeatureLookup(nn.Module):
    """
    A single module that, given a batch of token IDs [0..109],
    produces an embedding of dimension d_model.

    - For special tokens (like {0,107,108,109}), we use a small learned embedding.
    - For real product IDs [1..65], we gather from a
      precomputed feature_tensor and pass through an MLP to get a d_model vector.
    """

    def __init__(self,
                 d_model: int,
                 feature_tensor: torch.Tensor,
                 special_token_ids: list,
                 hidden_dim: int = 0):
        super().__init__()
        """
        :param d_model: dimension of the final embedding for each token
        :param feature_tensor: shape (max_token_id+1, feature_dim)
                              A buffer/parameter with row i = feature vector for token i
        :param special_token_ids: list of token IDs that have no product features (like [0,107,108,109])
        :param hidden_dim: optional dimension for an MLP hidden layer if you want more nonlinearity
        """

        self.d_model = d_model
        self.feature_dim = feature_tensor.shape[1]
        self.max_id = feature_tensor.shape[0] - 1

        # We'll store the product feature table as a *buffer* if we do NOT want to learn it.
        # If you want to fine-tune it, you can store as a Parameter. For now, let's do buffer:
        self.register_buffer("feature_table", feature_tensor)

        # Build a dictionary mapping special IDs -> index in special embedding
        # E.g. {0: 0, 107: 1, 108: 2, 109: 3, ...}
        self.special_id_map = {}
        for idx, tok_id in enumerate(special_token_ids):
            self.special_id_map[tok_id] = idx

        self.num_special = len(special_token_ids)
        self.special_embed = nn.Embedding(self.num_special, d_model)

        # MLP or linear to map product features -> d_model
        if hidden_dim > 0:
            # Example small MLP
            self.product_mlp = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model),
            )
        else:
            # Single linear layer
            self.product_mlp = nn.Linear(self.feature_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        :param token_ids: (batch_size, seq_len) integer IDs
        :return: (batch_size, seq_len, d_model) final embeddings
        """
        device = token_ids.device
        B, S = token_ids.shape

        # We'll create an output buffer (B,S,d_model)
        out = torch.zeros((B, S, self.d_model), device=device)

        # 1) Identify special vs. product tokens
        # We'll build an index map telling us which row in 'special_embed' each token corresponds to,
        # or -1 if it's not a special token.

        special_idx = torch.full_like(token_ids, -1, device=device)
        for i, (special_id) in enumerate(self.special_id_map.items()):
            # special_id is (tok_id -> index_in_special_embed)
            # Actually we need to invert:  special_id_map[tok_id] = index_in_special_embed
            pass

        # We should actually do something like:
        # for token_id, embed_index in self.special_id_map.items():
        #     mask = (token_ids == token_id)
        #     special_idx[mask] = embed_index

        for special_token, embed_index in self.special_id_map.items():
            mask = (token_ids == special_token)
            special_idx[mask] = embed_index

        is_special = (special_idx >= 0)
        is_product = ~is_special

        # 2) Embed special tokens
        special_positions = special_idx[is_special]  # shape: (num_special_positions,)
        special_embeds = self.special_embed(special_positions)  # (num_special_positions, d_model)
        out[is_special] = special_embeds

        # 3) Real product tokens
        # Gather each product's row from our feature_table
        # shape: (num_product_positions, feature_dim)
        product_feature_vecs = self.feature_table[token_ids[is_product]]

        # Pass them through the MLP/linear
        product_embeds = self.product_mlp(product_feature_vecs)  # (num_product_positions, d_model)
        product_embeds = F.relu(product_embeds)
        # Typically we multiply by sqrt(d_model) for Transformer scale
        product_embeds = product_embeds * math.sqrt(self.d_model)

        out[is_product] = product_embeds

        return out

## 3. Integrate into Transformer Model

def build_transformer(..., feature_tensor, special_token_ids, d_model=256, hidden_dim=512):
    # Create the custom embedding module for all tokens
    #   - Pass it the product feature table
    #   - Provide the IDs that are considered "special"
    #   - Maybe give it an MLP hidden_dim for more non‐linear capacity
    src_embed = SpecialPlusFeatureLookup(
        d_model=d_model,
        feature_tensor=feature_tensor,
        special_token_ids=special_token_ids,
        hidden_dim=hidden_dim
    )
    tgt_embed = SpecialPlusFeatureLookup(
        d_model=d_model,
        feature_tensor=feature_tensor,
        special_token_ids=special_token_ids,
        hidden_dim=hidden_dim
    )
    lto_embed = SpecialPlusFeatureLookup(
        d_model=d_model,
        feature_tensor=feature_tensor,
        special_token_ids=special_token_ids,
        hidden_dim=hidden_dim
    )

    # Then your usual positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    lto_pos = PositionalEncoding(d_model, lto_seq_len, dropout)

    # Construct the rest of the Transformer (encoder, decoder, etc.)
    ...

    # Return the final model
    return Transformer(
        encoder, decoder,
        src_embed, tgt_embed, lto_embed,
        src_pos, tgt_pos, lto_pos,
        projection_layer
    )

