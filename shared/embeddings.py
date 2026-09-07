"""
Token embedding with an optional product-attribute branch.

Product tokens get:   id_embedding(t) + gamma * feat_proj(features[t])
Other tokens get:     id_embedding(t)

`gamma` is a single learnable scalar, so the network can weight the whole
feature branch up or down. The feature table itself is a frozen buffer.

This is the implementation shared, byte-for-byte, by:
    model2_decoderonly_feature_performer.py
    model4_decoderonly_feature_performer.py
    model4_mixture_decoderonly_feature_performer.py
    model4_mixture2_decoderonly_feature_performer.py
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpecialPlusFeatureLookup(nn.Module):
    def __init__(self, d_model: int,
                 feature_tensor: torch.Tensor,
                 product_ids: list[int],
                 vocab_size_src: int):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_tensor.size(1)

        # ── id and feature branches ─────────────────────────────
        self.id_embed = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)

        # constant look-up table, e.g. (60, 34)
        self.register_buffer("feat_tbl", feature_tensor, persistent=False)

        # mask: True for product tokens
        prod_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        prod_mask[product_ids] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)

        # learnable scale so the network can re-weight the branches
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids: torch.Tensor,
                ext_features: torch.Tensor | None = None):
        """
        ids          : (B, T) int64
        ext_features : (B, T, feature_dim) optional, used only for UNK products
        """
        ids_long = ids.long()

        # id branch
        id_vec = self.id_embed(ids_long)                 # (B, T, D)

        # feature branch
        raw_feat = self.feat_tbl[ids_long] if ext_features is None else ext_features
        feat_vec = self.feat_proj(raw_feat)              # (B, T, D)

        # zero out features for NON-product tokens
        keep = self.prod_mask[ids_long]                  # (B, T) bool
        feat_vec = feat_vec * keep.unsqueeze(-1)

        # weighted sum
        return id_vec + self.gamma * feat_vec
