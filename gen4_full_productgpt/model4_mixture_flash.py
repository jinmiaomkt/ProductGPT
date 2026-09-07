#!/usr/bin/env python3
"""
model4_mixture_flash.py

FlashAttention transformer with mixture-head projection.
- Same CausalSelfAttention as model4_decoderonly_feature_flash.py
- Adds MultiHeadProjection with per-user gating
- Each user learns a soft mixture over H prediction heads
- Unseen users use mean gate from training population

Architecture:
  tokens → SpecialPlusFeatureLookup → [TransformerBlock × N] → MultiHeadProjection(user_id) → logits
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# Embedding (same as flash model)
# ═══════════════════════════════════════════════════════════
class SpecialPlusFeatureLookup(nn.Module):
    def __init__(self, d_model: int, feature_tensor: torch.Tensor,
                 product_ids: List[int], vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_tensor.size(1)
        self.id_embed = nn.Embedding(vocab_size, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)
        self.register_buffer("feat_tbl", feature_tensor, persistent=False)
        prod_mask = torch.zeros(vocab_size, dtype=torch.bool)
        prod_mask[product_ids] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids_long = ids.long()
        id_vec = self.id_embed(ids_long)
        raw_feat = self.feat_tbl[ids_long]
        feat_vec = self.feat_proj(raw_feat)
        keep = self.prod_mask[ids_long]
        feat_vec = feat_vec * keep.unsqueeze(-1)
        return id_vec + self.gamma * feat_vec


# ═══════════════════════════════════════════════════════════
# FlashAttention block (same as flash model)
# ═══════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════
# Per-user mixture gate
# ═══════════════════════════════════════════════════════════
class UserGate(nn.Module):
    """
    Learns a (num_users, num_heads) logit matrix.
    user_idx → softmax(logits[user_idx]) → alpha ∈ simplex(H)
    """
    def __init__(self, num_users: int, num_heads: int):
        super().__init__()
        self.logits = nn.Embedding(num_users, num_heads)
        nn.init.zeros_(self.logits.weight)  # uniform prior
        self.register_buffer("mean_alpha", torch.ones(num_heads) / num_heads)
        self.use_mean_gate = False

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Returns alpha (B, H) on the simplex."""
        if self.use_mean_gate:
            return self.mean_alpha.unsqueeze(0).expand(user_idx.size(0), -1)
        return F.softmax(self.logits(user_idx), dim=-1)

    def update_mean_gate(self, train_user_ids: Optional[List[int]] = None):
        """Compute mean alpha across training users."""
        with torch.no_grad():
            if train_user_ids is not None and len(train_user_ids) > 0:
                ids = torch.tensor(train_user_ids, device=self.logits.weight.device)
                alphas = F.softmax(self.logits(ids), dim=-1)
                self.mean_alpha.copy_(alphas.mean(dim=0))
            else:
                self.mean_alpha.fill_(1.0 / self.mean_alpha.size(0))


# ═══════════════════════════════════════════════════════════
# Multi-head projection
# ═══════════════════════════════════════════════════════════
class MultiHeadProjection(nn.Module):
    """
    H independent linear heads: d_model → vocab_size each.
    Output = sum_h alpha_h * softmax(head_h(x))
    """
    def __init__(self, d_model: int, vocab_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_heads)
        ])

    def forward(self, hidden: torch.Tensor, alpha: torch.Tensor,
                return_head_logits: bool = False):
        """
        Args:
            hidden: (B, T, d_model)
            alpha:  (B, H) mixture weights
        Returns:
            logits: (B, T, vocab_size) — mixture of head outputs
            head_logits: (B, T, H, vocab_size) if requested
        """
        B, T, D = hidden.shape
        H = self.num_heads

        # Stack head logits: (B, T, H, V)
        head_logits = torch.stack([head(hidden) for head in self.heads], dim=2)

        # Mix: alpha (B, 1, H, 1) * softmax(head_logits) → sum over H
        alpha_expanded = alpha[:, None, :, None]  # (B, 1, H, 1)
        head_probs = F.softmax(head_logits, dim=-1)  # (B, T, H, V)
        mixed_probs = (alpha_expanded * head_probs).sum(dim=2)  # (B, T, V)

        if return_head_logits:
            return mixed_probs, head_logits
        return mixed_probs


# ═══════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════
class MixtureFlashTransformer(nn.Module):
    def __init__(
        self,
        vocab_size_tgt: int,
        vocab_size_src: int,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        feature_tensor: torch.Tensor,
        special_token_ids: List[int],
        num_users: int,
        num_mixture_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size_tgt = vocab_size_tgt
        self.num_users = num_users
        self.num_mixture_heads = num_mixture_heads

        # Determine product IDs (non-special tokens in valid range)
        first_prod, last_prod = 13, 56
        unk_prod = 59
        product_ids = list(range(first_prod, last_prod + 1)) + [unk_prod]

        # Embedding
        self.embed = SpecialPlusFeatureLookup(
            d_model=d_model,
            feature_tensor=feature_tensor,
            product_ids=product_ids,
            vocab_size=vocab_size_src,
        )
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer decoder blocks
        self.decoder = nn.ModuleDict({
            "layers": nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]),
            "norm": nn.LayerNorm(d_model),
        })

        # Mixture projection
        self.gate = UserGate(num_users, num_mixture_heads)
        self.projection = MultiHeadProjection(d_model, vocab_size_tgt, num_mixture_heads)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        user_ids: torch.Tensor,
        return_hidden: bool = False,
        return_head_logits: bool = False,
    ):
        """
        Args:
            x:        (B, T) token IDs
            user_ids: (B,)  user indices (0 = unknown)
            return_hidden: also return transformer hidden states
            return_head_logits: also return per-head logits
        Returns:
            probs: (B, T, vocab_size_tgt) mixed probabilities
            or tuple of (probs, hidden, head_logits) depending on flags
        """
        B, T = x.shape
        device = x.device

        # Embed + positional
        pos = torch.arange(T, device=device).unsqueeze(0)
        h = self.drop(self.embed(x) + self.pos_embed(pos))

        # Transformer blocks
        for layer in self.decoder["layers"]:
            h = layer(h)
        h = self.decoder["norm"](h)

        # User gate
        alpha = self.gate(user_ids)  # (B, H)

        # Multi-head projection
        if return_head_logits:
            mixed_probs, head_logits = self.projection(h, alpha, return_head_logits=True)
        else:
            mixed_probs = self.projection(h, alpha)
            head_logits = None

        if return_hidden or return_head_logits:
            return mixed_probs, h, head_logits
        return mixed_probs

    def set_mean_gate_from_train_users(self, train_user_ids: List[int]):
        """Call after loading checkpoint to set mean gate for unseen users."""
        self.gate.update_mean_gate(train_user_ids)


# ═══════════════════════════════════════════════════════════
# Builder (compatible interface)
# ═══════════════════════════════════════════════════════════
def build_transformer(
    vocab_size_tgt: int,
    vocab_size_src: int,
    max_seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    feature_tensor: torch.Tensor,
    special_token_ids: List[int],
    num_users: int,
    num_mixture_heads: int = 4,
    **kwargs,
) -> MixtureFlashTransformer:
    return MixtureFlashTransformer(
        vocab_size_tgt=vocab_size_tgt,
        vocab_size_src=vocab_size_src,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        feature_tensor=feature_tensor,
        special_token_ids=special_token_ids,
        num_users=num_users,
        num_mixture_heads=num_mixture_heads,
        **kwargs,
    )