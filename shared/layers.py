"""
Basic Transformer building blocks shared by the ProductGPT model variants.

Every class here is a verbatim copy of the implementation that was duplicated,
byte-for-byte, across this canonical cluster of eight model modules:

    model2_decoderonly_feature_performer.py
    model2_decoderonly_index_performer.py
    model4_bigbird.py
    model4_decoderonly_feature_performer.py
    model4_decoderonly_index_performer.py
    model4_decoderonly_index_performer_original.py
    model4_mixture_decoderonly_feature_performer.py
    model4_mixture2_decoderonly_feature_performer.py

Numerics are unchanged. Two deliberate notes:

* `LayerNormalization` uses `.std()` (Bessel-corrected, denominator N-1), which
  differs slightly from `torch.nn.LayerNorm` (which uses the biased variance).
  This is preserved on purpose so existing checkpoints stay comparable.

* `PositionalEncoding._extend_pe` rebuilds the whole table. The index-variant
  models instead appended only the new rows. Both compute the identical
  sinusoid values for every position, so the two are numerically equivalent.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """Approximate GeLU using the tanh approximation."""
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(
            self.dropout(gelu_approx(self.linear_1(x)))
        )


class InputEmbeddings(nn.Module):
    """Token embedding for an integer vocabulary, scaled by sqrt(d_model)."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding, up to a fixed max_seq_len."""

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def _extend_pe(self, new_len: int, d_model: int):
        device = self.pe.device
        pe = torch.zeros(1, new_len, d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len, x.size(-1))

        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        return self.proj(x)
