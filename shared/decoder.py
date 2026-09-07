"""
Decoder-only stack (pre-norm self-attention + feed-forward).

As with `CausalPerformer`, the mixture models threaded an optional `gate`
argument through `Decoder -> DecoderBlock -> attention`. That is a strict
superset: with `gate=None` (the default) these classes behave exactly like the
non-mixture versions, so both families share them.

The attention module is injected, so this works with either `CausalPerformer`
or any other block exposing `forward(q, k, v, gate=None)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import LayerNormalization, ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int,
                 self_attention_block: nn.Module,
                 feed_forward_block: nn.Module,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_attn = ResidualConnection(d_model, dropout)
        self.residual_ff = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None):
        # 1) Self-attention
        x = self.residual_attn(
            x,
            lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm, gate=gate),
        )
        # 2) Feed-forward
        x = self.residual_ff(x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None):
        for layer in self.layers:
            x = layer(x, gate=gate)
        return self.norm(x)
