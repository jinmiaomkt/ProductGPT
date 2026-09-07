"""
Causal Performer self-attention (linear attention via random features).

This is the union of the two variants that were duplicated across the canonical
model modules:

* the plain variant  -> `forward(q, k, v)`
* the mixture variant -> `forward(q, k, v, gate=...)`

The mixture variant is a strict superset: its only extra behaviour is scaling the
per-head output by a gate before the output projection. With `gate=None` (the
default) this class is numerically identical to the plain variant, so both the
plain and the mixture models can share it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import gelu_approx


class CausalPerformer(nn.Module):
    """Blockwise causal Performer, used for self-attention."""

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1,
                 kernel_type: str = "exp",
                 nb_features: int = 16,
                 block_size_h: int = 1,
                 block_size_w: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        self.nb_features = nb_features
        self.kernel_type = kernel_type

        self.block_size_h = block_size_h
        self.block_size_w = block_size_w

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._create_feature_map()

    def _create_feature_map(self):
        omega = torch.randn(self.nb_features, self.d_k) / math.sqrt(self.d_k)
        # Performer random features are not trained.
        self.omega = nn.Parameter(omega, requires_grad=False)

    def _kernel_function(self, x: torch.Tensor):
        if self.kernel_type == "gelu":
            return gelu_approx(x) + 1e-6
        elif self.kernel_type == "exp":
            return torch.exp(-0.5 * (x ** 2))
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel_type}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                gate: torch.Tensor | None = None):
        """
        q, k, v : (B, seq_len, d_model)
        gate    : (B, n_heads) optional per-user mixture gate. When None this
                  behaves exactly like the non-mixture Performer.
        """
        B, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Project
        q = self.w_q(q).view(B, seq_len_q, self.n_heads, self.d_k)
        k = self.w_k(k).view(B, seq_len_k, self.n_heads, self.d_k)
        v = self.w_v(v).view(B, seq_len_k, self.n_heads, self.d_k)

        # Apply kernel
        q_prime = self._kernel_function(q @ self.omega.T)
        k_prime = self._kernel_function(k @ self.omega.T)

        # Normalize along feature dimension
        q_prime = q_prime / (q_prime.sum(dim=-1, keepdim=True) + 1e-6)
        k_prime = k_prime / (k_prime.sum(dim=-1, keepdim=True) + 1e-6)

        # Prefix sums over the key dimension
        K_cum = torch.cumsum(k_prime, dim=1)                                   # (B, T_k, H, r)
        KV_cum = torch.cumsum(k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=1)  # (B, T_k, H, r, d_k)

        # Causal block boundaries: block i ends at (i+1)*block_size_w - 1
        q_indices = torch.arange(seq_len_q, device=q.device)
        q_block_indices = q_indices // self.block_size_h
        key_indices = (q_block_indices + 1) * self.block_size_w - 1
        key_indices = key_indices.clamp(max=seq_len_k - 1)

        indices = key_indices.view(1, -1, 1, 1).expand(B, -1, self.n_heads, self.nb_features)
        K_cum_selected = K_cum.gather(dim=1, index=indices)

        indices_kv = key_indices.view(1, -1, 1, 1, 1).expand(
            B, -1, self.n_heads, self.nb_features, self.d_k
        )
        KV_cum_selected = KV_cum.gather(dim=1, index=indices_kv)

        numerator = torch.sum(q_prime.unsqueeze(-1) * KV_cum_selected, dim=-2)
        denominator = torch.sum(q_prime * K_cum_selected, dim=-1, keepdim=True)
        out = numerator / (denominator + 1e-6)

        # Mixture-head gating: (B, H) -> broadcast to (B, 1, H, 1)
        if gate is not None:
            out = out * gate[:, None, :, None]

        out = out.reshape(B, seq_len_q, self.d_model)
        out = self.w_o(out)
        return out
