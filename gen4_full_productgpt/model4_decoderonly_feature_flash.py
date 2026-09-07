from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# 1. SpecialPlusFeatureLookup  (UNCHANGED from your original)
##############################################################################
class SpecialPlusFeatureLookup(nn.Module):
    def __init__(self, d_model: int,
                 feature_tensor: torch.Tensor,
                 product_ids: list[int],
                 vocab_size_src: int):

        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_tensor.size(1)

        # ── id and feature branches ─────────────────────────────
        self.id_embed  = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)

        # constant look-up table (V, 34)
        self.register_buffer("feat_tbl", feature_tensor, persistent=False)

        # mask: True for product tokens
        prod_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        prod_mask[product_ids] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)

        # learnable scale so the network can re-weight the branches
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids: torch.Tensor,
                ext_features: torch.Tensor | None = None):
        ids_long = ids.long()
        id_vec = self.id_embed(ids_long)
        raw_feat = self.feat_tbl[ids_long] if ext_features is None else ext_features
        feat_vec = self.feat_proj(raw_feat)
        keep = self.prod_mask[ids_long]
        feat_vec = feat_vec * keep.unsqueeze(-1)
        return id_vec + self.gamma * feat_vec

##############################################################################
# 2. PositionalEncoding  (UNCHANGED)
##############################################################################
class PositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

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
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len, x.size(-1))
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)

##############################################################################
# 3. CausalSelfAttention  (REPLACES CausalPerformer)
#
#    Uses F.scaled_dot_product_attention which automatically selects:
#      - FlashAttention-2  (if bf16/fp16 + Ampere+ GPU)
#      - Memory-efficient attention  (if fp32 or older GPU)
#    Both compute EXACT softmax attention. No approximation.
##############################################################################
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        # Fused QKV projection (more efficient than 3 separate linears)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V in one matmul
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, heads, T, d_k)

        # PyTorch 2.0+: automatically picks best backend
        #   Ampere + bf16/fp16  → FlashAttention-2
        #   Ampere + fp32       → Memory-efficient attention
        #   Older GPU           → Math fallback (still exact)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

##############################################################################
# 4. FeedForwardBlock  (UNCHANGED, switched to nn.GELU for clarity)
##############################################################################
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))

##############################################################################
# 5. DecoderBlock  (Pre-norm, same structure as before)
##############################################################################
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn   = FeedForwardBlock(d_model, d_ff, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

##############################################################################
# 6. Decoder  (stack of blocks + final norm)
##############################################################################
class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

##############################################################################
# 7. Transformer  (decoder-only GPT-style)
##############################################################################
class Transformer(nn.Module):
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
        special_token_ids: torch.Tensor,
    ):
        super().__init__()

        self.token_embed = SpecialPlusFeatureLookup(
            d_model=d_model,
            feature_tensor=feature_tensor,
            product_ids=list(range(13, 57)) + [59],
            vocab_size_src=vocab_size_src,
        )
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        # Build N decoder blocks — standard attention, no Performer
        blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.decoder = Decoder(d_model, blocks)

        # Final projection to decision vocab
        self.projection = nn.Linear(d_model, vocab_size_tgt)

        # Contrastive / auxiliary head (kept for compatibility)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 128),
        )

        # Parameter init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, input_seq: torch.Tensor, return_hidden: bool = False):
        x = self.token_embed(input_seq)
        x = self.pos_enc(x)
        x = self.decoder(x)
        logits = self.projection(x)

        if return_hidden:
            return logits, x
        return logits

##############################################################################
# 8. Build function
#
#    Signature kept compatible: nb_features and kernel_type are accepted
#    but IGNORED, so existing configs and Ray Tune code won't break.
##############################################################################
def build_transformer(
    vocab_size_src: int,
    vocab_size_tgt: int,
    max_seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    feature_tensor: torch.Tensor,
    special_token_ids: torch.Tensor,
    # Kept for backward compatibility — ignored by standard attention
    nb_features: int = 0,
    kernel_type: str = "exp",
):
    return Transformer(
        vocab_size_src=vocab_size_src,
        vocab_size_tgt=vocab_size_tgt,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        feature_tensor=feature_tensor,
        special_token_ids=special_token_ids,
    )