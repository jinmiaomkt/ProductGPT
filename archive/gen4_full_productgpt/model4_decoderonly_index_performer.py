"""
Generation 4 - decoder-only Performer with plain index (ID) embeddings.

This is the ablation against the feature-based variant: products are ordinary
learned embeddings with no attribute projection.

Only the pieces unique to this variant live here (`Transformer`,
`build_transformer`). The blocks it used to define inline were byte-identical
copies shared with seven other model modules and now live in shared/.
Numerics are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from shared.attention import CausalPerformer
from shared.decoder import Decoder, DecoderBlock
from shared.layers import (
    FeedForwardBlock,
    InputEmbeddings,
    PositionalEncoding,
    ProjectionLayer,
)

class Transformer(nn.Module):
    """
    A single-stack (decoder-only) Transformer that
    takes a single sequence of tokens and does next-token prediction.
    """
    def __init__(self, 
                 vocab_size: int, 
                 max_seq_len: int,
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int,
                 d_ff: int, 
                 dropout: float, 
                 nb_features: int,
                 kernel_type="exp"):
        super().__init__()
        # Embedding
        self.token_emb = InputEmbeddings(d_model, vocab_size)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # Build N decoder blocks
        blocks = []
        for _ in range(n_layers):
            performer = CausalPerformer(d_model = d_model, 
                                        n_heads = n_heads, 
                                        dropout = dropout, 
                                        kernel_type = kernel_type,
                                        nb_features = nb_features)
            ff_block  = FeedForwardBlock(d_model, d_ff, dropout)
            blk = DecoderBlock(d_model, performer, ff_block, dropout)
            blocks.append(blk)
        self.decoder = Decoder(d_model, nn.ModuleList(blocks))
        # Final projection to vocab
        self.projection = ProjectionLayer(d_model, vocab_size)

        # (Optional) param init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        input_seq: (B, seq_len) integer tokens
        returns:   (B, seq_len, vocab_size)
        """
        x = self.token_emb(input_seq)
        x = self.pos_enc(x)
        x = self.decoder(x)
        logits = self.projection(x)
        return logits

##############################################################################
# 12. Build function
##############################################################################
def build_transformer(vocab_size: int,
                      max_seq_len: int,
                      d_model: int,
                      n_layers: int,
                      n_heads: int,
                      d_ff: int,
                      dropout: float,
                      nb_features: int,
                      kernel_type: str="exp"):
    
    transformer = Transformer(
        vocab_size   = vocab_size,
        max_seq_len  = max_seq_len,
        d_model      = d_model,
        n_layers     = n_layers,
        n_heads      = n_heads,
        d_ff         = d_ff,
        dropout      = dropout,
        kernel_type  = kernel_type,
        nb_features  = nb_features,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
