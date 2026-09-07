"""
Generation 4 - decoder-only Performer with feature-based product embeddings.

This module now only contains what is unique to this variant: the `Transformer`
assembly and `build_transformer`. The building blocks it used to define inline
(gelu_approx, LayerNormalization, FeedForwardBlock, InputEmbeddings,
PositionalEncoding, ResidualConnection, CausalPerformer, DecoderBlock, Decoder,
ProjectionLayer, SpecialPlusFeatureLookup) were byte-identical copies shared
with seven other model modules and now live in shared/. Numerics are unchanged.

Also removed: the module-level `pd.read_excel("/home/ec2-user/data/...")` that
ran on import and made this file unimportable off the original EC2 box. The
feature table is passed in as `feature_tensor`, exactly as the trainers already
did; build it with `shared.features.load_feature_tensor(path)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from shared.attention import CausalPerformer
from shared.decoder import Decoder, DecoderBlock
from shared.embeddings import SpecialPlusFeatureLookup
from shared.layers import (
    FeedForwardBlock,
    PositionalEncoding,
    ProjectionLayer,
)

##############################################################################
# 11. DecoderOnlyTransformer (GPT-style)
##############################################################################
class Transformer(nn.Module):
    """
    A single-stack (decoder-only) Transformer that
    takes a single sequence of tokens and does next-token prediction.
    """
    def __init__(self, 
                 vocab_size_tgt: int, 
                 vocab_size_src: int,
                 # tgt_seq_len: int,
                 # lto_seq_len: int,
                 max_seq_len: int,
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int,
                 d_ff: int, 
                 nb_features: int,
                 dropout: float, 
                 feature_tensor: torch.Tensor,
                 special_token_ids: torch.Tensor,
                 kernel_type="exp"):
        super().__init__()

        self.token_embed = SpecialPlusFeatureLookup(
                d_model        = d_model,
                feature_tensor = feature_tensor,           # (59, 34)
                product_ids    = list(range(13, 57)) + [59],      # 13 … 56
                vocab_size_src = vocab_size_src                # 59
        )

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
        self.projection = ProjectionLayer(d_model, vocab_size_tgt)

        self.proj_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 128))

        # (Optional) param init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_seq: torch.Tensor, return_hidden=False):
        """
        input_seq: (B, seq_len) integer tokens
        returns:   (B, seq_len, vocab_size)
        """
        x = self.token_embed(input_seq)
        x = self.pos_enc(x)
        x = self.decoder(x)
        logits = self.projection(x)
        # logits = self.decision_head(x)

        if return_hidden:
            return logits, x
        return logits

##############################################################################
# 12. Build function
##############################################################################
def build_transformer(vocab_size_src: int,
                      vocab_size_tgt: int,
                      max_seq_len: int,
                      d_model: int,
                      n_layers: int,
                      n_heads: int,
                      d_ff: int,
                      dropout: float,
                      nb_features: int,
                      feature_tensor: torch.Tensor,
                      special_token_ids: torch.Tensor,
                      kernel_type: str="exp"):
    
    transformer = Transformer(
        vocab_size_src   = vocab_size_src,
        vocab_size_tgt   = vocab_size_tgt,
        max_seq_len  = max_seq_len,
        d_model      = d_model,
        n_layers     = n_layers,
        n_heads      = n_heads,
        d_ff         = d_ff,
        dropout      = dropout,
        nb_features  = nb_features,
        feature_tensor = feature_tensor,
        special_token_ids = special_token_ids,
        kernel_type  = kernel_type
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
