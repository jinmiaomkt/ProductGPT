"""
Generation 2 (LP / Duplet) - decoder-only Performer, feature-based embeddings.

Same architecture family as generation 4, but the event block is 5 tokens
(`ai_rate=5`: LTO offer + previous decision) with no obtained-products stream.

Only the pieces unique to this variant live here. The shared blocks now live in
shared/; numerics are unchanged. The module-level `pd.read_excel(...)` that ran
on import has been removed - pass the feature table in as `feature_tensor`,
built with `shared.features.load_feature_tensor(path)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from shared.attention import CausalPerformer
from shared.decoder import Decoder, DecoderBlock
from shared.embeddings import SpecialPlusFeatureLookup
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

        # Embedding
        # self.token_emb = InputEmbeddings(d_model, vocab_size)
        # self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # self.token_embed = SpecialPlusFeatureLookup(
        #     d_model = d_model,
        #     feature_tensor = feature_tensor,
        #     special_token_ids = special_token_ids,
        #     hidden_dim = d_ff # using d_ff as MLP hidden size
        # )

        # lto_embed = SpecialPlusFeatureLookup(
        #     d_model = d_model,
        #     feature_tensor = feature_tensor,
        #     special_token_ids = special_token_ids,
        #     hidden_dim = d_ff
        # )

        # tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        
        # src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        # tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        # lto_pos = PositionalEncoding(d_model, lto_seq_len, dropout)

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
    
    return transformer
