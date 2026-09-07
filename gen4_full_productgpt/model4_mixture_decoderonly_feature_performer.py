"""
Generation 4 - mixture-head decoder-only Performer, feature-based embeddings.

Each user gets their own softmax gate over the attention heads (`UserHeadGate`),
which is how consumer heterogeneity is modelled. That gate is threaded through
Decoder -> DecoderBlock -> CausalPerformer.

The shared blocks now live in shared/. Their `gate` argument defaults to None,
in which case they behave exactly like the non-mixture variants, so the same
code serves both. Numerics are unchanged.

Also removed: the module-level `pd.read_excel("/home/ec2-user/data/...")` that
ran on import. Pass the feature table in as `feature_tensor`; build it with
`shared.features.load_feature_tensor(path)`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from shared.attention import CausalPerformer
from shared.decoder import Decoder, DecoderBlock
from shared.embeddings import SpecialPlusFeatureLookup
from shared.layers import (
    FeedForwardBlock,
    PositionalEncoding,
    ProjectionLayer,
    gelu_approx,
)

class UserHeadGate(nn.Module):
    """
    gate(u) -> weights over heads: [B, H], sum_h gate=1
    Train: use per-user gates
    Val/Test: use mean gate across users
    """
    def __init__(self, num_users: int, num_heads: int):
        super().__init__()
        self.logits = nn.Embedding(num_users, num_heads)  # [U, H]
        nn.init.zeros_(self.logits.weight)               # start ~uniform after softmax

        self.register_buffer("mean_gate", torch.full((num_heads,), 1.0 / num_heads))
        self.use_mean_gate = False

    @torch.no_grad()
    def update_mean_gate(self):
        probs = torch.softmax(self.logits.weight, dim=-1)     # [U, H]
        self.mean_gate.copy_(probs.mean(dim=0))               # [H]

    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor:
        # user_ids: [B]
        if self.use_mean_gate:
            return self.mean_gate[None, :].expand(user_ids.size(0), -1)  # [B, H]
        return torch.softmax(self.logits(user_ids), dim=-1)              # [B, H]


##############################################################################
# 1. gelu_approx
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
                 num_users: int,
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
        self.gate = UserHeadGate(num_users=num_users, num_heads=n_heads)

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
    
    def forward(self, input_seq: torch.Tensor, user_ids: Optional[torch.LongTensor] = None, return_hidden=False):
        x = self.token_embed(input_seq)
        x = self.pos_enc(x)

        gate = None
        if user_ids is not None:
            gate = self.gate(user_ids)   # (B, H)

        x = self.decoder(x, gate=gate)
        logits = self.projection(x)

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
                      num_users: int,
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
        num_users    = num_users,
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
    
