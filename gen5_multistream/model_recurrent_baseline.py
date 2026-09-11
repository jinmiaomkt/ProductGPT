"""
Recurrent baselines (GRU / LSTM) for the gen-5 pipeline.

WHY THESE, AND NOT baselines/
-----------------------------
The GRU/LSTM scripts in baselines/ feed the 15 raw integer token ids per event
straight into the RNN as floats, with no embedding, and depend on boto3/S3.
Token ids are arbitrary labels, so treating id 40 as "more than" id 13 is
meaningless; a reviewer would call that baseline a straw man. They are also
wired to the pre-fix data and the old splits.

These baselines instead share the transformer's FEATURE pipeline exactly --
the same product feature lookup (SpecialPlusFeatureLookup), the same
within-event attention pooling, the same decision embedding -- and differ
only in how events are combined OVER TIME. That isolates the claim a paper
actually wants to make: does the attention architecture beat a well-specified
recurrent model given identical inputs?

WHAT IS AND IS NOT INCLUDED
---------------------------
Included (per-event feature extraction, identical to the transformer):
    product_embed   feature-table lookup for LTO and obtained products
    offer_pool      attention pooling over an event's 4 LTO tokens
    outcome_pool    attention pooling over an event's 10 obtained tokens
    decision_embed  previous decision

Deliberately EXCLUDED (the transformer's cross-event mechanisms):
    offer_inventory_attn   the O(S^2) cross-attention over all prior inventory
    inventory_gru          a second recurrent state; the RNN here is the state
    event_model            the causal self-attention stack

The pooling is attention over tokens WITHIN one event, not across events, so
it is feature extraction rather than sequence modelling. The recurrence is
causal by construction, so no mask is needed.

Same call signature as the transformer -- model(lto, obtained, prev, user_idx)
-> (B, S, vocab_size_tgt) logits -- so the trainer, evaluation and the 2x2
cells are shared without modification.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model_multistream_state_space import (
    FIRST_PROD_ID,
    LAST_PROD_ID,
    PAD_ID,
    UNK_PROD_ID,
    AttentionPool,
    SpecialPlusFeatureLookup,
)


class RecurrentBaseline(nn.Module):
    def __init__(
        self,
        cell: str,
        vocab_size_src: int,
        vocab_size_tgt: int,
        d_model: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        feature_tensor: torch.Tensor,
        num_users: Optional[int] = None,
        use_user_embedding: bool = False,
    ):
        super().__init__()
        cell = cell.lower()
        if cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        self.cell = cell

        product_ids = list(range(FIRST_PROD_ID, LAST_PROD_ID + 1)) + [UNK_PROD_ID]
        self.product_embed = SpecialPlusFeatureLookup(
            d_model=d_model, feature_tensor=feature_tensor,
            product_ids=product_ids, vocab_size_src=vocab_size_src)
        self.decision_embed = nn.Embedding(vocab_size_src, d_model)
        self.offer_pool = AttentionPool(d_model, dropout)
        self.outcome_pool = AttentionPool(d_model, dropout)

        self.use_user_embedding = bool(use_user_embedding and num_users is not None)
        self.user_embed = (nn.Embedding(num_users, d_model)
                           if self.use_user_embedding else None)

        fusion_in = 3 * d_model + (d_model if self.use_user_embedding else 0)
        self.event_fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

        # Stacked single-layer RNNs with EXPLICIT dropout between them, rather
        # than one nn.GRU(num_layers=N, dropout=p). The two are the same model:
        # PyTorch's `dropout` argument applies dropout to each layer's output
        # except the last, which is exactly what the loop in forward() does.
        #
        # The reason is a crash. With num_layers > 1 and dropout > 0, cuDNN
        # allocates a dropout-state descriptor whose teardown fast-fails the
        # process on exit (0xC0000409) under PyTorch 2.11 / cu128 on Windows.
        # Results are already written by then, but the non-zero exit would make
        # PBS and the Telegram trap report a successful run as FAILED. Minimal
        # repro: nn.GRU(64, 64, num_layers=2, dropout=0.1).cuda() exits 127 in
        # Git Bash; the same with dropout=0.0, or one layer, exits 0.
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn_layers = nn.ModuleList(
            rnn_cls(input_size=d_model, hidden_size=d_model, num_layers=1,
                    batch_first=True)
            for _ in range(n_layers))
        self.rnn_dropout = nn.Dropout(dropout)

        self.head_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(d_model, vocab_size_tgt)

    def forward(self, lto_ids: torch.Tensor, obtained_ids: torch.Tensor,
                prev_dec_ids: torch.Tensor,
                user_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        lto_ids = lto_ids.long()
        obtained_ids = obtained_ids.long()
        prev_dec_ids = prev_dec_ids.long()

        z_x = self.offer_pool(self.product_embed(lto_ids), lto_ids.ne(PAD_ID))
        z_o = self.outcome_pool(self.product_embed(obtained_ids),
                                obtained_ids.ne(PAD_ID))
        z_y = self.decision_embed(prev_dec_ids)
        pieces = [z_x, z_o, z_y]

        if self.use_user_embedding and user_idx is not None:
            z_u = self.user_embed(user_idx.long()).unsqueeze(1).expand(-1, z_x.size(1), -1)
            pieces.append(z_u)

        s = self.event_fusion(torch.cat(pieces, dim=-1))   # (B,S,D)
        # cuDNN RNNs need contiguous input; the output is causal by design.
        last = len(self.rnn_layers) - 1
        for i, rnn in enumerate(self.rnn_layers):
            s, _ = rnn(s.contiguous())
            if i < last:
                s = self.rnn_dropout(s)
        return self.output_head(self.head_trunk(s))       # (B,S,V)


def build_recurrent_baseline(cell: str, vocab_size_src: int, vocab_size_tgt: int,
                             d_model: int, n_layers: int, d_ff: int,
                             dropout: float, feature_tensor: torch.Tensor,
                             num_users: Optional[int] = None,
                             use_user_embedding: bool = False,
                             **_ignored) -> RecurrentBaseline:
    return RecurrentBaseline(
        cell=cell, vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt,
        d_model=d_model, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        feature_tensor=feature_tensor, num_users=num_users,
        use_user_embedding=use_user_embedding)
