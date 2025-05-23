# dataset4_decoderonly.py
# ────────────────────────────────────────────────────────────────
# Rolling-window dataset for decision-only training.
#
# Each __getitem__ returns:
#   {
#       "aggregate_input":  LongTensor(ctx_window)   # left-padded context
#       "label"           :  LongTensor(1)          # the decision id
#   }
#
# “ctx_window” == seq_len_ai in your config.
# seq_len_tgt is accepted but ignored (always 1).
# ────────────────────────────────────────────────────────────────
import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List, Dict

# ═══════════════════════════════════════════════════════════════
def load_json_dataset(filepath: str) -> List[Dict]:
    with open(filepath, "r") as fp:
        return json.load(fp)

# ═══════════════════════════════════════════════════════════════
class TransformerDataset(Dataset):
    """
    One sample  ==  one decision token.

    Parameters
    ----------
    data            : list[dict]      raw sessions
    tokenizer_ai    : Tokenizer       for AggregateInput stream
    tokenizer_tgt   : Tokenizer       for decision ids (1‥9).  *Only* used
                                        to pick the correct token-ids; you can
                                        pass the same tokenizer as tokenizer_ai.
    seq_len_ai      : int             rolling context window (ctx_window)
    seq_len_tgt     : int             kept for API compatibility (ignored)
    num_heads       : int             (unused here)
    ai_rate         : int             (unused here)
    pad_token       : int             id of [PAD] (defaults to 0)
    sos_token       : int             id of [SOS] (unused here)
    """
    def __init__(self,
                 data,
                 tokenizer_ai:  Tokenizer,
                 tokenizer_tgt: Tokenizer,
                 seq_len_ai:  int,
                 seq_len_tgt: int,
                 num_heads:   int,
                 ai_rate:     int,
                 pad_token:   int = 0,
                 sos_token:   int = 10):
        self.ctx_window     = seq_len_ai
        self.tk_ai          = tokenizer_ai
        self.pad_id         = pad_token

        # pre-compute ALL (context, label) pairs --------------------------------
        self.samples = []         # each element is (List[int] ctx, int label)

        for session in data:
            # tokenise whole AggregateInput once
            token_ids = self.tk_ai.encode(
                " ".join(map(str, session["AggregateInput"]))
                if isinstance(session["AggregateInput"], list)
                else str(session["AggregateInput"])
            ).ids

            # iterate over every decision position (= id 1‥9)
            for idx, tok_id in enumerate(token_ids):
                if 1 <= tok_id <= 9:                        # it's a decision
                    left = max(0, idx - self.ctx_window)    # window start
                    ctx  = token_ids[left:idx]              # preceding tokens
                    # left-pad so len(ctx) == ctx_window
                    padded = [self.pad_id]*(self.ctx_window - len(ctx)) + ctx
                    self.samples.append( (padded, tok_id) )

        # sanity print (optional)
        print(f"[Dataset] built {len(self.samples)} rolling-window samples "
              f"from {len(data)} sessions")

    # ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    # ────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        ctx, lbl = self.samples[idx]
        return {
            "aggregate_input": torch.tensor(ctx,  dtype=torch.long),
            "label"          : torch.tensor([lbl], dtype=torch.long)  # shape (1,)
        }
