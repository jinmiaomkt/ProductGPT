# dataset4_decision_only.py
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers

def load_json_dataset(filepath: str) -> List[dict]:
    """Load the list-of-dicts dataset that you already store on disk."""
    with open(filepath, "r") as f:
        return json.load(f)
class TransformerDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer_ai: Tokenizer,
        tokenizer_tgt: Tokenizer,
        seq_len_ai: int,
        seq_len_tgt: int,
        num_heads: int,
        ai_rate: int,
        pad_token: int = 0,
        sos_token: int = 10,
    ):
        self.data = data
        self.tok_ai = tokenizer_ai
        self.tok_tgt = tokenizer_tgt
        self.L_ai = seq_len_ai
        self.L_tgt = seq_len_tgt
        self.ai_rate = ai_rate

        self.pad = pad_token
        self.sos = sos_token

    # ------------------------------ helpers --------------------------------
    def _encode(self, text: str, tok: Tokenizer, max_len: int) -> List[int]:
        """Encode and trim to max_len."""
        ids = tok.encode(text).ids
        if len(ids) > max_len:
            # keep the *last* tokens – most recent context
            ids = ids[-max_len:]
        return ids

    # ------------------------------ API ------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # turn list → space-separated string so tokenizer keeps numbers intact
        ai_text  = " ".join(map(str, item["PreviousDecision"]))
        tgt_text = " ".join(map(str, item["Decision"]))

        # --- encode & truncate/pad ----------------------------------------
        ai_ids  = self._encode(ai_text,  self.tok_ai,  self.L_ai)
        tgt_ids = self._encode(tgt_text, self.tok_tgt, self.L_tgt)

        ai_pad  = self.L_ai  - len(ai_ids)
        tgt_pad = self.L_tgt - len(tgt_ids)

        aggregate_input = torch.tensor(
            ai_ids + [self.pad] * ai_pad, dtype=torch.long
        )

        # ------------------------------------------------------------------
        # Compute labels that align with decision positions inside window
        # decision positions = ai_rate-1, 2*ai_rate-1, ...
        # gather the *same* tokens from ai_ids
        dec_pos = list(range(self.ai_rate - 1, len(ai_ids), self.ai_rate))
        label_tokens = [ai_ids[p] for p in dec_pos][: self.L_tgt]
        label_pad = self.L_tgt - len(label_tokens)
        label = torch.tensor(
            label_tokens + [self.pad] * label_pad, dtype=torch.long
        )

        return {
            "aggregate_input": aggregate_input,  # shape: (seq_len_ai,)
            "label": label                       # shape: (seq_len_tgt,)
        }
