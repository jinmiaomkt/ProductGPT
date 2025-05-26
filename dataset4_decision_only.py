# dataset4_decision_only.py  ──────────────────────────────────────────────
# Turns every decision in the raw JSON into *one* training sample whose
# aggregate_input is the N tokens that precede that decision.
# ------------------------------------------------------------------------
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# ───────────────────────── I/O helper ────────────────────────────────────
def load_json_dataset(filepath: str) -> List[Dict]:
    with open(filepath, "r") as fp:
        return json.load(fp)

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 subset,
                 tok_ai, tok_tgt,
                 seq_len_ai, seq_len_tgt,
                 num_heads, ai_rate,
                 pad_token=0):
        self.data      = subset           # reference, not materialised
        self.tok_ai    = tok_ai
        self.tok_tgt   = tok_tgt
        self.seq_ai    = seq_len_ai
        self.seq_tgt   = seq_len_tgt
        self.pad_id    = pad_token

    def __len__(self):
        return len(self.data)             # ONE per session

    def _pad(self, ids, L):
        return ids[:L] + [self.pad_id] * (L - len(ids))

    def __getitem__(self, idx):
        record = self.data[idx]           # one raw session/user
        ai_ids  = self.tok_ai.encode(record["PreviousDecision"]).ids
        tgt_ids = self.tok_tgt.encode(record["Decision"]).ids

        return {
            "aggregate_input": torch.tensor(self._pad(ai_ids,  self.seq_ai),  dtype=torch.long),
            "label":           torch.tensor(self._pad(tgt_ids, self.seq_tgt), dtype=torch.long),
        }