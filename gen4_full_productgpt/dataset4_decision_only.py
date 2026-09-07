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
        rec = self.data[idx]

        # ------------- INPUT -----------------
        src_txt = " ".join(map(str, rec["PreviousDecision"])) \
                if isinstance(rec["PreviousDecision"], (list, tuple)) else str(rec["PreviousDecision"])
        ai_ids  = self._pad(self.tok_ai.encode(src_txt).ids,  self.seq_ai)

        # ------------- TARGET ----------------
        tgt_txt = " ".join(map(str, rec["Decision"])) \
                if isinstance(rec["Decision"], (list, tuple)) else str(rec["Decision"])
        tgt_ids = self._pad(self.tok_tgt.encode(tgt_txt).ids, self.seq_tgt)

        return {
            "aggregate_input": torch.tensor(ai_ids,  dtype=torch.long),
            "label":           torch.tensor(tgt_ids, dtype=torch.long),
        }
