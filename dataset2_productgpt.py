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
                 tok_lp, tok_tgt,
                 seq_len_lp, seq_len_tgt,
                 num_heads, lp_rate,
                 pad_token=0):
        self.data      = subset           # reference, not materialised
        self.tok_lp    = tok_lp
        self.tok_tgt   = tok_tgt
        self.seq_lp    = seq_len_lp
        self.seq_tgt   = seq_len_tgt
        self.pad_id    = pad_token

    def __len__(self):
        return len(self.data)             # ONE per session

    def _pad(self, ids, L):
        return ids[:L] + [self.pad_id] * (L - len(ids))
    
    def __getitem__(self, idx):
        rec = self.data[idx]

        # ------------- INPUT -----------------
        src_txt = " ".join(map(str, rec["AggregateInput"])) \
                if isinstance(rec["AggregateInput"], (list, tuple)) else str(rec["AggregateInput"])
        lp_ids  = self._pad(self.tok_lp.encode(src_txt).ids,self.seq_lp)

        # ------------- TARGET ----------------
        tgt_txt = " ".join(map(str, rec["Decision"])) \
                if isinstance(rec["Decision"], (list, tuple)) else str(rec["Decision"])
        tgt_ids = self._pad(self.tok_tgt.encode(tgt_txt).ids, self.seq_tgt)

        return {
            "LTO_PreviousDecision": torch.tensor(lp_ids,  dtype=torch.long),
            "label":           torch.tensor(tgt_ids, dtype=torch.long),
        }
