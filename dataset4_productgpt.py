# dataset4_decision_only.py  ──────────────────────────────────────────────
# Turns every decision in the raw JSON into *one* training sample whose
# aggregate_input is the N tokens that precede that decision.
# ------------------------------------------------------------------------
import json, itertools
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# ───────────────────────── I/O helper ────────────────────────────────────
# def load_json_dataset(filepath: str) -> List[Dict]:
#     with open(filepath, "r") as fp:
#         return json.load(fp)

# def load_json_dataset(path, keep_uids=None):
#     raw = json.loads(Path(path).read_text())
#     # … existing explode logic …
#     if keep_uids is not None:
#         data = [r for r in data if str(r["uid"]) in keep_uids]
#     return data

def load_json_dataset(path: str | Path,
                      keep_uids: set[str] | None = None):
    """
    Return exploded list of records; optionally keep only rows whose UID
    string is in keep_uids.
    """
    raw = json.loads(Path(path).read_text())

    # your original explode_record(...) helper here ------------
    def explode_record(rec):
        uid = str(rec["uid"][0] if isinstance(rec["uid"], list) else rec["uid"])
        # …
        # yield {"uid": uid, ...}

    # build the *full* list first
    data = list(itertools.chain.from_iterable(
        explode_record(r) for r in (raw if isinstance(raw, list) else
                                    [{k: raw[k][i] for k in raw} 
                                     for i in range(len(raw["uid"]))])
    ))

    # optional filtering
    if keep_uids is not None:
        data = [row for row in data if row["uid"] in keep_uids]

    return data
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
        src_txt = " ".join(map(str, rec["AggregateInput"])) \
                if isinstance(rec["AggregateInput"], (list, tuple)) else str(rec["AggregateInput"])
        ai_ids  = self._pad(self.tok_ai.encode(src_txt).ids,  self.seq_ai)

        # ------------- TARGET ----------------
        tgt_txt = " ".join(map(str, rec["Decision"])) \
                if isinstance(rec["Decision"], (list, tuple)) else str(rec["Decision"])
        tgt_ids = self._pad(self.tok_tgt.encode(tgt_txt).ids, self.seq_tgt)

        return {
            "aggregate_input": torch.tensor(ai_ids,  dtype=torch.long),
            "label":           torch.tensor(tgt_ids, dtype=torch.long),
        }
