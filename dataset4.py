# dataset4.py
import json, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List, Dict


def load_json_dataset(path: str) -> List[Dict]:
    with open(path, "r") as fp:
        return json.load(fp)

class TransformerDataset(torch.utils.data.Dataset):
    """
    One item = the whole session.
    Returns
        • tokens  – LongTensor(seq)
        • label_mask – BoolTensor(seq)  True where token ∈ {1,…,9}
    """
    def __init__(self, sessions, tokenizer, *, input_key="AggregateInput",
                 pad_token=0):
        self.tok  = tokenizer
        self.pad  = pad_token
        self.items = []

        for sess in sessions:
            seq = sess[input_key]
            ids = self.tok.encode(" ".join(map(str, seq)) if isinstance(seq, list)
                                  else str(seq)).ids
            mask = torch.tensor([1 <= t <= 9 for t in ids], dtype=torch.bool)
            self.items.append((torch.tensor(ids), mask))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

    # ------------------------------------------------------------------
    @staticmethod
    def collate(samples, pad_id=0):
        toks, masks = zip(*samples)
        L = max(len(t) for t in toks)
        pad = lambda arr, val: torch.cat(
            [torch.full((L-len(arr),), val, dtype=arr.dtype), arr])
        batch_tokens = torch.stack([pad(t, pad_id) for t in toks])   # (B,L)
        batch_mask   = torch.stack([pad(m, False)  for m in masks])  # (B,L)
        return batch_tokens, batch_mask


