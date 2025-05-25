# dataset4.py  ──────────────────────────────────────────────────────────
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import numpy as np


# ----------------------------------------------------------------------
def load_json_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """
    The raw file has ONE big dict whose values are lists.  Re-zip those
    lists into a list[dict] so every element is a “session”.
    """
    with open(path, "r") as fp:
        big = json.load(fp)

    keys = list(big.keys())
    n    = len(big[keys[0]])
    return [{k: big[k][i] for k in keys} for i in range(n)]


# ----------------------------------------------------------------------
def _split_tokens(field):
    """Turn any supported field into a list[int] tokens."""
    # field may be str, int, list[str], list[int]
    if isinstance(field, list):
        # a list with one string, or already ints
        if len(field) == 1 and isinstance(field[0], str):
            return [int(t) for t in field[0].split()]
        return [int(x) for x in field]

    if isinstance(field, str):
        return [int(t) for t in field.split()]

    if isinstance(field, (int, np.integer)):
        return [int(field)]

    raise ValueError(f"Unsupported token field type: {type(field)}")


def _scalar_label(field):
    """Return a single int – the last token in `Decision`."""
    return _split_tokens(field)[-1]


# ----------------------------------------------------------------------
class TransformerDataset(Dataset):
    """
    Returns
    -------
    tokens : LongTensor(seq_len)    – token ids
    mask   : BoolTensor(seq_len)    – True where 1..9
    label  : LongTensor()           – scalar Decision
    """

    def __init__(
        self,
        sessions: List[Dict[str, Any]],
        tokenizer: Tokenizer,
        *,
        input_key: str = "PreviousDecision",
        label_key: str = "Decision",
        pad_token: int = 0,
    ):
        self.tok  = tokenizer
        self.pad  = pad_token
        self.data = []

        for sess in sessions:
            tokens = _split_tokens(sess[input_key])
            label  = _scalar_label(sess[label_key])

            ids   = self.tok.encode(" ".join(map(str, tokens))).ids
            ids_t = torch.tensor(ids, dtype=torch.long)
            msk_t = torch.tensor([1 <= t <= 9 for t in ids], dtype=torch.bool)
            lbl_t = torch.tensor(label, dtype=torch.long)

            self.data.append((ids_t, msk_t, lbl_t))

    # basic Dataset API -------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ------------------------------------------------------------------
    @staticmethod
    def collate(samples, *, pad_id: int = 0, max_len: int | None = None):
        toks, masks, labels = zip(*samples)
        L = max(len(t) for t in toks)
        if max_len and L > max_len:
            L = max_len

        def lp(t, fill, dtype):
            if len(t) > L:              # hard-clip (keep most recent)
                t = t[-L:]
            pad = L - len(t)
            if pad:
                t = torch.cat((torch.full((pad,), fill, dtype=dtype), t))
            return t

        tok_t = torch.stack([lp(t, pad_id, torch.long) for t in toks])
        msk_t = torch.stack([lp(m, False, torch.bool) for m in masks])
        lbl_t = torch.stack(labels)

        return tok_t, msk_t, lbl_t
