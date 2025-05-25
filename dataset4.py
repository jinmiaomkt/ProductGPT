# dataset4.py
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


# ---------------------------------------------------------------------
def load_json_dataset(path: str | Path) -> List[Dict]:
    with open(path, "r") as fp:
        return json.load(fp)


# ---------------------------------------------------------------------
class TransformerDataset(Dataset):
    """
    One item = one full session.

    Returns
    -------
    tokens : LongTensor(seq_len)
        Integer token IDs (pad on the LEFT when collated).
    label_mask : BoolTensor(seq_len)
        True wherever the token is in {1,…,9}.
    """

    def __init__(
        self,
        sessions,
        tokenizer: Tokenizer,
        *,
        input_key: str = "AggregateInput",
        pad_token: int = 0,
    ):
        self.tok = tokenizer
        self.pad = pad_token
        self.items = []

        for sess in sessions:
            seq = sess[input_key]
            text = " ".join(map(str, seq)) if isinstance(seq, list) else str(seq)
            ids = self.tok.encode(text).ids

            ids_tensor = torch.as_tensor(ids, dtype=torch.long)
            mask_tensor = torch.as_tensor(
                [1 <= t <= 9 for t in ids], dtype=torch.bool
            )
            self.items.append((ids_tensor, mask_tensor))

    # --- standard Dataset API ----------------------------------------
    def __len__(self):                # noqa: D401
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    @staticmethod
    def collate(samples, *, pad_id=0, max_len=None):
        """
        Left-pad to a common length **and** optionally hard-clip the batch
        to `max_len` tokens (from the *right*, i.e. keep the most recent
        part of the session).
        """
        toks, masks = zip(*samples)
        L = max(len(t) for t in toks)

        if max_len is not None and L > max_len:
            # we will clip later
            L = max_len

        def left_pad(t, value, dtype):
            if len(t) > L:          # clip (right side kept)
                t = t[-L:]
            pad_len = L - len(t)
            if pad_len == 0:
                return t
            pad_tensor = torch.full((pad_len,), value, dtype=dtype)
            return torch.cat([pad_tensor, t])

        batch_tokens = torch.stack(
            [left_pad(t, pad_id, torch.long) for t in toks]
        )
        batch_masks = torch.stack(
            [left_pad(m, False, torch.bool) for m in masks]
        )
        return batch_tokens, batch_masks