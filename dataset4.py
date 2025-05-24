# dataset4.py
import json, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List, Dict


def load_json_dataset(path: str) -> List[Dict]:
    with open(path, "r") as fp:
        return json.load(fp)


class TransformerDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 tokenizer_ai: Tokenizer,
                 input_key: str = "AggregateInput",
                 pad_token: int = 0,
                 ctx_window: int = 64,
                 ai_rate: int = 15):

        self.tk_ai = tokenizer_ai
        self.pad_id = pad_token
        self.ctx_cap = ctx_window * ai_rate
        self.sessions = []    # list of tokenized sessions
        self.samples = []     # list of (session_id, decision_token_idx, label)

        for sess in data:
            stream = sess[input_key]
            ids = self.tk_ai.encode(
                " ".join(map(str, stream)) if isinstance(stream, list)
                else str(stream)
            ).ids
            sid = len(self.sessions)
            self.sessions.append(ids)

            for idx, tok_id in enumerate(ids):
                if 1 <= tok_id <= 9:
                    self.samples.append((sid, idx, tok_id))

        print(f"[Dataset] built {len(self.samples)} samples from {len(data)} sessions using key='{input_key}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sess_id, idx, lbl = self.samples[i]
        ids = self.sessions[sess_id]
        start = max(0, idx - self.ctx_cap)
        ctx = ids[start:idx]
        return {
            "aggregate_input": torch.tensor(ctx, dtype=torch.long),
            "label": torch.tensor([lbl], dtype=torch.long)
        }

    @staticmethod
    def collate_fn(batch, pad_id: int = 0, ctx_window: int = 64, ai_rate: int = 15):
        ctxs = [b["aggregate_input"] for b in batch]
        lbls = torch.stack([b["label"] for b in batch])

        max_ctx_len = ctx_window * ai_rate
        max_len = min(max_ctx_len, max(x.size(0) for x in ctxs))
        padded = torch.full((len(ctxs), max_len), pad_id, dtype=torch.long)

        for i, x in enumerate(ctxs):
            x = x[-max_len:]
            padded[i, -x.size(0):] = x

        return {"aggregate_input": padded, "label": lbls}