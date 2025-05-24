# dataset_longformer.py
# ────────────────────────────────────────────────────────────────
# Decision-only dataset *without* a rolling-window constraint.
#
# For each decision token (id 1‥9) we emit:
#   {
#       "aggregate_input":  LongTensor(seq_len ≤ full session)   # full prefix
#       "label"           :  LongTensor(1)                      # decision id
#   }
#
# A custom `collate_fn` left-pads examples in the batch to the
# same length (pad id = 0 by default).
# ────────────────────────────────────────────────────────────────
import json, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List, Dict


# ═══════════════════════════════════════════════════════════════
def load_json_dataset(path: str) -> List[Dict]:
    with open(path, "r") as fp:
        return json.load(fp)
# ═══════════════════════════════════════════════════════════════


class TransformerDataset(Dataset):
    """
    One training example == one decision token (1‥9) together with
    the *entire* token prefix that precedes it.

    Parameters
    ----------
    data            : list[dict]
    tokenizer_ai    : Tokenizer       tokenises AggregateInput
    pad_token       : int             id of [PAD] (default 0)
    """
    def __init__(self,
                 data: List[Dict],
                 tokenizer_ai:  Tokenizer,
                 input_key: str = "AggregateInput",
                 pad_token:     int = 0):
        self.tk_ai   = tokenizer_ai
        self.pad_id  = pad_token
        self.samples = []                     # [(ctx_ids, label_id), …]

        for sess in data:
            stream = sess[input_key]
            ids = self.tk_ai.encode(
                " ".join(map(str, stream)) if isinstance(stream, list)
                else str(stream)
            ).ids

            for idx, tok_id in enumerate(ids):
                if 1 <= tok_id <= 9:          # decision token
                    ctx = ids[:idx]           # FULL prefix (could be empty)
                    self.samples.append((ctx, tok_id))

        print(f"[Dataset] built {len(self.samples)} samples "
              f"from {len(data)} sessions using key='{input_key}'")

    # ------------------------------------------------------------------
    def __len__(self): return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, i):
        ctx, lbl = self.samples[i]
        return {
            "aggregate_input": torch.tensor(ctx,  dtype=torch.long),
            "label"          : torch.tensor([lbl], dtype=torch.long)
        }

    @staticmethod
    def collate_fn(batch, pad_id: int = 0, ctx_window: int = 64, ai_rate: int = 15):
        ctxs  = [b["aggregate_input"] for b in batch]
        lbls  = torch.stack([b["label"] for b in batch])      # (B,1)

        # --- left-pad ------------------------------------------------------
        max_ctx_len = ctx_window * ai_rate
        max_len = min(max_ctx_len, max(x.size(0) for x in ctxs))
        padded  = torch.full((len(ctxs), max_len), pad_id, dtype=torch.long)

        for i, x in enumerate(ctxs):
            x = x[-max_len:]            #  ←  keep only the *last* ‹max_len› tokens
            padded[i, -x.size(0):] = x  # left-pad

        return {"aggregate_input": padded,
                "label": lbls}

