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

# ───────────────────────── Dataset class ─────────────────────────────────
class TransformerDataset(Dataset):
    """
    Each item returned corresponds to **one** decision token (`label`)
    and the `seq_len_ai` aggregate-input tokens that *immediately precede*
    that decision.

    Parameters
    ----------
    seq_len_ai   : rolling-window length fed into the model
    seq_len_tgt  : kept for compatibility – usually 1 now
    ai_rate      : distance (in AggregateInput) between two consecutive
                   decision positions.   decision_j is located at
                   index (j+1)*ai_rate − 1 in AggregateInput.
    """
    def __init__(
        self,
        data,
        tokenizer_ai:  Tokenizer,
        tokenizer_tgt: Tokenizer,
        seq_len_ai:   int,
        seq_len_tgt:  int,
        num_heads:    int,      # not used, kept for API parity
        ai_rate:      int,
        pad_token:    int = 0,
        sos_token:    int = 10,
    ):
        self.tok_ai   = tokenizer_ai
        self.tok_tgt  = tokenizer_tgt
        self.L_ai     = seq_len_ai
        self.L_tgt    = seq_len_tgt          # normally 1 for rolling window
        self.ai_rate  = ai_rate
        self.pad_id   = pad_token

        # -------- pre-materialise every slice in memory -----------------
        self.samples = []       # each element is (agg_ids, decision_id)
        for obj in data:
            agg_ids = self.tok_ai.encode(
                " ".join(map(str, obj["PreviousDecision"]))
            ).ids
            dec_ids = self.tok_tgt.encode(
                " ".join(map(str, obj["Decision"]))
            ).ids

            for j, dec_token in enumerate(dec_ids):
                # position of this decision in AggregateInput
                pos = (j + 1) * ai_rate - 1
                if pos >= len(agg_ids):
                    # malformed sample – skip if AggregateInput is too short
                    continue

                # ------------- rolling window (inclusive of pos) --------
                start = max(0, pos - self.L_ai + 1)
                window = agg_ids[start : pos + 1]        # oldest → current
                # left-pad so len == L_ai
                if len(window) < self.L_ai:
                    window = [self.pad_id] * (self.L_ai - len(window)) + window

                self.samples.append((window, dec_token))

    # ───────────────────── Dataset protocol ─────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, dec_token = self.samples[idx]

        aggregate_input = torch.tensor(window, dtype=torch.long)

        # -------- labels -------------------------------------------------
        # “current decision only” → length-1; still pad to L_tgt so the rest
        # of your pipeline does not break.  Change here if you want a block.
        label_vec = [dec_token] + [self.pad_id] * (self.L_tgt - 1)
        label = torch.tensor(label_vec, dtype=torch.long)

        return {
            "aggregate_input": aggregate_input,   # shape: (L_ai,)
            "label":           label              # shape: (L_tgt,)
        }
