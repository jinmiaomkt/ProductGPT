# dataset4_decoderonly.py
# ————————————————————————————————————————————————————————————
#  * Keeps only the **last ctx_window tokens** of AggregateInput.
#  * Left-pads so chronology is preserved after trimming.
#  * `seq_len_tgt` can be ≠  ctx_window // ai_rate; we still enforce it.
# ————————————————————————————————————————————————————————————
import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# ─────────────────────────── util ───────────────────────────
def load_json_dataset(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# ─────────────────────────── dataset ────────────────────────
class TransformerDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer_ai:  Tokenizer,
                 tokenizer_tgt: Tokenizer,
                 seq_len_ai:  int,
                 seq_len_tgt: int,
                 num_heads:   int,
                 ai_rate:     int,
                 pad_token:   int = 0,
                 sos_token:   int = 10):
        """
        Parameters
        ----------
        seq_len_ai   : max tokens fed into the model (ctx_window).
        seq_len_tgt  : max decision tokens kept as label.
        ai_rate      : distance between decision tokens in AggregateInput.
        """
        self.data           = data
        self.tk_ai          = tokenizer_ai
        self.tk_tgt         = tokenizer_tgt
        self.seq_len_ai     = seq_len_ai
        self.seq_len_tgt    = seq_len_tgt
        self.ai_rate        = ai_rate
        self.pad_token_id   = pad_token
        self.sos_token_id   = sos_token

    # ————————————————————————————————————————————————
    def __len__(self):
        return len(self.data)

    # ————————————————————————————————————————————————
    def __getitem__(self, idx):
        item = self.data[idx]

        # ---------- tokenise ----------
        ai_tokens = self.tk_ai.encode(
            " ".join(map(str, item["AggregateInput"]))
            if isinstance(item["AggregateInput"], list)
            else str(item["AggregateInput"])
        ).ids

        tgt_tokens = self.tk_tgt.encode(
            " ".join(map(str, item["Decision"]))
            if isinstance(item["Decision"], list)
            else str(item["Decision"])
        ).ids

        # ---------- trim to ctx_window ----------
        if len(ai_tokens) > self.seq_len_ai:
            ai_tokens = ai_tokens[-self.seq_len_ai:]

        if len(tgt_tokens) > self.seq_len_tgt:
            tgt_tokens = tgt_tokens[-self.seq_len_tgt:]

        # ---------- left-pad ----------
        pad_ai  = self.seq_len_ai  - len(ai_tokens)
        pad_tgt = self.seq_len_tgt - len(tgt_tokens)

        aggregate_input = torch.tensor(
            [self.pad_token_id] * pad_ai + ai_tokens, dtype=torch.int64)

        label = torch.tensor(
            [self.pad_token_id] * pad_tgt + tgt_tokens, dtype=torch.int64)

        return {"aggregate_input": aggregate_input,
                "label":           label}
