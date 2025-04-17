import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Load JSON dataset
def load_json_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Transformer Dataset
class TransformerDataset(Dataset):
    def __init__(self, data, tokenizer_ai, tokenizer_tgt, seq_len_ai, seq_len_tgt, num_heads, ai_rate, pad_token=0, sos_token=10):
        self.data = data
        self.tokenizer_ai = tokenizer_ai
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len_ai = seq_len_ai
        self.seq_len_tgt = seq_len_tgt
        self.ai_rate = ai_rate
        self.pad_token = torch.tensor([pad_token], dtype=torch.int64)
        self.sos_token = torch.tensor([sos_token], dtype=torch.int64)
        # self.eos_token = torch.tensor([eos_token], dtype=torch.int64)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        ai_text = " ".join(map(str, data_item["PreviousDecision"])) if isinstance(data_item["PreviousDecision"], list) else str(data_item["PreviousDecision"])
        tgt_text = " ".join(map(str, data_item["Decision"])) if isinstance(data_item["Decision"], list) else str(data_item["Decision"])

        aggregate_input_tokens = self.tokenizer_ai.encode(ai_text).ids[:self.seq_len_ai]
        tgt_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids[:self.seq_len_tgt]

        ai_pad = max(0, self.seq_len_ai - len(aggregate_input_tokens))
        tgt_pad = max(0, self.seq_len_tgt - len(tgt_input_tokens))
        
        # Encoder input with [SOS], [EOS], and padding
        aggregate_input = torch.cat([
            # self.sos_token,
            torch.tensor(aggregate_input_tokens, dtype=torch.int64),
            # self.eos_token,
            torch.tensor([self.pad_token] * ai_pad, dtype=torch.int64)
        ])

        # Label with [EOS] and padding
        label = torch.cat([
            torch.tensor(tgt_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_pad, dtype=torch.int64)
        ])

        return {
            "aggregate_input": aggregate_input,
            "label": label
        }