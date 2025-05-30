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
    def __init__(self, data, tokenizer_src, tokenizer_tgt, seq_len_src, seq_len_tgt, num_heads, source_rate, pad_token=9, sos_token=7, eos_token=0):
        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # self.tokenizer_lto = tokenizer_lto
        self.seq_len_src = seq_len_src 
        self.seq_len_tgt = seq_len_tgt
        # self.seq_len_lto = seq_len_lto 
        self.num_heads = num_heads
        self.source_rate = source_rate
        # self.lto_rate = lto_rate
        self.pad_token = torch.tensor([pad_token], dtype=torch.int64)
        self.sos_token = torch.tensor([sos_token], dtype=torch.int64)
        self.eos_token = torch.tensor([eos_token], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        # Extract F5A and Decision
        src_text = " ".join(map(str, data_item["Item"])) if isinstance(data_item["Item"], list) else str(data_item["Item"])
        tgt_text = " ".join(map(str, data_item["Decision"])) if isinstance(data_item["Decision"], list) else str(data_item["Decision"])
        # lto_text = " ".join(map(str, data_item["F5A"])) if isinstance(data_item["F5A"], list) else str(data_item["F5A"])

        # Tokenize source and target
        src_input_tokens = self.tokenizer_src.encode(src_text).ids[:self.seq_len_src]
        tgt_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids[:self.seq_len_tgt - 1]
        # tgt_input_tokens_b = self.tokenizer_tgt.encode(tgt_text).ids[:self.seq_len_tgt]
        # lto_input_tokens = self.tokenizer_src.encode(lto_text).ids[:self.seq_len_lto]

        # Padding calculations
        src_pad = max(0, self.seq_len_src - len(src_input_tokens))
        tgt_pad = max(0, self.seq_len_tgt - len(tgt_input_tokens) - 1)
        # lto_pad = max(0, self.seq_len_lto - len(lto_input_tokens))

        # Encoder input with [SOS], [EOS], and padding
        source_input = torch.cat([
            # self.sos_token,
            torch.tensor(src_input_tokens, dtype=torch.int64),
            # self.eos_token,
            torch.tensor([self.pad_token] * src_pad, dtype=torch.int64)
        ])

        # Limited-time Offer input with [SOS], [EOS], and padding
        # lto_input = torch.cat([
        #     # self.sos_token,
        #     torch.tensor(lto_input_tokens, dtype=torch.int64),
        #     # self.eos_token,
        #     torch.tensor([self.pad_token] * lto_pad, dtype=torch.int64)
        # ])

        # Decoder input with [SOS] and padding
        target_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_pad, dtype=torch.int64)
        ])

        # Label with [EOS] and padding
        label = torch.cat([
            torch.tensor(tgt_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_pad, dtype=torch.int64)
        ])

        # Double check the size of the tensors to make sure they are all seq_len long
        assert source_input.size(0) == self.seq_len_src 
        # assert lto_input.size(0) == self.seq_len_lto
        assert target_input.size(0) == self.seq_len_tgt
        assert label.size(0) == self.seq_len_tgt    

        return {
            "source_input": source_input,
            # "lto_input": lto_input,
            "target_input": target_input,
            #"long_long_self_mask": (source_input != self.pad_token).unsqueeze(0).int() & causal_mask_square(source_input.size(0)),
            #"short_short_self_mask": (target_input != self.pad_token).unsqueeze(0).int() & causal_mask_square(target_input.size(0)),
            #"short_long_cross_mask": (target_input != self.pad_token).unsqueeze(0).unsqueeze(0).repeat_interleave(self.source_rate, dim=-1).repeat(self.num_heads, 1, 1) & causal_mask_rectangular(target_input.size(0)),
            # "short_short_cross_mask": (target_input != self.pad_token).unsqueeze(0).int() & causal_mask_square(target_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

# # Causal mask for autoregressive decoding
# def causal_mask_square(size):
#     mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
#     return (mask == 0)  # shape: (1, size, size), dtype: bool

# def causal_mask_rectangular(size, rate=10):
#     base_mask = causal_mask_square(size).type(torch.int)  # (1, size, size)
#     expanded_mask = base_mask.repeat_interleave(rate, dim=-1)  # (1, 1, size, size * rate)
#     return (expanded_mask == 1)  # Convert back to boolean
