# # import json
# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# # # Load JSON dataset
# # def load_json_dataset(filepath):
# #     with open(filepath, 'r') as f:
# #         return json.load(f)

# # # Transformer Dataset
# # class TransformerDataset(Dataset):
# #     def __init__(self, data, tokenizer_src, tokenizer_tgt, seq_len_src, seq_len_tgt, num_heads, source_rate, pad_token=0, sos_token=7, eos_token=8):
# #         self.data = data
# #         self.tokenizer_src = tokenizer_src
# #         self.tokenizer_tgt = tokenizer_tgt
# #         # self.tokenizer_lto = tokenizer_lto
# #         self.seq_len_src = seq_len_src 
# #         self.seq_len_tgt = seq_len_tgt
# #         # self.seq_len_lto = seq_len_lto 
# #         self.num_heads = num_heads
# #         self.source_rate = source_rate
# #         # self.lto_rate = lto_rate
# #         self.pad_token = torch.tensor([pad_token], dtype=torch.int64)
# #         self.sos_token = torch.tensor([sos_token], dtype=torch.int64)
# #         self.eos_token = torch.tensor([eos_token], dtype=torch.int64)

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         data_item = self.data[idx]

# #         # Extract F5A and Decision
# #         src_text = " ".join(map(str, data_item["Item"])) if isinstance(data_item["Item"], list) else str(data_item["Item"])
# #         tgt_text = " ".join(map(str, data_item["Decision"])) if isinstance(data_item["Decision"], list) else str(data_item["Decision"])
# #         # lto_text = " ".join(map(str, data_item["F5A"])) if isinstance(data_item["F5A"], list) else str(data_item["F5A"])

# #         # Tokenize source and target
# #         src_input_tokens = self.tokenizer_src.encode(src_text).ids[:self.seq_len_src]
# #         tgt_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids[:self.seq_len_tgt -1]
# #         # lto_input_tokens = self.tokenizer_src.encode(lto_text).ids[:self.seq_len_lto]

# #         # Padding calculations
# #         src_pad = max(0, self.seq_len_src - len(src_input_tokens))
# #         tgt_pad = max(0, self.seq_len_tgt - len(tgt_input_tokens) - 1)
# #         # lto_pad = max(0, self.seq_len_lto - len(lto_input_tokens))

# #         # Encoder input with [SOS], [EOS], and padding
# #         source_input = torch.cat([
# #             # self.sos_token,
# #             torch.tensor(src_input_tokens, dtype=torch.int64),
# #             # self.eos_token,
# #             torch.tensor([self.pad_token] * src_pad, dtype=torch.int64)
# #         ])

# #         # Limited-time Offer input with [SOS], [EOS], and padding
# #         # lto_input = torch.cat([
# #         #     # self.sos_token,
# #         #     torch.tensor(lto_input_tokens, dtype=torch.int64),
# #         #     # self.eos_token,
# #         #     torch.tensor([self.pad_token] * lto_pad, dtype=torch.int64)
# #         # ])

# #         # Decoder input with [SOS] and padding
# #         target_input = torch.cat([
# #             self.sos_token,
# #             torch.tensor(tgt_input_tokens, dtype=torch.int64),
# #             torch.tensor([self.pad_token] * tgt_pad, dtype=torch.int64)
# #         ])

# #         # Label with [EOS] and padding
# #         label = torch.cat([
# #             torch.tensor(tgt_input_tokens, dtype=torch.int64),
# #             self.eos_token,
# #             torch.tensor([self.pad_token] * tgt_pad, dtype=torch.int64)
# #         ])

# #         # Double check the size of the tensors to make sure they are all seq_len long
# #         assert source_input.size(0) == self.seq_len_src 
# #         # assert lto_input.size(0) == self.seq_len_lto
# #         assert target_input.size(0) == self.seq_len_tgt
# #         assert label.size(0) == self.seq_len_tgt    

# #         return {
# #             "source_input": source_input,
# #             # "lto_input": lto_input,
# #             "target_input": target_input,
# #             "long_long_self_mask": (source_input != self.pad_token).unsqueeze(0).int() & causal_mask_square(source_input.size(0)),
# #             "short_short_self_mask": (target_input != self.pad_token).unsqueeze(0).int() & causal_mask_square(target_input.size(0)),
# #             "short_long_cross_mask": (target_input != self.pad_token).unsqueeze(0).unsqueeze(0).repeat_interleave(self.source_rate, dim=-1).repeat(self.num_heads, 1, 1) & causal_mask_rectangular(target_input.size(0)),
# #             # "short_short_cross_mask": (target_input != self.pad_token).unsqueeze(0).int() & causal_mask_square(target_input.size(0)),
# #             "label": label,  # (seq_len)
# #             "src_text": src_text,
# #             "tgt_text": tgt_text
# #         }

# # # Causal mask for autoregressive decoding
# # def causal_mask_square(size):
# #     mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
# #     return (mask == 0)  # shape: (1, size, size), dtype: bool

# # def causal_mask_rectangular(size, rate=10):
# #     base_mask = causal_mask_square(size).type(torch.int)  # (1, size, size)
# #     expanded_mask = base_mask.repeat_interleave(rate, dim=-1)  # (1, 1, size, size * rate)
# #     return (expanded_mask == 1)  # Convert back to boolean

# import json
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# # --------------------------
# # Helper Functions
# # --------------------------

# def load_json_dataset(filepath):
#     with open(filepath, 'r') as f:
#         return json.load(f)

# def split_into_segments(tokens, seg_len):
#     """
#     Splits a list of tokens into segments of maximum length `seg_len`.
#     """
#     return [tokens[i:i+seg_len] for i in range(0, len(tokens), seg_len)]

# def pad_segment(tokens, seg_len, pad_value):
#     """
#     Pads or truncates a list of tokens to exactly seg_len tokens.
#     """
#     if len(tokens) < seg_len:
#         tokens = tokens + [pad_value] * (seg_len - len(tokens))
#     else:
#         tokens = tokens[:seg_len]
#     return tokens

# # def causal_mask_square(size):
# #     """
# #     Creates an upper-triangular (causal) mask of shape (1, size, size).
# #     The mask has True (or 1) in positions that are allowed (i.e. lower-triangular)
# #     and False (or 0) in the positions that should be masked out.
# #     """
# #     mask = torch.triu(torch.ones((1, size, size), dtype=torch.int), diagonal=1)
# #     return (mask == 0)

# # def causal_mask_rectangular(size, rate=10):
# #     """
# #     Creates a causal rectangular mask. This is used for cross attention
# #     where the target length is expanded by the `rate`. Adjust as needed.
# #     """
# #     base_mask = causal_mask_square(size).int()
# #     expanded_mask = base_mask.repeat_interleave(rate, dim=-1)
# #     return (expanded_mask == 1)

# def causal_mask_with_memory(query_len, mem_len):
#     """
#     Creates a causal mask of shape (1, query_len, mem_len + query_len).
    
#     The mask is designed for self-attention when M memory tokens are concatenated
#     to the beginning of the current segment. The first mem_len columns (corresponding
#     to memory tokens) are fully visible, and the subsequent query_len columns follow a 
#     standard lower-triangular causal mask.
#     """
#     # Memory tokens: always visible (ones)
#     mem_mask = torch.ones((query_len, mem_len), dtype=torch.int)
#     # Current tokens: standard lower-triangular mask of shape (query_len, query_len)
#     current_mask = torch.tril(torch.ones((query_len, query_len), dtype=torch.int))
#     # Concatenate along the last dimension so that each query can attend to M memory tokens + current tokens.
#     full_mask = torch.cat([mem_mask, current_mask], dim=-1)  # shape: (query_len, mem_len+query_len)
#     return (full_mask == 1).unsqueeze(0)  # add a batch dimension: (1, query_len, mem_len+query_len)

# def causal_mask_rectangular(query_len, mem_len, rate=10):
#     """
#     Creates a rectangular causal mask for cross attention.
#     (For simplicity, this example simply repeats a square mask.)
#     """
#     # Memory tokens: always visible (ones)
#     mem_mask = torch.ones((query_len * rate, mem_len), dtype=torch.int) 

#     base_mask = causal_mask_with_memory(query_len, 0).int()  # mem_len=0 gives a square mask
#     expanded_mask = base_mask.repeat_interleave(rate, dim=-1) # (query_len * rate, query_len)

#     full_mask = torch.cat([mem_mask, expanded_mask], dim=-1)  # shape: (query_len, mem_len+query_len)

#     return (full_mask == 1).unsqueeze(0)

# # --------------------------
# # Transformer Dataset with Segmentation
# # --------------------------

# class TransformerDataset(Dataset):
#     def __init__(self, data, tokenizer_src, tokenizer_tgt, seq_len_src, seq_len_tgt, num_heads, source_rate, mem_len, pad_token=0, sos_token=7, eos_token=8):
#         """
#         data: loaded JSON data (list of dicts)
#         tokenizer_src, tokenizer_tgt: tokenizers for source and target texts
#         seq_len_src: fixed segment length for source
#         seq_len_tgt: fixed segment length for target (including special tokens)
#         num_heads: number of attention heads (used for creating masks)
#         source_rate: used to expand the cross mask dimensions
#         pad_token, sos_token, eos_token: special tokens
#         """
#         self.data = data
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.seq_len_src = seq_len_src 
#         self.seq_len_tgt = seq_len_tgt
#         self.num_heads = num_heads
#         self.source_rate = source_rate
#         self.mem_len = mem_len
#         self.pad_token = pad_token
#         self.sos_token = sos_token
#         self.eos_token = eos_token

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data_item = self.data[idx]

#         # Extract text fields.
#         src_text = (
#             " ".join(map(str, data_item["Item"]))
#             if isinstance(data_item["Item"], list)
#             else str(data_item["Item"])
#         )
#         tgt_text = (
#             " ".join(map(str, data_item["Decision"]))
#             if isinstance(data_item["Decision"], list)
#             else str(data_item["Decision"])
#         )

#         # Tokenize the texts.
#         src_tokens = self.tokenizer_src.encode(src_text).ids
#         tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

#         # Split tokens into segments.
#         # For the source, each segment is exactly seq_len_src tokens.
#         src_segments = split_into_segments(src_tokens, self.seq_len_src)
#         # For the target, reserve one slot (for SOS or EOS) so use (seq_len_tgt - 1) tokens per segment.
#         tgt_segments = split_into_segments(tgt_tokens, self.seq_len_tgt - 1)

#         # Align the number of segments. (If one side has extra tokens, we ignore the extras.)
#         num_segments = min(len(src_segments), len(tgt_segments))
#         src_segments = src_segments[:num_segments]
#         tgt_segments = tgt_segments[:num_segments]

#         # Prepare lists to hold processed segments and masks.
#         source_segments = []
#         target_input_segments = []
#         label_segments = []
#         long_long_self_masks = []
#         short_short_self_masks = []
#         short_long_cross_masks = []

#         for src_seg, tgt_seg in zip(src_segments, tgt_segments):
#             # Process source segment.
#             padded_src = pad_segment(src_seg, self.seq_len_src, self.pad_token)
#             source_tensor = torch.tensor(padded_src, dtype=torch.int64)
#             source_segments.append(source_tensor)
#             # Create the encoder (self) mask for this segment.
#             src_mask = ((source_tensor != self.pad_token).unsqueeze(0).int() 
#                         & causal_mask_with_memory(self.seq_len_src, self.mem_len))
#             long_long_self_masks.append(src_mask)

#             # Process target segment for decoder input.
#             # Prepend SOS.
#             tgt_input = [self.sos_token] + tgt_seg
#             tgt_input = pad_segment(tgt_input, self.seq_len_tgt, self.pad_token)
#             target_tensor = torch.tensor(tgt_input, dtype=torch.int64)
#             target_input_segments.append(target_tensor)

#             # Process label segment.
#             # Append EOS.
#             label_seq = tgt_seg + [self.eos_token]
#             label_seq = pad_segment(label_seq, self.seq_len_tgt, self.pad_token)
#             label_tensor = torch.tensor(label_seq, dtype=torch.int64)
#             label_segments.append(label_tensor)

#             # Create the decoder self-attention mask.
#             tgt_mask = ((target_tensor != self.pad_token).unsqueeze(0).int() 
#                         & causal_mask_with_memory(self.seq_len_tgt, self.mem_len))
#             short_short_self_masks.append(tgt_mask)
#             # Create the cross-attention mask.
#             cross_mask = ((target_tensor != self.pad_token).unsqueeze(0).unsqueeze(0)
#                           .repeat_interleave(self.source_rate, dim=-1)
#                           .repeat(self.num_heads, 1, 1)
#                           & causal_mask_rectangular(self.seq_len_tgt, self.mem_len, self.source_rate))
#             short_long_cross_masks.append(cross_mask)

#         # Return a dictionary where all segments generated from the same idx are grouped together.
#         return {
#             "source_segments": torch.stack(source_segments, dim=0),        # (num_segments, seq_len_src)
#             "target_input_segments": torch.stack(target_input_segments, dim=0),# (num_segments, seq_len_tgt)
#             "label_segments": torch.stack(label_segments, dim=0),              # (num_segments, seq_len_tgt)
#             "long_long_self_masks": long_long_self_masks,  # List of encoder masks per segment
#             "short_short_self_masks": short_short_self_masks,  # List of decoder self masks per segment
#             "short_long_cross_masks": short_long_cross_masks,  # List of decoder cross masks per segment
#             "src_text": src_text,
#             "tgt_text": tgt_text
#         }

import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# --------------------------
# Helper Functions
# --------------------------

def load_json_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def split_into_segments(tokens, seg_len):
    """
    Splits a list of tokens into segments of maximum length `seg_len`.
    """
    return [tokens[i:i+seg_len] for i in range(0, len(tokens), seg_len)]

def pad_segment(tokens, seg_len, pad_value):
    """
    Pads or truncates a list of tokens to exactly seg_len tokens.
    """
    if len(tokens) < seg_len:
        tokens = tokens + [pad_value] * (seg_len - len(tokens))
    else:
        tokens = tokens[:seg_len]
    return tokens

def causal_mask_with_memory(query_len, mem_len):
    """
    Creates a causal mask of shape (1, query_len, mem_len + query_len).

    For each query position in the current segment (of length query_len):
      - The first mem_len positions (corresponding to memory tokens) are fully visible.
      - The following query_len positions follow a lower-triangular (causal) mask.
    """
    mem_mask = torch.ones((query_len, mem_len), dtype=torch.bool)
    # Current part: lower triangular mask.
    current_mask = torch.tril(torch.ones((query_len, query_len), dtype=torch.bool))
    full_mask = torch.cat([mem_mask, current_mask], dim=-1)  # shape: (query_len, mem_len+query_len)
    return full_mask.unsqueeze(0)  # add batch dimension -> (1, query_len, mem_len+query_len)

def causal_mask_rectangular(query_len, mem_len, rate=10):
    """
    Creates a causal rectangular mask for cross attention.
    """
    mem_mask = torch.ones((query_len, mem_len), dtype=torch.bool) # shape: (query_len, mem_len)
    base_mask = causal_mask_with_memory(query_len, 0).int()  
    expanded_mask = base_mask.repeat_interleave(rate, dim=-1) # shape: (query_len, query_len * rate)
    full_mask = torch.cat([mem_mask, expanded_mask], dim=-1)  # shape: (query_len, mem_len + query_len * rate)
    
    return (full_mask == 1)



import torch

def causal_mask_with_memory(query_len, mem_len):
    """
    Creates a causal mask of shape (1, 1, query_len, mem_len + query_len).

    - The first mem_len positions (memory tokens) are fully visible.
    - The next query_len positions follow a lower-triangular (causal) mask.
    """
    mem_mask = torch.ones((query_len, mem_len), dtype=torch.bool)           # (query_len, mem_len)
    current_mask = torch.tril(torch.ones((query_len, query_len), dtype=torch.bool))  # (query_len, query_len)
    full_mask_2d = torch.cat([mem_mask, current_mask], dim=-1)             # (query_len, mem_len+query_len)

    # Now make it 4D: (1, 1, query_len, mem_len+query_len)
    return full_mask_2d.unsqueeze(0).unsqueeze(0)

def causal_mask_rectangular(query_len, mem_len, rate=10):
    """
    Creates a rectangular causal mask for cross attention of shape:
    (1, 1, query_len, mem_len + query_len * rate).
    """
    # memory part is fully visible:
    mem_mask = torch.ones((query_len, mem_len), dtype=torch.bool)  # (query_len, mem_len)

    # Causal part for the repeated "expanded" segment:
    base_mask = causal_mask_with_memory(query_len, 0)  # shape: (1, 1, query_len, query_len)
    # base_mask is 4D, but no memory => last dim is just query_len

    # We only need to expand the last dimension (the key_len) by `rate`.
    # base_mask.shape => (1, 1, query_len, query_len)
    # We repeat_interleave along dim=-1
    expanded_mask = base_mask.repeat_interleave(rate, dim=-1)  # shape: (1, 1, query_len, query_len * rate)

    # Now create a (query_len, mem_len + query_len*rate) in 2D:
    #   first part is mem_mask, second part is expanded_mask (minus the leading 2 dims).
    mem_mask_2d = mem_mask
    expanded_mask_2d = expanded_mask[0, 0]  # shape: (query_len, query_len*rate)
    full_mask_2d = torch.cat([mem_mask_2d, expanded_mask_2d], dim=-1) # (query_len, mem_len + query_len*rate)

    # Return as 4D: (1, 1, query_len, mem_len + query_len*rate)
    return full_mask_2d.unsqueeze(0).unsqueeze(0)



# --------------------------
# Transformer Dataset with Segmentation and Memory-Aware Masks
# --------------------------

class TransformerDataset(Dataset):
    def __init__(self, data, tokenizer_src, tokenizer_tgt, seq_len_src, seq_len_tgt, num_heads, source_rate, mem_len_src, mem_len_tgt, pad_token=0, sos_token=7, eos_token=8):
        """
        data: loaded JSON data (list of dicts)
        tokenizer_src, tokenizer_tgt: tokenizers for source and target texts
        seq_len_src: fixed segment length for source (current tokens only)
        seq_len_tgt: fixed segment length for target (current tokens only; special tokens are added separately)
        num_heads: number of attention heads (used for creating masks)
        source_rate: used to expand the cross mask dimensions
        mem_len: number of memory tokens that will be concatenated to each segment
        pad_token, sos_token, eos_token: special tokens
        """
        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len_src = seq_len_src 
        self.seq_len_tgt = seq_len_tgt
        self.num_heads = num_heads
        self.source_rate = source_rate
        self.mem_len_src = mem_len_src
        self.mem_len_tgt = mem_len_tgt
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        # Extract text fields.
        src_text = (
            " ".join(map(str, data_item["Item"]))
            if isinstance(data_item["Item"], list)
            else str(data_item["Item"])
        )
        tgt_text = (
            " ".join(map(str, data_item["Decision"]))
            if isinstance(data_item["Decision"], list)
            else str(data_item["Decision"])
        )

        # Tokenize the texts.
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Split tokens into segments.
        src_segments = split_into_segments(src_tokens, self.seq_len_src)
        tgt_segments = split_into_segments(tgt_tokens, self.seq_len_tgt - 1)  # reserve one slot for SOS/EOS

        # Align number of segments (ignore extra tokens on either side).
        num_segments = min(len(src_segments), len(tgt_segments))
        src_segments = src_segments[:num_segments]
        tgt_segments = tgt_segments[:num_segments]

        # Prepare lists to hold processed segments and masks.
        source_segments = []
        target_input_segments = []
        label_segments = []
        long_long_self_masks = []
        short_short_self_masks = []
        short_long_cross_masks = []

        for src_seg, tgt_seg in zip(src_segments, tgt_segments):
            print("Source Segments")  

            # --- Process Source Segment ---
            padded_src = pad_segment(src_seg, self.seq_len_src, self.pad_token)
            source_tensor = torch.tensor(padded_src, dtype=torch.int64)  # shape: (L,)
            source_segments.append(source_tensor)

            # Build a key mask for current tokens: mark non-pad positions.
            current_token_mask = (source_tensor != self.pad_token)  # shape: (L,)
            print("current_token_mask shape:", current_token_mask.shape)  # Expect (L,)

            memory_token_mask = torch.ones(self.mem_len_src, dtype=torch.bool)
            print("memory_token_mask shape:", memory_token_mask.shape)  # Expect (mem_len,)

            # Concatenate to form a key mask of shape (mem_len + L,)
            key_mask = torch.cat([memory_token_mask, current_token_mask], dim=0)
            print("key_mask shape:", key_mask.shape)  # Expect (mem_len+L,)

            causal = causal_mask_with_memory(self.seq_len_src, self.mem_len_src).squeeze(0)
            print("causal mask shape after squeeze:", causal.shape)  # Expect (L, mem_len+L)

            # Final encoder mask: combine causal mask with token mask.
            combined_mask = key_mask & causal
            print("combined_mask shape:", combined_mask.shape)  # Expect (L, mem_len+L)

            src_mask = combined_mask.unsqueeze(0)  # shape: (1, L, mem_len+L)
            print("src_mask shape:", src_mask.shape)  # Expect (1, L, mem_len+L)

            long_long_self_masks.append(src_mask)
            

            # --- Process Target Segment (Decoder) ---
            # For decoder input, prepend SOS.
            print("Target Segments")  # Expect (1, L, mem_len+L)

            tgt_input = [self.sos_token] + tgt_seg
            tgt_input = pad_segment(tgt_input, self.seq_len_tgt, self.pad_token)
            target_tensor = torch.tensor(tgt_input, dtype=torch.int64)
            target_input_segments.append(target_tensor)
            # For label, append EOS.
            label_seq = tgt_seg + [self.eos_token]
            label_seq = pad_segment(label_seq, self.seq_len_tgt, self.pad_token)
            label_tensor = torch.tensor(label_seq, dtype=torch.int64)
            label_segments.append(label_tensor)

            # Build decoder mask similarly.
            current_token_mask_dec = (target_tensor != self.pad_token)
            print("current_token_mask_dec shape:", current_token_mask_dec.shape)  # Expect (L,)

            memory_token_mask_dec = torch.ones(self.mem_len_tgt, dtype=torch.bool)
            print("memory_token_mask_dec shape:", memory_token_mask_dec.shape)  # Expect (L,)

            key_mask_dec = torch.cat([memory_token_mask_dec, current_token_mask_dec], dim=0)
            print("key_mask_dec shape:", key_mask_dec.shape)  # Expect (L,)

            # query_mask_dec = torch.ones(self.seq_len_tgt, dtype=torch.bool)
            # final_mask_dec = query_mask_dec.unsqueeze(1) & key_mask_dec.unsqueeze(0)
            causal_dec = causal_mask_with_memory(self.seq_len_tgt, self.mem_len_tgt).squeeze(0)
            print("causal_dec shape:", causal_dec.shape)  # Expect (L,)

            combined_mask_dec = key_mask_dec & causal_dec
            print("combined_mask_dec shape:", combined_mask_dec.shape)  # Expect (L,)

            tgt_mask = combined_mask_dec.unsqueeze(0)
            print("tgt_mask shape:", tgt_mask.shape)  # Expect (L,)
            
            short_short_self_masks.append(tgt_mask)

            print("Cross Attention")  # Expect (1, L, mem_len+L)

            # Create cross-attention mask (here we simply use a rectangular causal mask for target length).
            #               "short_long_cross_mask": (target_input != self.pad_token).unsqueeze(0).unsqueeze(0).repeat_interleave(self.source_rate, dim=-1).repeat(self.num_heads, 1, 1) & causal_mask_rectangular(target_input.size(0)),

            current_token_mask_cross = (target_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).repeat_interleave(self.source_rate, dim=-1).repeat(self.num_heads, 1, 1)
            print("current_token_mask_cross shape:", current_token_mask_cross.shape)  # Expect (L,)

            mem_mask_cross = torch.ones((self.seq_len_tgt, self.mem_len_tgt), dtype=torch.bool) # shape: (query_len, mem_len)
            print("mem_mask_cross shape:", mem_mask_cross.shape)  # Expect (L,)
            
            key_mask_cross = torch.cat([mem_mask_cross, current_token_mask_cross], dim=0)
            print("key_mask_cross shape:", key_mask_cross.shape)  # Expect (L,)

            causal_cross = causal_mask_rectangular(self.seq_len_tgt, self.mem_len_tgt).squeeze(0)
            print("causal_cross shape:", causal_cross.shape)  # Expect (L,)

            combined_mask_cross = key_mask_cross & causal_cross
            print("combined_mask_cross shape:", combined_mask_cross.shape)  # Expect (L,)

            cross_mask = combined_mask_cross.unsqueeze(0)
            print("cross_mask shape:", cross_mask.shape)  # Expect (L,)

            short_long_cross_masks.append(cross_mask)

        return {
            "source_segments": torch.stack(source_segments, dim=0),         # (num_segments, seq_len_src)
            "target_input_segments": torch.stack(target_input_segments, dim=0), # (num_segments, seq_len_tgt)
            "label_segments": torch.stack(label_segments, dim=0),               # (num_segments, seq_len_tgt)
            "long_long_self_masks": long_long_self_masks,   # List of encoder masks per segment
            "short_short_self_masks": short_short_self_masks, # List of decoder masks per segment
            "short_long_cross_masks": short_long_cross_masks, # List of decoder cross masks per segment
            "src_text": src_text,
            "tgt_text": tgt_text
        }
