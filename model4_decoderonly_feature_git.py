import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

df = pd.read_excel("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx", sheet_name=0)

## NewProductIndex3 is not a feature
feature_cols = [
    "Rarity", 
    "MaxLife", 
    "MaxOffense", 
    "MaxDefense",
    "WeaponTypeOneHandSword", 
    "WeaponTypeTwoHandSword", 
    "WeaponTypeArrow", 
    "WeaponTypeMagic", 
    "WeaponTypePolearm",
    "EthnicityIce", 
    "EthnicityRock", 
    "EthnicityWater", 
    "EthnicityFire", 
    # "EthnicityGrass", 
    "EthnicityThunder", 
    "EthnicityWind", 
    "GenderFemale", 
    "GenderMale",
    # "CountryFengDan", 
    "CountryRuiYue", 
    "CountryDaoQi", 
    "CountryZhiDong", 
    "CountryMengDe", 
    # "CountryXuMi",
    "type_figure",  # if it's numeric or one-hot encoded; else skip if it's a string
    "MinimumAttack", 
    "MaximumAttack",
    "MinSpecialEffect", 
    "MaxSpecialEffect",
    "SpecialEffectEfficiency", 
    "SpecialEffectExpertise",
    "SpecialEffectAttack", 
    "SpecialEffectSuper",
    "SpecialEffectRatio", 
    "SpecialEffectPhysical", 
    "SpecialEffectLife", 
    # "NewProductIndex3",  # only if you actually want it as a feature
    "LTO" 
]

# --------------------------------------------
# decision / product partition of the vocab
# --------------------------------------------
DECISION_IDS = torch.tensor([1,2,3,4,5,6,7,8,9])          # len = 9
PRODUCT_IDS  = torch.tensor(list(range(13, 57)))          # 13 … 56

## Determine max_token_id and the dimension
max_token_id =  68
# This is the count of numeric columns (your "feature_dim"). 
# For example, if you have 38 numeric columns (NOT counting the ID).
feature_dim = 34

feature_array = np.zeros((max_token_id + 1, feature_dim), dtype=np.float32)

for idx in range(len(df)):
    # The product ID for this row (should be in [1..65])
    token_id = int(df["NewProductIndex6"].iloc[idx])  
    
    # Gather the numeric features from the row. 
    # If you only want the columns in `feature_cols` (excluding "NewProductIndex3"), do:
    feats = df.loc[idx, feature_cols].values  # shape (34,) => [Rarity, MaxLife, MaxOffense, ...]

    feature_array[token_id, :] = feats

feature_tensor = torch.from_numpy(feature_array)  

##############################################################################
# 1. gelu_approx
##############################################################################
def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate GeLU using the Tanh approximation.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))

class SpecialPlusFeatureLookup(nn.Module):
    """
    A single module that, given a batch of token IDs [0..109],
    produces an embedding of dimension d_model.

    - For special tokens (like {0,107,108,109}), we use a small learned embedding.
    - For real product IDs [1..65], we gather from a
      precomputed feature_tensor and pass through an MLP to get a d_model vector.
    """

    def __init__(self,
                d_model: int,
                feature_tensor: torch.Tensor,
                product_ids: list[int], 
                vocab_size_src: int):
        super().__init__()
        """
        :param d_model: dimension of the final embedding for each token
        :param feature_tensor: shape (max_token_id+1, feature_dim)
                              A buffer/parameter with row i = feature vector for token i
        :param special_token_ids: list of token IDs that have no product features (like [0,107,108,109])
        :param hidden_dim: optional dimension for an MLP hidden layer if you want more nonlinearity
        """
        self.d_model = d_model
        self.feature_dim = feature_tensor.shape[1]
        self.max_id = feature_tensor.shape[0] - 1

        self.id_embed = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(feature_tensor.size(1), d_model, bias=False)
        self.register_buffer("feature_table", feature_tensor, persistent=False)

        product_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        product_mask[product_ids] = True
        self.register_buffer("product_mask", product_mask, persistent=False)   

        # Build a dictionary mapping special IDs -> index in special embedding
        # self.special_id_map = {}
        # for idx, tok_id in enumerate(special_token_ids):
        #     self.special_id_map[tok_id] = idx

        # self.num_special = len(special_token_ids)
        # self.special_embed = nn.Embedding(self.num_special, d_model)

        # # MLP or linear to map product features -> d_model
        # if hidden_dim > 0:
        #     self.product_mlp = nn.Sequential(
        #         nn.Linear(self.feature_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, d_model),
        #     )
        # else:
        #     # Single linear layer
        #     self.product_mlp = nn.Linear(self.feature_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        :param token_ids: (batch_size, seq_len) integer IDs
        :return: (batch_size, seq_len, d_model) final embeddings
        """     
        ids_long = token_ids.long()
        id_vec   = self.id_embed(ids_long)                                 # (B,T,D)

        feature_raw = self.feature_table[ids_long]                            # (B,T,F)
        feat_vec = self.feat_proj(feature_raw)                              # (B,T,D)

        # ---- feature branch only for product tokens ------------------
        prod_mask_bt = self.product_mask[ids_long]                          # (B,T) bool
        if prod_mask_bt.any():
            feats_raw   = self.feat_tbl[ids_long[prod_mask_bt]]             # (N,F)
            feats_proj  = self.feat_proj(feats_raw)                         # (N,D)

            feat_vec = torch.zeros_like(id_vec)                             # (B,T,D)
            feat_vec[prod_mask_bt] = feats_proj
        else:
            feat_vec = torch.zeros_like(id_vec)

        return id_vec + feat_vec                                            # (B,T,D)
    
        # device = token_ids.device
        # B, S = token_ids.shape

        # # We'll create an output buffer (B,S,d_model)
        # out = torch.zeros((B, S, self.d_model), device=device)

        # # 1) Identify special vs. product tokens
        # # We'll build an index map telling us which row in 'special_embed' each token corresponds to,
        # # or -1 if it's not a special token.

        # special_idx = torch.full_like(token_ids, -1, device=device)
        # # for i, (special_id) in enumerate(self.special_id_map.items()):
        # #     # special_id is (tok_id -> index_in_special_embed)
        # #     # Actually we need to invert:  special_id_map[tok_id] = index_in_special_embed
        # #     pass

        # for special_token, embed_index in self.special_id_map.items():
        #     mask = (token_ids == special_token)
        #     special_idx[mask] = embed_index

        # is_special = (special_idx >= 0)
        # is_product = ~is_special

        # # 2) Embed special tokens
        # special_positions = special_idx[is_special]  # shape: (num_special_positions,)
        # special_embeds = self.special_embed(special_positions)  # (num_special_positions, d_model)
        # out[is_special] = special_embeds

        # # 3) Real product tokens
        # # Gather each product's row from our feature_table
        # # shape: (num_product_positions, feature_dim)
        # product_feature_vecs = self.feature_table[token_ids[is_product]]

        # # Pass them through the MLP/linear
        # product_embeds = self.product_mlp(product_feature_vecs)  # (num_product_positions, d_model)
        # product_embeds = gelu_approx(product_embeds)
        # # Typically we multiply by sqrt(d_model) for Transformer scale
        # product_embeds = product_embeds * math.sqrt(self.d_model)

        # out[is_product] = product_embeds
        # return out
    
##############################################################################
# 2. LayerNormalization
##############################################################################
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias  = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

##############################################################################
# 3. FeedForwardBlock
##############################################################################
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(
            self.dropout(gelu_approx(self.linear_1(x)))
        )

##############################################################################
# 4. InputEmbeddings
##############################################################################
class InputEmbeddings(nn.Module):
    """
    Learns a token embedding for an integer vocabulary,
    and optionally scales by sqrt(d_model).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len)
        # returns: (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

##############################################################################
# 5. PositionalEncoding
##############################################################################
class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding, up to a fixed max_seq_len.
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0)/d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape => (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # Add (batch=1) from self.pe up to seq_len
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)

##############################################################################
# 6. ResidualConnection
##############################################################################
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

##############################################################################
# 7. CausalPerformer (Self-Attention)
##############################################################################
class CausalPerformer(nn.Module):
    """
    Your blockwise causal performer from previous code, used for self-attention.
    """
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float=0.1,
                 kernel_type: str="exp",
                 block_size_h: int=1,
                 block_size_w: int=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        self.nb_features = max(1, math.ceil(math.log(d_model)))
        self.kernel_type = kernel_type

        self.block_size_h = block_size_h
        self.block_size_w = block_size_w

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._create_feature_map()

    def _create_feature_map(self):
        omega = torch.randn(self.nb_features, self.d_k) / math.sqrt(self.d_k)
        # Typically we don't train omega in Performer random feature approach
        self.omega = nn.Parameter(omega, requires_grad=False)

    def _kernel_function(self, x: torch.Tensor):
        # "exp" or "gelu" kernel
        if self.kernel_type == "gelu":
            return gelu_approx(x) + 1e-6
        elif self.kernel_type == "exp":
            return torch.exp(-0.5 * (x**2))
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel_type}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q,k,v: (B, seq_len, d_model)
        B, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Project
        q = self.w_q(q).view(B, seq_len_q, self.n_heads, self.d_k)
        k = self.w_k(k).view(B, seq_len_k, self.n_heads, self.d_k)
        v = self.w_v(v).view(B, seq_len_k, self.n_heads, self.d_k)

        # Apply kernel
        q_prime = self._kernel_function(q @ self.omega.T)
        k_prime = self._kernel_function(k @ self.omega.T)

        # Normalize along feature dimension
        q_prime = q_prime / (q_prime.sum(dim=-1, keepdim=True) + 1e-6)
        k_prime = k_prime / (k_prime.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute prefix sums over k dimension
        K_cum = torch.cumsum(k_prime, dim=1)  # (B, seq_len_k, n_heads, r)
        KV_cum = torch.cumsum(k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=1)

        # Determine block boundaries (causal)
        # block i => last index is (i+1)*block_size_w - 1
        q_indices = torch.arange(seq_len_q, device=q.device)
        q_block_indices = q_indices // self.block_size_h
        key_indices = (q_block_indices + 1)*self.block_size_w - 1
        key_indices = key_indices.clamp(max=seq_len_k - 1)

        # Gather from prefix sums
        # K_cum_selected => shape (B, seq_len_q, n_heads, r)
        indices = key_indices.view(1, -1, 1, 1).expand(B, -1, self.n_heads, self.nb_features)
        K_cum_selected = K_cum.gather(dim=1, index=indices)

        # KV_cum_selected => shape (B, seq_len_q, n_heads, r, d_k)
        indices_kv = key_indices.view(1, -1, 1, 1, 1).expand(B, -1, self.n_heads, self.nb_features, self.d_k)
        KV_cum_selected = KV_cum.gather(dim=1, index=indices_kv)

        numerator   = torch.sum(q_prime.unsqueeze(-1) * KV_cum_selected, dim=-2)
        denominator = torch.sum(q_prime * K_cum_selected, dim=-1, keepdim=True)
        out = numerator / (denominator + 1e-6)

        out = out.reshape(B, seq_len_q, self.d_model)
        out = self.w_o(out)
        return out

# class CausalSelfAttention(nn.Module):
#     """
#     A standard GPT-like self-attention with a causal mask 
#     that prevents attending to future tokens.
#     If you prefer Performer, you can adapt that code here 
#     but ensure it's strictly causal.
#     """
#     def __init__(self, d_model: int, n_heads: int, dropout: float=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.n_heads = n_heads
#         self.d_k = d_model // n_heads

#         self.w_q = nn.Linear(d_model, d_model, bias=False)
#         self.w_k = nn.Linear(d_model, d_model, bias=False)
#         self.w_v = nn.Linear(d_model, d_model, bias=False)
#         self.w_o = nn.Linear(d_model, d_model, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: (B, T, d_model)
#         B, T, D = x.shape
#         # Project
#         q = self.w_q(x).view(B, T, self.n_heads, self.d_k)
#         k = self.w_k(x).view(B, T, self.n_heads, self.d_k)
#         v = self.w_v(x).view(B, T, self.n_heads, self.d_k)

#         # (B, T, n_heads, d_k) => (B, n_heads, T, d_k)
#         q = q.permute(0, 2, 1, 3)
#         k = k.permute(0, 2, 1, 3)
#         v = v.permute(0, 2, 1, 3)

#         # scaled dot product
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_heads, T, T)

#         # causal mask => only attend to positions up to i
#         # shape => (T, T)
#         causal_mask = torch.tril(torch.ones(T, T, device=x.device))  # 1 => can attend, 0 => cannot
#         scores = scores.masked_fill(causal_mask==0, float('-inf'))

#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#         out = torch.matmul(attn, v)  # (B, n_heads, T, d_k)

#         # reorder => (B, T, n_heads, d_k)
#         out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
#         out = self.w_o(out)
#         return out

##############################################################################
# 8. DecoderBlock (Self-Attn + FeedForward)
##############################################################################
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int,
                 self_attention_block: CausalPerformer,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block

        self.residual_attn = ResidualConnection(d_model, dropout)
        self.residual_ff   = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor):
        # 1) Self-attn
        # x = self.residual_attn(x, self.self_attention_block)
        x = self.residual_attn(x, lambda x_norm:
            self.self_attention_block(x_norm, x_norm, x_norm))
        # 2) Feed-forward
        x = self.residual_ff(x, self.feed_forward_block)
        return x

##############################################################################
# 9. Decoder (stack of blocks)
##############################################################################
class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

##############################################################################
# 10. ProjectionLayer
##############################################################################
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        return self.proj(x)

##############################################################################
# 11. DecoderOnlyTransformer (GPT-style)
##############################################################################
class Transformer(nn.Module):
    """
    A single-stack (decoder-only) Transformer that
    takes a single sequence of tokens and does next-token prediction.
    """
    def __init__(self, 
                 vocab_size_tgt: int, 
                 vocab_size_src: int,
                 # tgt_seq_len: int,
                 # lto_seq_len: int,
                 max_seq_len: int,
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int,
                 d_ff: int, 
                 dropout: float, 
                 feature_tensor: torch.Tensor,
                 special_token_ids: torch.Tensor,
                 kernel_type="exp"):
        super().__init__()

        # Embedding
        # self.token_emb = InputEmbeddings(d_model, vocab_size)
        # self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # self.token_embed = SpecialPlusFeatureLookup(
        #     d_model = d_model,
        #     feature_tensor = feature_tensor,
        #     special_token_ids = special_token_ids,
        #     hidden_dim = d_ff # using d_ff as MLP hidden size
        # )

        # lto_embed = SpecialPlusFeatureLookup(
        #     d_model = d_model,
        #     feature_tensor = feature_tensor,
        #     special_token_ids = special_token_ids,
        #     hidden_dim = d_ff
        # )

        # tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        
        # src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        # tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        # lto_pos = PositionalEncoding(d_model, lto_seq_len, dropout)

        self.token_embed = SpecialPlusFeatureLookup(
                d_model        = d_model,
                feature_tensor = feature_tensor,           # (59, 34)
                product_ids    = list(range(13, 57), 59),      # 13 … 56
                vocab_size_src     = vocab_size_src                # 59
        )

        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # Build N decoder blocks
        blocks = []
        for _ in range(n_layers):
            performer = CausalPerformer(d_model, n_heads, dropout)
            ff_block  = FeedForwardBlock(d_model, d_ff, dropout)
            blk = DecoderBlock(d_model, performer, ff_block, dropout)
            blocks.append(blk)
        self.decoder = Decoder(d_model, nn.ModuleList(blocks))
        # Final projection to vocab
        self.projection = ProjectionLayer(d_model, vocab_size_tgt)

        # (Optional) param init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        input_seq: (B, seq_len) integer tokens
        returns:   (B, seq_len, vocab_size)
        """
        x = self.token_embed(input_seq)
        x = self.pos_enc(x)
        x = self.decoder(x)
        logits = self.projection(x)
        # logits = self.decision_head(x)
        return logits

##############################################################################
# 12. Build function
##############################################################################
def build_transformer(vocab_size_src: int,
                      vocab_size_tgt: int,
                      max_seq_len: int,
                      d_model: int,
                      n_layers: int,
                      n_heads: int,
                      d_ff: int,
                      dropout: float,
                      feature_tensor: torch.Tensor,
                      special_token_ids: torch.Tensor,
                      kernel_type: str="exp"):
    
    transformer = Transformer(
        vocab_size_src   = vocab_size_src,
        vocab_size_tgt   = vocab_size_tgt,
        max_seq_len  = max_seq_len,
        d_model      = d_model,
        n_layers     = n_layers,
        n_heads      = n_heads,
        d_ff         = d_ff,
        dropout      = dropout,
        feature_tensor = feature_tensor,
        special_token_ids = special_token_ids,
        kernel_type  = kernel_type
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
