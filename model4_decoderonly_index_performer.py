import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, 
                 d_model: int, 
                 n_heads: int,
                 dropout: float=0.1,
                 kernel_type: str="exp",
                 nb_features: int=16,
                 block_size_h: int=1,
                 block_size_w: int=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        self.nb_features = nb_features
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
        q_indices = torch.arange(seq_len_q, device=q.device) # (0, 1, 2, ..., 1023)
        q_block_indices = q_indices // self.block_size_h # (0, 1, 2, ..., 1023) // 24 --> (0, 0, 0, ..., 41)
        key_indices = (q_block_indices + 1)*self.block_size_w - 1
        key_indices = key_indices.clamp(max=seq_len_k - 1)
        # Clamping is a method in which we limit a number in a range or in between two given numbers.

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

# -------------------------------------------------------------------
# Simple GPT-style causal self-attention (no Performer)
# -------------------------------------------------------------------
# class CausalSelfAttention(nn.Module):
#     def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.n_heads = n_heads
#         self.d_k     = d_model // n_heads

#         self.w_q = nn.Linear(d_model, d_model, bias=False)
#         self.w_k = nn.Linear(d_model, d_model, bias=False)
#         self.w_v = nn.Linear(d_model, d_model, bias=False)
#         self.w_o = nn.Linear(d_model, d_model, bias=False)
#         self.dropout = nn.Dropout(dropout)

#     # def forward(self, x):
#     def forward(self, x, k=None, v=None):
#         # x : (B, T, d_model)
#         B, T, _ = x.size()
#         q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,T,d_k)
#         k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
#         v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

#         scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)            # (B,h,T,T)
#         mask   = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
#         scores = scores.masked_fill(~mask, float('-inf'))

#         attn = self.dropout(torch.softmax(scores, dim=-1))
#         out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)       # (B,T,d_model)
#         return self.w_o(out)

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
                 vocab_size: int, 
                 max_seq_len: int,
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int,
                 d_ff: int, 
                 dropout: float, 
                 nb_features: int,
                 kernel_type="exp"):
        super().__init__()
        # Embedding
        self.token_emb = InputEmbeddings(d_model, vocab_size)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # Build N decoder blocks
        blocks = []
        for _ in range(n_layers):
            performer = CausalPerformer(d_model = d_model, 
                                        n_heads = n_heads, 
                                        dropout = dropout, 
                                        kernel_type = kernel_type,
                                        nb_features = nb_features)
            ff_block  = FeedForwardBlock(d_model, d_ff, dropout)
            blk = DecoderBlock(d_model, performer, ff_block, dropout)
            blocks.append(blk)
        self.decoder = Decoder(d_model, nn.ModuleList(blocks))
        # Final projection to vocab
        self.projection = ProjectionLayer(d_model, vocab_size)

        # (Optional) param init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        input_seq: (B, seq_len) integer tokens
        returns:   (B, seq_len, vocab_size)
        """
        x = self.token_emb(input_seq)
        x = self.pos_enc(x)
        x = self.decoder(x)
        logits = self.projection(x)
        return logits

##############################################################################
# 12. Build function
##############################################################################
def build_transformer(vocab_size: int,
                      max_seq_len: int,
                      d_model: int,
                      n_layers: int,
                      n_heads: int,
                      d_ff: int,
                      dropout: float,
                      nb_features: int,
                      kernel_type: str="exp"):
    
    transformer = Transformer(
        vocab_size   = vocab_size,
        max_seq_len  = max_seq_len,
        d_model      = d_model,
        n_layers     = n_layers,
        n_heads      = n_heads,
        d_ff         = d_ff,
        dropout      = dropout,
        kernel_type  = kernel_type,
        nb_features  = nb_features,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
