import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        # print(f"Embedding num_embeddings: {self.embedding.num_embeddings}")
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

# --- New: StochasticLayerWrapper for layer dropout ---
class StochasticLayerWrapper(nn.Module):
    def __init__(self, layer: nn.Module, dropout_rate: float):
        """
        Wrap a block (encoder or decoder block) and randomly drop it during training.
        When the layer is not dropped, its residual contribution is scaled by 1/(1 - dropout_rate).
        """
        super().__init__()
        self.layer = layer
        self.dropout_rate = dropout_rate

    def forward(self, x, *args, **kwargs):
        # During evaluation, always run the layer.
        if not self.training or random.random() >= self.dropout_rate:
            out = self.layer(x, *args, **kwargs)
            # Assume the layer returns x + F(x); scale F(x) accordingly:
            return x + (out - x) / (1 - self.dropout_rate)
        else:
            # Skip the layer entirely.
            return x
        
# class PerformerSelfAttention(nn.Module):
#     def __init__(self, d_model, h, dropout, kernel_type="exp"):
#         super().__init__()
#         self.d_model = d_model
#         self.h = h
#         assert d_model % h == 0, "d_model must be divisible by h"
#         self.d_k = d_model // h
#         self.nb_features = math.ceil(math.log(d_model))
#         self.kernel_type = kernel_type
        
#         self.w_q = nn.Linear(d_model, d_model, bias=False)
#         self.w_k = nn.Linear(d_model, d_model, bias=False)
#         self.w_v = nn.Linear(d_model, d_model, bias=False)
#         self.w_o = nn.Linear(d_model, d_model, bias=False)

#         self.dropout = nn.Dropout(dropout)
#         self.create_feature_map()
    
#     def create_feature_map(self):
#         """Create feature map for kernel approximation (Random Fourier Features)."""
#         self.omega = nn.Parameter(torch.randn(self.nb_features, self.d_k) / math.sqrt(self.d_k), requires_grad=False)
    
#     def kernel_function(self, x):
#         """Apply kernel transformation to approximate attention."""
#         if self.kernel_type == "relu":
#             return F.relu(x) + 1e-6  # Avoid numerical instability
#         elif self.kernel_type == "exp":
#             return torch.exp(-0.5 * (x ** 2))
#         else:
#             raise ValueError("Unsupported kernel type")

#     def forward(self, q, k, v, mask=None):
#         batch_size, seq_len, _ = q.shape
        
#         # Project queries, keys, and values
#         q = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k)
#         x = self.w_k(k)
#         batch_size, actual_seq_len, _ = x.shape
#         k = x .view(batch_size, actual_seq_len, self.h, self.d_k)
#         v = self.w_v(v).view(batch_size, actual_seq_len, self.h, self.d_k)

#         # Compute random feature maps
#         q_prime = self.kernel_function(q @ self.omega.T)  # (batch, seq_len, h, nb_features)
#         k_prime = self.kernel_function(k @ self.omega.T)  # (batch, seq_len, h, nb_features)

#         # Normalize
#         q_prime = q_prime / (q_prime.sum(dim=-2, keepdim=True) + 1e-6)
#         k_prime = k_prime / (k_prime.sum(dim=-2, keepdim=True) + 1e-6)

#         # Approximate attention using kernel trick
#         kv = torch.einsum("bshd,bshm->bhmd", k_prime, v)  # (batch, h, d_k, nb_features)
#         qkv = torch.einsum("bshd,bhmd->bshm", q_prime, kv)  # (batch, seq_len, h, d_k)

#         # Reshape and project output
#         qkv = qkv.contiguous().view(batch_size, seq_len, self.h * self.d_k)
#         return self.w_o(qkv)

# -----------------------
# Performer Self-Attention with Memory
# -----------------------

class PerformerAttention(nn.Module):
    def __init__(self, d_model, h, dropout, kernel_type="exp"):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.nb_features = math.ceil(math.log(d_model))
        self.kernel_type = kernel_type

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # # (batch, h, seq_len_d, seq_len_e)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            # print(mask.shape)
            # print(attention_scores.shape)
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len_d, seq_len_e) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len_d, seq_len_e) --> (batch, h, seq_len_d, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask = None, memory_k = None, memory_v = None):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)    # (batch, h, seq_len_d, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)            # (batch, h, seq_len_e, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)    # (batch, h, seq_len_e, d_k)

        if memory_k is not None and memory_v is not None:
            mem_len = memory_k.size(1)
            mem_k = self.w_k(memory_k).view(key.shape[0], mem_len, self.h, self.d_k)
            mem_v = self.w_v(memory_v).view(value.shape[0], mem_len, self.h, self.d_k)
            key = torch.cat([mem_k, key], dim=1) # shape: (batch_size, mem_len + seq_len, self.h, self.d_k)
            value = torch.cat([mem_v, value], dim=1) # shape: (batch_size, mem_len + seq_len, self.h, self.d_k)

        # Calculate attention
        x, self.attention_scores = PerformerAttention.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


# class EncoderBlock(nn.Module):

#     def __init__(self, features: int, self_attention_block: PerformerSelfAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         # self.cross_attention_block = cross_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

#     def forward(self, x, long_long_self_mask):
#         x = self.residual_connections[0](x, lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm, long_long_self_mask))
#         # limited_time_offer = self.residual_connections[1](limited_time_offer, lambda lto_norm:  self.cross_attention_block(lto_norm, x, x, short_long_cross_mask))
#         x = self.residual_connections[1](x, self.feed_forward_block)
#         return x 

class EncoderBlockWithMemory(nn.Module):
    """
    An encoder block that maintains a fixed set of memory tokens.
    """
    def __init__(self, 
                 features: int, 
                 self_attention_block: PerformerAttention,
                 feed_forward_block: FeedForwardBlock, 
                 num_memory_tokens: int, 
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(2)
        ])
        # Initialize memory tokens as learnable parameters.
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, features))
    
    def forward(self, x, mask, prev_memory=None):
        batch_size = x.size(0)
        if prev_memory is None:
            # Expand memory tokens for the current batch.
            memory = self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
        
        # Self-attention with memory: the memory tokens are fed in as extra keys and values.
        x = self.residual_connections[0](x, lambda x_norm: 
            self.self_attention_block(x_norm, x_norm, x_norm, mask, memory_k=memory, memory_v=memory))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        # Update memory tokens (here, a simple update using mean pooling; you could design a more sophisticated update)
        new_memory = x.mean(dim=1, keepdim=True).expand(-1, memory.size(1), -1)
        return x, new_memory

# class Encoder(nn.Module):

#     def __init__(self, features: int, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization(features)

#     def forward(self, x, long_long_self_mask):
#         for layer in self.layers:
#             x = layer(x, long_long_self_mask)
#         return self.norm(x)

class EncoderWithMemory(nn.Module):
    """
    An encoder consisting of a stack of memory-enabled encoder blocks.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, mask, memories=None):
        new_memories = []
        for i, layer in enumerate(self.layers):
            prev_memory = memories[i] if memories is not None else None
            x, new_memory = layer(x, mask, prev_memory)
            new_memories.append(new_memory)
        return self.norm(x), new_memories
    
# class DecoderBlock(nn.Module):

#     def __init__(self, features: int, self_attention_block: PerformerSelfAttention, cross_attention_block: PerformerSelfAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
#         super().__init__()
#         self.self_attention_block = self_attention_block
#         self.cross_attention_block = cross_attention_block
#         self.feed_forward_block = feed_forward_block
#         self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

#     def forward(self, x, encoder_output, short_short_self_mask, short_long_cross_mask):
#         x = self.residual_connections[0](x, lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm, short_short_self_mask))
#         x = self.residual_connections[1](x, lambda x_norm: self.cross_attention_block(x_norm, encoder_output, encoder_output, short_long_cross_mask))
#         x = self.residual_connections[2](x, self.feed_forward_block)
#         return x

class DecoderBlockWithMemory(nn.Module):
    """
    A decoder block with recurrent memory for its self-attention and standard cross-attention.
    """
    def __init__(self, features: int, 
                 self_attention_block: PerformerAttention,
                 cross_attention_block: PerformerAttention, 
                 feed_forward_block: FeedForwardBlock,
                 num_memory_tokens: int, 
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(3)
        ])
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, features))
    
    def forward(self, x, encoder_output, short_self_mask, cross_mask, prev_memory_self = None, prev_memory_cross = None):
        batch_size = x.size(0)
        if prev_memory_self is None:
            memory_self = self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory_self = prev_memory_self
        if prev_memory_cross is None:
            memory_cross = self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory_cross = prev_memory_cross
        
        # Self-attention with recurrent memory tokens.
        x = self.residual_connections[0](x, lambda x_norm: 
            self.self_attention_block(x_norm, x_norm, x_norm, short_self_mask, memory_k = memory_self, memory_v = memory_self))
        new_memory_self = x.mean(dim=1, keepdim=True).expand(-1, memory_self.size(1), -1)

        # Cross-attention: no memory is used here.
        x = self.residual_connections[1](x, lambda x_norm:
            self.cross_attention_block(x_norm, encoder_output, encoder_output, cross_mask, memory_k = memory_cross, memory_v = memory_cross))
        new_memory_cross = x.mean(dim=1, keepdim=True).expand(-1, memory_cross.size(1), -1)

        x = self.residual_connections[2](x, self.feed_forward_block)
        
        # new_memory = x.mean(dim=1, keepdim=True).expand(-1, memory.size(1), -1)
        return x, new_memory_self, new_memory_cross

# class Decoder(nn.Module):

#     def __init__(self, features: int, layers: nn.ModuleList) -> None:
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNormalization(features)

#     def forward(self, x, encoder_output, short_short_self_mask, short_long_cross_mask):
#         for layer in self.layers:
#             x = layer(x, encoder_output, short_short_self_mask, short_long_cross_mask)
#         return self.norm(x)

class DecoderWithMemory(nn.Module):
    """
    A decoder consisting of a stack of memory-enabled decoder blocks.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, short_self_mask, cross_mask, memories_self = None, memories_cross = None):
        new_memories_self = []
        new_memories_cross = []
        for i, layer in enumerate(self.layers):
            prev_memory_self = memories_self[i] if memories_self is not None else None
            prev_memory_cross = memories_cross[i] if memories_cross is not None else None
            x, new_memory_self, new_memory_cross = layer(x, encoder_output, short_self_mask, cross_mask, prev_memory_self, prev_memory_cross)
            new_memories_self.append(new_memory_self)
            new_memories_cross.append(new_memory_cross)
        return self.norm(x), new_memories_self, new_memories_cross

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
# class Transformer(nn.Module):

#     def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         # self.lto_embed = lto_embed
#         self.src_pos = src_pos
#         self.tgt_pos = tgt_pos
#         # self.lto_pos = lto_pos
#         self.projection_layer = projection_layer

#     def encode(self, src: torch.Tensor, long_long_self_mask: torch.Tensor):
#         # (batch, seq_len, d_model)
#         src = self.src_embed(src)
#         src = self.src_pos(src)
#         # lto = self.lto_embed(lto)
#         # lto = self.lto_pos(lto)
#         return self.encoder(src, long_long_self_mask)
    
#     def decode(self, encoder_output: torch.Tensor, short_short_self_mask: torch.Tensor, tgt: torch.Tensor, short_long_cross_mask: torch.Tensor):
#         # (batch, seq_len, d_model)
#         tgt = self.tgt_embed(tgt)
#         tgt = self.tgt_pos(tgt)
#         return self.decoder(tgt, encoder_output, short_short_self_mask, short_long_cross_mask)
    
#     def project(self, x):
#         # (batch, seq_len, vocab_size)
#         return self.projection_layer(x)


class RecurrentMemoryTransformer(nn.Module):
    def __init__(self, encoder: EncoderWithMemory, 
                 decoder: DecoderWithMemory,
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, 
               src: torch.Tensor, 
               mask: torch.Tensor, 
               encoder_memories=None):
        src = self.src_embed(src)
        src = self.src_pos(src)
        enc_output, new_memories = self.encoder(src, mask, encoder_memories)
        return enc_output, new_memories
    
    def decode(self, 
               encoder_output: torch.Tensor, 
               tgt: torch.Tensor,
               short_self_mask: torch.Tensor, 
               cross_mask: torch.Tensor, 
               decoder_memories_self=None,
               decoder_memories_cross=None):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        dec_output, new_memories_self, new_memories_cross = self.decoder(tgt, encoder_output, short_self_mask, cross_mask, decoder_memories_self, decoder_memories_cross)
        return dec_output, new_memories_self, new_memories_cross
    
    def project(self, x):
        return self.projection_layer(x)

# def build_transformer(src_vocab_size: int, 
#                       tgt_vocab_size: int,
#                       src_seq_len: int, 
#                       tgt_seq_len: int,
#                       d_model: int, 
#                       N: int, 
#                       h: int,
#                       dropout: float, 
#                       d_ff: int,
#                       layer_dropout_rate: float = 0.1) -> Transformer:
#     # Create embedding layers
#     src_embed = InputEmbeddings(d_model, src_vocab_size)
#     tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
#     # Create positional encoding layers
#     src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
#     tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
#     # Create encoder blocks and wrap each with stochastic depth
#     encoder_blocks = []
#     for _ in range(N):
#         encoder_self_attention_block = PerformerSelfAttention(d_model, h, dropout)
#         feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
#         # Wrap the block for layer dropout
#         encoder_block = StochasticLayerWrapper(encoder_block, dropout_rate=layer_dropout_rate)
#         encoder_blocks.append(encoder_block)

#     # Create decoder blocks and wrap each with stochastic depth
#     decoder_blocks = []
#     for _ in range(N):
#         decoder_self_attention_block = PerformerSelfAttention(d_model, h, dropout)
#         decoder_cross_attention_block = PerformerSelfAttention(d_model, h, dropout)
#         feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
#         decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
#         # Wrap the block for layer dropout
#         decoder_block = StochasticLayerWrapper(decoder_block, dropout_rate=layer_dropout_rate)
#         decoder_blocks.append(decoder_block)
    
#     # Build encoder, decoder, and projection layer
#     encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
#     decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
#     projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
#     # Create the transformer
#     transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
#     # Initialize parameters
#     for p in transformer.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
    
#     return transformer


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N: int, h: int,dropout: float, d_ff: int, num_memory_tokens_source: int, num_memory_tokens_target: int, layer_dropout_rate: float = 0.1) -> RecurrentMemoryTransformer:
    # Create embedding layers.
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create positional encoding layers.
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Build encoder blocks with memory.
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn = PerformerAttention(d_model, h, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        enc_block = EncoderBlockWithMemory(d_model, encoder_self_attn, ff_block, num_memory_tokens_source, dropout)
        # Wrap with stochastic depth.
        enc_block = StochasticLayerWrapper(enc_block, dropout_rate=layer_dropout_rate)
        encoder_blocks.append(enc_block)
    
    # Build decoder blocks with memory.
    decoder_blocks = []
    for _ in range(N):
        dec_self_attn = PerformerAttention(d_model, h, dropout)
        dec_cross_attn = PerformerAttention(d_model, h, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        dec_block = DecoderBlockWithMemory(d_model, dec_self_attn, dec_cross_attn, ff_block, num_memory_tokens_target, dropout)
        dec_block = StochasticLayerWrapper(dec_block, dropout_rate=layer_dropout_rate)
        decoder_blocks.append(dec_block)
    
    encoder = EncoderWithMemory(d_model, nn.ModuleList(encoder_blocks))
    decoder = DecoderWithMemory(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = RecurrentMemoryTransformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize parameters.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer