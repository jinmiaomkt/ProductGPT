import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def feature_map(x):
        """
        Apply a kernel transformation to the input (e.g., ReLU or softmax-free kernel).
        This is the key step for linearizing attention.
        """
        return F.relu(x) + 1e-6  # Avoid numerical instability with a small constant

    @staticmethod
    def linear_attention(query, key, value):
        """
        Compute linear attention using kernel-based feature maps.
        """
        # Apply feature map transformations to query and key
        query = MultiHeadAttentionBlock.feature_map(query)
        key = MultiHeadAttentionBlock.feature_map(key)

        # Compute key-value product and normalize
        key_value = torch.einsum("bhnd,bhne->bhde", key, value)  # Key-Value matrix (batch, h, d_k, d_k)
        query_key = torch.einsum("bhnd,bhde->bhne", query, key_value)  # Query-Key aggregation

        # Normalize output
        # Add an extra dimension to key.sum(dim=2)
        normalizer = 1 / (torch.einsum("bhnd,bhne->bhn", query, key.sum(dim=2, keepdim=True)) + 1e-6)
        output = query_key * normalizer.unsqueeze(-1)  # Element-wise scaling

        return output

    def forward(self, q, k, v, mask=None):
        # Linear projections
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)

        # Reshape into multiple heads
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)

        # Compute linear attention
        x = self.linear_attention(query, key, value)  # (batch, h, seq_len, d_k)

        # Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)  # (batch, seq_len, d_model)
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, limited_time_offer, src_src_self_mask, lto_src_cross_mask):
        x = self.residual_connections[0](x, lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm, src_src_self_mask))
        limited_time_offer = self.residual_connections[1](limited_time_offer, lambda lto_norm: 
                                                          self.cross_attention_block(lto_norm, x, x, lto_src_cross_mask))
        limited_time_offer = self.residual_connections[2](limited_time_offer, self.feed_forward_block)
        return limited_time_offer

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, limited_time_offer, src_src_self_mask, lto_src_cross_mask):
        for layer in self.layers:
            limited_time_offer = layer(x, limited_time_offer, src_src_self_mask, lto_src_cross_mask)
        return self.norm(limited_time_offer)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, tgt_tgt_self_mask, tgt_lto_cross_mask):
        x = self.residual_connections[0](x, lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm, tgt_tgt_self_mask))
        x = self.residual_connections[1](x, lambda x_norm: self.cross_attention_block(x_norm, encoder_output, encoder_output, tgt_lto_cross_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, tgt_tgt_self_mask, tgt_lto_cross_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_tgt_self_mask, tgt_lto_cross_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, lto_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, lto_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.lto_embed = lto_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.lto_pos = lto_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_src_self_mask: torch.Tensor, lto: torch.Tensor, lto_src_cross_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        lto = self.lto_embed(lto)
        lto = self.lto_pos(lto)
        return self.encoder(src, lto, src_src_self_mask, lto_src_cross_mask)
    
    def decode(self, encoder_output: torch.Tensor, tgt_tgt_self_mask: torch.Tensor, tgt: torch.Tensor, tgt_lto_cross_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, tgt_tgt_self_mask, tgt_lto_cross_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, lto_vocab_size: int, src_seq_len: int, tgt_seq_len: int, lto_seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    lto_embed = InputEmbeddings(d_model, lto_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    lto_pos = PositionalEncoding(d_model, lto_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_cross_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, lto_embed, src_pos, tgt_pos, lto_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer