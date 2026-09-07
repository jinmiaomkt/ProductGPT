import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate GeLU (Tanh approximation).
    Formula:
       0.5 * x * (1 + tanh( sqrt(2/pi) * ( x + 0.044715*x^3 ) ))
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))

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
        return self.linear_2(self.dropout(gelu_approx(self.linear_1(x))))

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
        
class CausalPerformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, kernel_type: str = "exp",
                 block_size_h: int = 1, block_size_w: int = 1):
        """
        Blockwise causal Performer using cumulative summation.

        - block_size_h: Controls grouping along `seq_len_q`
        - block_size_w: Controls grouping along `seq_len_k`
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by number of heads"
        self.d_k = d_model // n_heads

        self.nb_features = max(1, math.ceil(math.log(d_model)))
        self.kernel_type = kernel_type

        # Blockwise settings
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
        self.omega = nn.Parameter(omega, requires_grad=False)

    def _kernel_function(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "gelu":
            return gelu_approx(x) + 1e-6
        elif self.kernel_type == "exp":
            return torch.exp(-0.5 * (x ** 2))
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Blockwise causal Performer using prefix sums.
        Supports the case where:
        - q: (B, seq_len_q, d_model)
        - k: (B, seq_len_k, d_model)
        - v: (B, seq_len_k, d_model)
        
        The two block sizes (block_size_h, block_size_w) are provided as parameters.
        
        For each query token at index i, we compute:
            q_block = floor(i / block_size_h)
            key_index = clamp( (q_block + 1)*block_size_w - 1, max=seq_len_k - 1 )
        Then we use cumulative sums on k_prime and k_prime*v, and gather the values at these key indices.
        """
        B, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Linear projections into (B, *, n_heads, d_k)
        q = self.w_q(q).view(B, seq_len_q, self.n_heads, self.d_k)
        k = self.w_k(k).view(B, seq_len_k, self.n_heads, self.d_k)
        v = self.w_v(v).view(B, seq_len_k, self.n_heads, self.d_k)

        # Compute kernel feature maps:
        # q_prime: (B, seq_len_q, n_heads, r)
        # k_prime: (B, seq_len_k, n_heads, r)
        q_prime = self._kernel_function(q @ self.omega.T)
        k_prime = self._kernel_function(k @ self.omega.T)

        # Normalize along the feature (r) dimension
        q_prime = q_prime / (q_prime.sum(dim=-1, keepdim=True) + 1e-6)
        k_prime = k_prime / (k_prime.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute cumulative sums over the key dimension.
        # K_cum: (B, seq_len_k, n_heads, r)
        K_cum = torch.cumsum(k_prime, dim=1)
        # For KV, we need to combine k_prime and v.
        # v: (B, seq_len_k, n_heads, d_k) => unsqueeze(-2): (B, seq_len_k, n_heads, 1, d_k)
        # k_prime: (B, seq_len_k, n_heads, r) => unsqueeze(-1): (B, seq_len_k, n_heads, r, 1)
        # Their product: (B, seq_len_k, n_heads, r, d_k)
        KV_cum = torch.cumsum(k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=1)

        # Determine block boundaries:
        # For queries, we have indices 0,...,seq_len_q-1.
        # Compute query block indices (block_height) and then corresponding key index:
        q_indices = torch.arange(seq_len_q, device=q.device)  # (seq_len_q,)
        q_block_indices = q_indices // self.block_size_h   # (seq_len_q,)
        # For each query token, allowed key index is: (q_block + 1)*block_size_w - 1
        key_indices = (q_block_indices + 1) * self.block_size_w - 1
        key_indices = key_indices.clamp(max=seq_len_k - 1)  # (seq_len_q,)

        # Expand indices to match K_cum dimensions.
        # K_cum shape: (B, seq_len_k, n_heads, r)
        # We need an index tensor of shape: (B, seq_len_q, n_heads, r)
        indices = key_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, seq_len_q, 1, 1)
        indices = indices.expand(B, seq_len_q, self.n_heads, self.nb_features)  # (B, seq_len_q, n_heads, r)

        # Gather cumulative sums at these key boundaries:
        K_cum_selected = K_cum.gather(dim=1, index=indices)  # (B, seq_len_q, n_heads, r)

        # For KV_cum, its shape is (B, seq_len_k, n_heads, r, d_k).
        # We need an index tensor of shape: (B, seq_len_q, n_heads, r, 1)
        indices_kv = key_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (1, seq_len_q, 1, 1, 1)
        indices_kv = indices_kv.expand(B, seq_len_q, self.n_heads, self.nb_features, self.d_k)  # (B, seq_len_q, n_heads, r, d_k)
        KV_cum_selected = KV_cum.gather(dim=1, index=indices_kv)  # (B, seq_len_q, n_heads, r, d_k)

        # Compute output for each query token:
        # For each query, we have q_prime: (B, seq_len_q, n_heads, r)
        # Dot with KV_cum_selected along dimension r:
        numerator = torch.sum(q_prime.unsqueeze(-1) * KV_cum_selected, dim=-2)  # (B, seq_len_q, n_heads, d_k)
        denominator = torch.sum(q_prime * K_cum_selected, dim=-1, keepdim=True)   # (B, seq_len_q, n_heads, 1)

        out = numerator / (denominator + 1e-6)  # (B, seq_len_q, n_heads, d_k)

        # Merge heads and apply final projection:
        out = out.reshape(B, seq_len_q, self.n_heads * self.d_k)
        return self.w_o(out)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: CausalPerformer, cross_attention_block: CausalPerformer, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, limited_time_offer):
        x = self.residual_connections[0](x, lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm))
        limited_time_offer = self.residual_connections[1](limited_time_offer, lambda lto_norm: self.cross_attention_block(lto_norm, x, x))
        limited_time_offer = self.residual_connections[2](limited_time_offer, self.feed_forward_block)
        return limited_time_offer

        
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, limited_time_offer):
        for layer in self.layers:
            limited_time_offer = layer(x, limited_time_offer)
        return self.norm(limited_time_offer)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: CausalPerformer, cross_attention_block: CausalPerformer, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output):
        x = self.residual_connections[0](x, lambda x_norm: self.self_attention_block(x_norm, x_norm, x_norm))
        x = self.residual_connections[1](x, lambda x_norm: self.cross_attention_block(x_norm, encoder_output, encoder_output))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
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

    def encode(self, src: torch.Tensor, lto: torch.Tensor):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        lto = self.lto_embed(lto)
        lto = self.lto_pos(lto)
        return self.encoder(src, lto)
    
    def decode(self, encoder_output: torch.Tensor, tgt: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int,
                      lto_vocab_size: int,
                      src_seq_len: int, 
                      tgt_seq_len: int,
                      lto_seq_len: int,
                      d_model: int, 
                      N: int, 
                      h: int,
                      dropout: float, 
                      kernel_type: str,
                      d_ff: int,
                      layer_dropout_rate: float = 0.1) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    lto_embed = InputEmbeddings(d_model, lto_vocab_size)
    
    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    lto_pos = PositionalEncoding(d_model, lto_seq_len, dropout)
    
    # Create encoder blocks and wrap each with stochastic depth
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = CausalPerformer(d_model, h, dropout, kernel_type, block_size_h=10, block_size_w=10)  
        ## Remmeber to Change NUMBER OF LIMITED TIME OFFER here
        encoder_cross_attention_block = CausalPerformer(d_model, h, dropout, kernel_type, block_size_h=12, block_size_w=10)  
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_cross_attention_block, feed_forward_block, dropout)
        # Wrap the block for layer dropout
        encoder_block = StochasticLayerWrapper(encoder_block, dropout_rate=layer_dropout_rate)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks and wrap each with stochastic depth
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = CausalPerformer(d_model, h, dropout, kernel_type, block_size_h=1, block_size_w=1)
        ## Remmeber to Change NUMBER OF LIMITED TIME OFFER here
        decoder_cross_attention_block = CausalPerformer(d_model, h, dropout, kernel_type, block_size_h=1, block_size_w=12)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        # Wrap the block for layer dropout
        decoder_block = StochasticLayerWrapper(decoder_block, dropout_rate=layer_dropout_rate)
        decoder_blocks.append(decoder_block)

    # Build encoder, decoder, and projection layer
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, lto_embed, src_pos, tgt_pos, lto_pos, projection_layer)

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer