import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torch.nn as nn
from transformers import BigBirdConfig
from transformers.models.big_bird.modeling_big_bird import BigBirdSelfAttention

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
# 7. Local Causal Self-Attention (BigBird)
##############################################################################

class LocalCausalSelfAttention(nn.Module):
    """
    BigBird block-sparse causal self-attention wrapped in the same
    signature as your hand-rolled layer.

    Parameters
    ----------
    d_model        : hidden size (must be divisible by `n_heads`)
    n_heads        : number of attention heads
    window_size    : *local* window width (tokens to the **left** only).
                     BigBird uses blocks ⇒  `window_size % block_size == 0`
    dropout        : dropout prob on outputs
    block_size     : size of each sparse block (BigBird default = 64)
    num_rand_blks  : # of random blocks per query block (adds global reach)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.1,
        block_size: int = 1,
        num_rand_blks: int = 0,
    ):
        super().__init__()

        # BigBird’s “window” is expressed in *blocks*, not raw tokens.
        if window_size % block_size != 0:
            raise ValueError(
                f"`window_size` ({window_size}) must be a multiple "
                f"of `block_size` ({block_size})."
            )

        self.block_size = block_size

        # --- tiny throw-away config just for this single layer ----------
        cfg = BigBirdConfig(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            attention_type="block_sparse",
            is_decoder=True,          # ← makes it *causal*
            use_bias=False,
            dropout=dropout,
            attention_probs_dropout_prob=dropout,
            block_size=block_size,
            num_random_blocks=num_rand_blks,
            # the list length == #layers ⇒ supply a singleton
            attention_window=[window_size // block_size],
        )

        self.attn = BigBirdSelfAttention(cfg)
        self.out  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, d_model)
        Returns tensor of the same shape.
        """
        B, L, _ = x.shape

        # BigBird expects:
        #   • input_ids   (unused here)
        #   • attention_mask ∈ {0,1} same shape as input (0 = pad)
        #   • head_mask       (optional)
        # We only need a binary “not-padding” mask.
        mask = torch.ones(B, L, dtype=torch.long, device=x.device)

        # BigBirdSelfAttention returns (context, attn_weights); ignore weights.
        context, _ = self.attn(
            hidden_states=x,
            attention_mask=mask,
            output_attentions=False,
        )
        return self.out(self.drop(context))

##############################################################################
# 8. DecoderBlock (Self-Attn + FeedForward)
##############################################################################
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int,
                 self_attention_block: LocalCausalSelfAttention,
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
                 window_size: int,
                 dropout: float, 
                 kernel_type="exp"):
        super().__init__()
        # Embedding
        self.token_emb = InputEmbeddings(d_model, vocab_size)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        # Build N decoder blocks
        blocks = []
        for _ in range(n_layers):
            bigbird = LocalCausalSelfAttention(d_model, n_heads, window_size)
            ff_block  = FeedForwardBlock(d_model, d_ff, dropout)
            blk = DecoderBlock(d_model, bigbird, ff_block, dropout)
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
                      window_size: int,
                      d_ff: int,
                      dropout: float,
                      kernel_type: str="exp"):
    
    transformer = Transformer(
        vocab_size   = vocab_size,
        max_seq_len  = max_seq_len,
        d_model      = d_model,
        n_layers     = n_layers,
        n_heads      = n_heads,
        window_size  = window_size,
        d_ff         = d_ff,
        dropout      = dropout,
        kernel_type  = kernel_type
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
