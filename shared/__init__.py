"""
shared/ — model components common to the ProductGPT generations.

These classes were previously copy-pasted across ~10 model modules. They are
reproduced here unchanged; see each submodule's docstring for exactly which
files it was extracted from and the two places where variants were unified.

Typical use inside a model module:

    from shared.layers import FeedForwardBlock, PositionalEncoding
    from shared.attention import CausalPerformer
    from shared.decoder import Decoder, DecoderBlock
    from shared.embeddings import SpecialPlusFeatureLookup
    from shared.vocab import PAD_ID, FEATURE_BEARING_IDS
    from shared.features import FEATURE_COLS, load_feature_tensor

Importing anything here reads no data files and needs no GPU.
"""
from __future__ import annotations

from .layers import (
    gelu_approx,
    LayerNormalization,
    FeedForwardBlock,
    InputEmbeddings,
    PositionalEncoding,
    ResidualConnection,
    ProjectionLayer,
)
from .attention import CausalPerformer
from .decoder import DecoderBlock, Decoder
from .embeddings import SpecialPlusFeatureLookup

__all__ = [
    "gelu_approx",
    "LayerNormalization",
    "FeedForwardBlock",
    "InputEmbeddings",
    "PositionalEncoding",
    "ResidualConnection",
    "ProjectionLayer",
    "CausalPerformer",
    "DecoderBlock",
    "Decoder",
    "SpecialPlusFeatureLookup",
]
