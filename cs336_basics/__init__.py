import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from .train_bpe import train_bpe
from .tokenizer import Tokenizer
from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .silu import silu
from .rope import RotaryPositionalEmbedding
from .softmax import softmax
from .scaled_dot_product_attention import scaled_dot_product_attention
from .multihead_self_attention import CausalSelfAttention
from .transformer_block import TransformerBlock
from .transformer_lm import TransformerLM
from .cross_entropy import cross_entropy