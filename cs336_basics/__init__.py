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