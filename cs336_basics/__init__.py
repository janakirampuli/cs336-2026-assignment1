import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from .train_bpe import train_bpe