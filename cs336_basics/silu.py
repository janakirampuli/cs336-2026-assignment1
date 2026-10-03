import torch

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)