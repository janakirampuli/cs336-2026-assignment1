import torch
import torch.nn as nn
from torch import einsum
from typing import Optional
import math

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_params()

    def reset_params(self):
        std = math.sqrt(2.0/(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, 0.0, std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum('... i, o i -> ... o', x, self.W)