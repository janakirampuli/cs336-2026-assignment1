import torch
import torch.nn as nn
from torch import einsum
import math

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_params()

    def reset_params(self):
        std = math.sqrt(2.0/(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, 0.0, std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum('... i, o i -> ... o', x, self.weight)