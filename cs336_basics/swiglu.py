import torch
import torch.nn as nn
from typing import Optional
from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: Optional[torch.device],
            dtype: Optional[torch.dtype]
    ):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = int(d_model * 8/3)
            d_ff = (d_ff + 64-1)//64 * 64

        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        # SiLU(x) = x * sigmoid(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        return self.w2(silu_w1_x * w3_x)