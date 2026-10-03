import torch
from torch import nn
from torch import einsum

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        output = einsum('... d, d -> ... d', x_normed, self.weight)
        return output.to(in_dtype)