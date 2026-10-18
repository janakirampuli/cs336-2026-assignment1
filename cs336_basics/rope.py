import torch
from torch import nn
import einops

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, inv_freq)

        cos_cache = angles.cos()
        sin_cache = angles.sin()

        self.register_buffer("cos_cache", cos_cache.to(device), persistent=False)
        self.register_buffer("sin_cache", sin_cache.to(device), persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_pairs = einops.rearrange(x, '... (d j) -> ... d j', j=2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        x_rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1)
        x_rotated = einops.rearrange(x_rotated_pairs, '... d j -> ... (d j)')
        return x_rotated.to(x.dtype)
