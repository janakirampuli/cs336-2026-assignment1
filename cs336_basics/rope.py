import torch
from torch import nn
from torch import einsum
import einops
from typing import Optional
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)

        cos_cache = freqs.cos()
        sin_cache = freqs.sin()

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    ):
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_pairs = einops.rearrange(x, '... (d j) -> ... d j', j=2)
        x_1 = x_pairs[..., 0]
        x_2 = x_pairs[..., 1]

        x_out1 = x_1 * cos - x_2 * sin
        x_out2 = x_1 * sin + x_2 * cos

        x_rotated_pairs = torch.stack([x_out1, x_out2], dim=-1)
        x_rotated = einops.rearrange(x_rotated_pairs, '... d j -> ... (d j)')
        return x_rotated.to(x.dtype)
