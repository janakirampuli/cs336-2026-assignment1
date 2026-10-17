import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .multihead_self_attention import CausalSelfAttention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float=10000.0,
            max_seq_len: int=4096,
            device: torch.device | None=None,
            dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype, theta=theta, max_seq_len=max_seq_len, use_rope=True, causal=True)

        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    ) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_output = self.attn(x_norm, token_positions)
        x = x + attn_output
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x