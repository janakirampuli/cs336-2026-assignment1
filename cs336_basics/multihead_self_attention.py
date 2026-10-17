import torch
import torch.nn as nn
import einops
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention

class CausalSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            device: torch.device | None=None,
            dtype: torch.dtype | None=None,
            theta: float = 10000.0,
            max_seq_len: int = 4096,
            use_rope: bool = True,
            causal: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3*d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

        self.causal = causal

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None=None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)

        q, k, v = torch.chunk(qkv, 3, -1)

        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = None
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )
        attn_output = scaled_dot_product_attention(q, k, v, causal_mask)
        attn_output = einops.rearrange(attn_output, 'b h s d -> b s (h d)', h=self.num_heads)
        return self.output_proj(attn_output)