import torch
from torch import einsum
import math
from .softmax import softmax

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None=None
) -> torch.Tensor:
    d_k = K.size(-1)
    scores = einsum('... q d, ... k d -> ... q k', Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)

    attn_weights = softmax(scores, -1)
    return einsum('... q k, ... k v -> ... q v', attn_weights, V)