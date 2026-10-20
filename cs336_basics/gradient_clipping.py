from collections.abc import Iterable

import torch

def gradient_clipping(
        params: Iterable[torch.nn.Parameter],
        max_l2_norm: float
) -> torch.Tensor:
    # returns the total gradient norm before clipping, for logging
    params_with_grad = [p for p in params if p.grad is not None]

    if not params_with_grad:
        return torch.tensor(0.0)

    total_norm = torch.sqrt(sum(p.grad.norm(2)**2 for p in params_with_grad))

    scale = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)
    for p in params_with_grad:
        p.grad.mul_(scale)

    return total_norm
