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

    if total_norm > max_l2_norm:
        for p in params_with_grad:
            p.grad.mul_(max_l2_norm / (total_norm + 1e-6))

    return total_norm
