import torch

def softmax(
        x: torch.Tensor,
        dim: int
) -> torch.Tensor:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_shifted = x - x_max
    exps = torch.exp(x_shifted)
    exps_sum = torch.sum(exps, dim=dim, keepdim=True)
    return exps / exps_sum