import torch

def cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor
) -> torch.Tensor:
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    shifted_logits = logits - max_logits
    log_sum_exp = torch.log(torch.exp(shifted_logits).sum(dim=-1, keepdim=
                                                          True))
    
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))

    loss_i = -target_logits + max_logits + log_sum_exp

    return loss_i.mean()