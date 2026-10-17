import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                alpha_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)
                p.add_(p, alpha=-lr * weight_decay)
        return loss