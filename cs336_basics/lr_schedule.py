import math

def get_lr_cosine_schedule(
        it: int,
        max_lr: float,
        min_lr: float,
        warmup_iters: int,
        cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_lr
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_lr + 0.5 * (1 + math.cos(progress * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr
