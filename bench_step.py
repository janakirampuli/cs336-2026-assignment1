import time

import numpy as np
import torch
import torch.nn.functional as F

import cs336_basics.multihead_self_attention as mhsa
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.transformer_lm import TransformerLM

device = torch.device("cuda")
V, T, D, L, H, DFF = 32000, 256, 512, 4, 16, 1344
data = np.memmap("data/owt/train.bin", dtype=np.uint16, mode="r")
own_sdpa = mhsa.scaled_dot_product_attention


def timed(fn, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def get_batch_vectorized(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length + 1)
    window = torch.from_numpy(dataset[idx].astype(np.int64))
    return (window[:, :-1].pin_memory().to(device, non_blocking=True),
            window[:, 1:].pin_memory().to(device, non_blocking=True))


def clip_no_sync(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.sqrt(sum(g.norm(2) ** 2 for g in grads))
    scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for g in grads:
        g.mul_(scale)
    return total_norm


print(f"get_batch (loop): {timed(lambda: get_batch(data, 128, T, device), 50):.2f} ms")
print(f"get_batch (vectorized): {timed(lambda: get_batch_vectorized(data, 128, T, device), 50):.2f} ms")


def bench(name, batch=128, compile_loss=False, fused_attn=False, no_sync_clip=False, vec_batch=False):
    torch._dynamo.reset()
    mhsa.scaled_dot_product_attention = (
        (lambda q, k, v, mask=None: F.scaled_dot_product_attention(q, k, v, is_causal=True)) if fused_attn else own_sdpa
    )
    torch.manual_seed(0)
    model = TransformerLM(V, T, L, D, H, DFF, device=device)
    opt = AdamW(model.parameters(), lr=1e-3)
    clip = clip_no_sync if no_sync_clip else gradient_clipping
    batch_fn = get_batch_vectorized if vec_batch else get_batch

    def loss_fn(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        return cross_entropy(logits.float(), y)

    if compile_loss:
        loss_fn = torch.compile(loss_fn)
    else:
        model = torch.compile(model)

    def step():
        x, y = batch_fn(data, batch, T, device)
        loss = loss_fn(x, y)
        loss.backward()
        clip(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(5):
        step()
    ms = timed(step, 30)
    print(f"{name:>44}: {ms:6.1f} ms/step | {batch * T / ms / 1e3:5.0f}K tok/s | "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    torch.cuda.reset_peak_memory_stats()


bench("current")
bench("+ compile loss with model", compile_loss=True)
bench("+ no-sync clip", compile_loss=True, no_sync_clip=True)
bench("+ vectorized get_batch", compile_loss=True, no_sync_clip=True, vec_batch=True)
bench("+ fused attention", compile_loss=True, no_sync_clip=True, vec_batch=True, fused_attn=True)
bench("all, batch 256", batch=256, compile_loss=True, no_sync_clip=True, vec_batch=True, fused_attn=True)
