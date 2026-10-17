import numpy as np
import torch
from numpy.typing import NDArray

def get_batch(
        dataset: NDArray,
        batch_size: int,
        context_length: int,
        device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = len(dataset)
    if data_len <= context_length:
        raise ValueError(f"dataset length {data_len} must be greater than context_length {context_length}")
    start_indices = torch.randint(low=0, high=data_len - context_length, size=(batch_size, ))

    x_batch = torch.stack([torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in start_indices])
    y_batch = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)) for i in start_indices])

    device = torch.device(device)
    if device.type == 'cuda':
        x_batch = x_batch.pin_memory().to(device, non_blocking=True)
        y_batch = y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

    return x_batch, y_batch
