import os
from typing import IO, BinaryIO

import torch

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    # torch.compile wraps the model; save the inner module so keys have no '_orig_mod.' prefix
    unwrapped_model = getattr(model, '_orig_mod', model)
    checkpoint_state = {
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint_state, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None
) -> int:
    # load onto cpu so a checkpoint trained on cuda opens on a machine without it;
    # load_state_dict then copies tensors to wherever the model lives
    checkpoint_state = torch.load(src, map_location='cpu')

    # strip the prefix from older checkpoints saved straight from a compiled model
    state_dict = checkpoint_state['model_state_dict']
    unwanted_prefix = '_orig_mod.'

    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    unwrapped_model = getattr(model, '_orig_mod', model)
    unwrapped_model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    return checkpoint_state['iteration']
