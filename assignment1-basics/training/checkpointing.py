import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    wandb_run_id: str | None = None,
) -> None:
    """Save checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps': iteration,
    }
    if wandb_run_id:
        state['wandb_run_id'] = wandb_run_id
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[str, int]:
    """Load checkpoint from source.
    
    Initialize objects with weights.
    """
    state = torch.load(src, weights_only=False)
    model.load_state_dict(state['model'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    wandb_run_id = state.get('wandb_run_id', None)
    return wandb_run_id, state['steps']


def load_latest_checkpoint(
    folder: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[str, int]:
    """Load latest checkpoint from folder.
    
    Latest is defined by the max step count.
    """
    steps = [int(f.split('.pt')[0]) for f in os.listdir(folder)]
    max_step = max(steps)
    return load_checkpoint(
        src=f'{folder}/{max_step}.pt',
        model=model,
        optimizer=optimizer,
    )