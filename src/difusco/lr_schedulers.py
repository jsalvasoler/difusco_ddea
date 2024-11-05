"""Misc. optimizer implementations."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_schedule_fn(scheduler: str, num_training_steps: int) -> Callable[[torch.optim.Optimizer], LambdaLR]:
    """
    Returns a callable scheduler_fn(optimizer).
    """
    if scheduler == "cosine-decay":
        scheduler_fn = partial(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=num_training_steps,
            eta_min=0.0,
        )
    elif scheduler == "one-cycle":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_one_cycle,
            num_training_steps=num_training_steps,
        )
    else:
        error_msg = f"Invalid scheduler {scheduler} given."
        raise ValueError(error_msg)

    return scheduler_fn


def get_one_cycle(optimizer: torch.optim.Optimizer, num_training_steps: int) -> LambdaLR:
    """
    Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        return float(2 - current_step / (num_training_steps / 2))

    return LambdaLR(optimizer, lr_lambda, -1)
