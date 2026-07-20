"""Tiny checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))

