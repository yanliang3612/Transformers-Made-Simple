"""Normalization layers.

LayerNorm is the standard choice in Transformers. RMSNorm is a simpler variant
used by several modern decoder-only language models.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes by the root mean square but does not subtract the mean.
    It is slightly cheaper than LayerNorm and works well in many LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def build_norm(dim: int, norm_type: str = "layernorm") -> nn.Module:
    """Small factory so model files can switch normalization style easily."""

    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")

