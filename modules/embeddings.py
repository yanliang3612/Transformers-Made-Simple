"""Embedding layers for text and images."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Token embedding with the common sqrt(dim) scaling.

    The scaling keeps embedding magnitudes similar to positional encodings.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = dim**0.5

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        return self.embedding(token_ids) * self.scale


class PatchEmbedding(nn.Module):
    """Turn an image into a sequence of patch embeddings.

    Vision Transformer treats image patches like tokens. A Conv2d with
    kernel_size=stride=patch_size is a compact way to extract non-overlapping
    patches and project each patch to model dimension.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 256,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (batch, channels, height, width)
        x = self.proj(images)
        # x: (batch, dim, grid_h, grid_w) -> (batch, num_patches, dim)
        return x.flatten(2).transpose(1, 2)

