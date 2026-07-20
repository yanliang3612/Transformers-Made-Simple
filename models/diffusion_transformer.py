"""Diffusion Transformer (DiT) style denoiser for image patches."""

from __future__ import annotations

import torch
from torch import nn

from modules.embeddings import PatchEmbedding
from modules.feedforward import FeedForward
from models.encoder_only import EncoderBlock


class TimestepEmbedding(nn.Module):
    """Embed diffusion timesteps and inject them into patch tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.net(timesteps.float().view(-1, 1))


class DiffusionTransformer(nn.Module):
    """Predict noise for a noised image, the core DiT training objective."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.position = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches, dim))
        self.time_embedding = TimestepEmbedding(dim)
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.to_patch = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
            nn.Linear(dim, patch_size * patch_size * in_channels),
        )

    def unpatchify(self, patches: torch.Tensor, image_size: int) -> torch.Tensor:
        batch = patches.size(0)
        grid = image_size // self.patch_size
        p = self.patch_size
        c = self.in_channels
        x = patches.view(batch, grid, grid, c, p, p)
        return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch, c, image_size, image_size)

    def forward(self, noised_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        image_size = noised_images.size(-1)
        x = self.patch_embedding(noised_images)
        x = x + self.position[:, : x.size(1)] + self.time_embedding(timesteps).unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        patch_noise = self.to_patch(self.norm(x))
        return self.unpatchify(patch_noise, image_size)

