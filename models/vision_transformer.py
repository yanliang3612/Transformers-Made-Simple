"""Vision Transformer (ViT) for image classification."""

from __future__ import annotations

import torch
from torch import nn

from modules.embeddings import PatchEmbedding
from models.encoder_only import EncoderBlock


class VisionTransformer(nn.Module):
    """Treat an image as a sequence of patch tokens plus one class token."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.size(0)
        patches = self.patch_embedding(images)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = self.dropout(x + self.position[:, : x.size(1)])

        for layer in self.layers:
            x = layer(x)

        return self.head(self.norm(x[:, 0]))

