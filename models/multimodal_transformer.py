"""Simple multimodal Transformer for text + image inputs."""

from __future__ import annotations

import torch
from torch import nn

from modules.embeddings import PatchEmbedding, TokenEmbedding
from modules.positional_encoding import LearnedPositionalEncoding
from models.encoder_only import EncoderBlock


class MultimodalTransformer(nn.Module):
    """Fuse text tokens and image patch tokens in one encoder.

    This is the easiest multimodal pattern to understand: map every modality to
    the same dimension, concatenate tokens, and let self-attention mix them.
    """

    def __init__(
        self,
        vocab_size: int,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        num_classes: int = 2,
        max_text_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_embedding = TokenEmbedding(vocab_size, dim)
        self.image_embedding = PatchEmbedding(image_size, patch_size, 3, dim)
        total_len = 1 + max_text_len + self.image_embedding.num_patches
        self.position = LearnedPositionalEncoding(total_len, dim, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embedding = nn.Embedding(3, dim)
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, token_ids: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        batch = token_ids.size(0)
        cls = self.cls_token.expand(batch, -1, -1)
        text = self.text_embedding(token_ids)
        image = self.image_embedding(images)

        x = torch.cat([cls, text, image], dim=1)
        modality_ids = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.long, device=x.device),
                torch.ones(batch, text.size(1), dtype=torch.long, device=x.device),
                torch.full((batch, image.size(1)), 2, dtype=torch.long, device=x.device),
            ],
            dim=1,
        )
        x = self.position(x + self.modality_embedding(modality_ids))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, 0]))

