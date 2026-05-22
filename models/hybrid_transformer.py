"""Hybrid Transformer mixing convolution and attention."""

from __future__ import annotations

import torch
from torch import nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbedding
from modules.feedforward import FeedForward
from modules.positional_encoding import LearnedPositionalEncoding


class ConvMixerBlock(nn.Module):
    """Local convolution followed by global self-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        local = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(local)
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class HybridTransformer(nn.Module):
    """A text classifier combining CNN locality and Transformer context."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        max_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, dim)
        self.position = LearnedPositionalEncoding(max_len, dim, dropout)
        self.layers = nn.ModuleList(
            [ConvMixerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.position(self.embedding(token_ids))
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.norm(x[:, 0]))

