"""Position information for Transformers.

Self-attention sees a set of vectors, so we add or learn positional information
to tell the model where each token or patch lives.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic fixed positional encoding from "Attention Is All You Need"."""

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned position embeddings, common in BERT, GPT, and ViT."""

    def __init__(self, max_len: int, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.position = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).expand(batch, seq_len)
        return self.dropout(x + self.position(positions))


def apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal RoPE implementation for educational decoder models.

    Args:
        q, k: tensors with shape (batch, heads, seq_len, head_dim)
    """

    head_dim = q.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")

    device = q.device
    seq_len = q.size(-2)
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, theta)
    sin = freqs.sin()[None, None, :, :]
    cos = freqs.cos()[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2)

    return rotate(q), rotate(k)

