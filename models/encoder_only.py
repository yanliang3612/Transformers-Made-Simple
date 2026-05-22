"""Encoder-only Transformer, similar in spirit to BERT."""

from __future__ import annotations

import torch
from torch import nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbedding
from modules.feedforward import FeedForward
from modules.positional_encoding import LearnedPositionalEncoding


class EncoderBlock(nn.Module):
    """Pre-LN encoder block: norm -> attention -> residual -> MLP -> residual."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class EncoderOnlyTransformer(nn.Module):
    """Small BERT-like model for classification or masked language modeling."""

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
        self.token_embedding = TokenEmbedding(vocab_size, dim)
        self.position = LearnedPositionalEncoding(max_len, dim, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.mlm_head = nn.Linear(dim, vocab_size)
        self.classifier = nn.Linear(dim, num_classes)

    def encode(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.position(self.token_embedding(token_ids))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.encode(token_ids, mask)
        cls_vector = hidden[:, 0]
        return {
            "hidden": hidden,
            "mlm_logits": self.mlm_head(hidden),
            "class_logits": self.classifier(cls_vector),
        }

