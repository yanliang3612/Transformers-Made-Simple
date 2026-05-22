"""Retrieval-augmented Transformer example."""

from __future__ import annotations

import torch
from torch import nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbedding
from modules.feedforward import FeedForward
from modules.masks import causal_mask
from modules.positional_encoding import LearnedPositionalEncoding


class RetrievalDecoderBlock(nn.Module):
    """Decoder block that can cross-attend to retrieved document embeddings."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.retrieval_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, retrieved: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.retrieval_attn(self.norm2(x), retrieved, retrieved))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class RetrievalTransformer(nn.Module):
    """A language model that conditions on retrieved context vectors."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, dim)
        self.position = LearnedPositionalEncoding(max_len, dim, dropout)
        self.retrieved_projection = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [RetrievalDecoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, retrieved_vectors: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(token_ids.size(1), token_ids.device)
        x = self.position(self.embedding(token_ids))
        retrieved = self.retrieved_projection(retrieved_vectors)
        for layer in self.layers:
            x = layer(x, retrieved, mask)
        return self.lm_head(self.norm(x))

