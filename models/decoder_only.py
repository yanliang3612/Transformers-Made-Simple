"""Decoder-only Transformer, similar in spirit to GPT."""

from __future__ import annotations

import torch
from torch import nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbedding
from modules.feedforward import GatedFeedForward
from modules.masks import causal_mask, combine_masks
from modules.positional_encoding import LearnedPositionalEncoding


class DecoderBlock(nn.Module):
    """Causal self-attention block for autoregressive language modeling."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, use_rope=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Small GPT-like language model."""

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
        self.token_embedding = TokenEmbedding(vocab_size, dim)
        self.position = LearnedPositionalEncoding(max_len, dim, dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = token_ids.size(1)
        mask = combine_masks(causal_mask(seq_len, token_ids.device), padding_mask)

        x = self.position(self.token_embedding(token_ids))
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))

