"""Mixture-of-Experts Transformer example."""

from __future__ import annotations

import torch
from torch import nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbedding
from modules.feedforward import MoEFeedForward
from modules.positional_encoding import LearnedPositionalEncoding


class MoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        num_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoEFeedForward(dim, mlp_dim, num_experts, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        moe_out, router_logits = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        return x, router_logits


class MoETransformer(nn.Module):
    """Encoder-style Transformer where MLP blocks are replaced by experts."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        num_experts: int = 4,
        max_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, dim)
        self.position = LearnedPositionalEncoding(max_len, dim, dropout)
        self.layers = nn.ModuleList(
            [MoEBlock(dim, num_heads, mlp_dim, num_experts, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.position(self.embedding(token_ids))
        router_logits = []
        for layer in self.layers:
            x, logits = layer(x, mask)
            router_logits.append(logits)
        return {"logits": self.head(self.norm(x[:, 0])), "router_logits": torch.stack(router_logits)}

