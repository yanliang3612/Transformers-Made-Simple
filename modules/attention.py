"""Attention layers.

The goal here is clarity over cleverness: shapes are named, masks are boolean,
and the same MultiHeadAttention works for self-attention and cross-attention.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .positional_encoding import apply_rotary_embedding


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Args:
        dim: model dimension
        num_heads: number of attention heads
        dropout: dropout applied to attention weights and output projection
        use_rope: apply rotary position embeddings to q and k
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x = x.view(batch, seq_len, self.num_heads, dim // self.num_heads)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention.

        query: (batch, query_len, dim)
        key/value: (batch, key_len, dim). If omitted, use query for self-attention.
        mask: broadcastable to (batch, heads, query_len, key_len)
        """

        key = query if key is None else key
        value = key if value is None else value

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        if self.use_rope:
            q, k = apply_rotary_embedding(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        return self.out_proj(self._merge_heads(context))


class LinearAttention(nn.Module):
    """A compact efficient-attention example.

    This uses a positive feature map so attention can be computed in roughly
    linear time with respect to sequence length. It is meant for learning, not
    as a production replacement for optimized kernels.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        return x.view(batch, seq_len, self.num_heads, dim // self.num_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = torch.nn.functional.elu(self._split_heads(self.q_proj(x))) + 1
        k = torch.nn.functional.elu(self._split_heads(self.k_proj(x))) + 1
        v = self._split_heads(self.v_proj(x))

        if mask is not None:
            # Padding mask shape can be (batch, 1, 1, seq_len).
            k = k * mask.transpose(-1, -2)
            v = v * mask.transpose(-1, -2)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        normalizer = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, normalizer)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out_proj(self.dropout(out))

