"""Encoder-decoder Transformer, useful for translation and summarization."""

from __future__ import annotations

import torch
from torch import nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbedding
from modules.feedforward import FeedForward
from modules.masks import causal_mask, combine_masks
from modules.positional_encoding import SinusoidalPositionalEncoding


class Seq2SeqEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Seq2SeqDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class EncoderDecoderTransformer(nn.Module):
    """A compact Transformer for sequence-to-sequence learning."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, dim)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, dim)
        self.src_position = SinusoidalPositionalEncoding(dim, max_len, dropout)
        self.tgt_position = SinusoidalPositionalEncoding(dim, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [Seq2SeqEncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [Seq2SeqDecoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, tgt_vocab_size)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        memory = self.src_position(self.src_embedding(src_ids))
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
        return memory

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_mask)
        tgt_mask = combine_masks(causal_mask(tgt_ids.size(1), tgt_ids.device), tgt_padding_mask)

        x = self.tgt_position(self.tgt_embedding(tgt_ids))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.lm_head(self.norm(x))

