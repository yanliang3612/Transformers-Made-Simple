"""Position information for Transformers.

Self-attention sees a set of vectors, so we add or learn positional information
to tell the model where each token or patch lives.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    中文说明：
    这是一个固定的正弦位置编码模块，来自 Transformer 论文
    "Attention Is All You Need"。它不通过训练学习位置向量，而是使用预先
    定义好的 sin/cos 公式为每个位置生成位置编码。偶数维度使用 sin,奇数
    维度使用 cos,并且不同维度对应不同频率，从而帮助模型同时感知局部和
    长距离的位置信息。初始化时会提前计算好 max_len 范围内的位置编码，
    并通过 register_buffer 保存起来；前向传播时，根据当前输入序列长度
    取出对应的位置编码，加到 token embedding 上，再经过 dropout 输出。

    English explanation:
    This is a fixed sinusoidal positional encoding module from the Transformer
    paper "Attention Is All You Need". Instead of learning position embeddings
    during training, it uses predefined sin/cos functions to generate
    positional encodings for each position. Even dimensions use sine functions,
    odd dimensions use cosine functions, and different dimensions use different
    frequencies, allowing the model to capture both local and long-range
    positional information. During initialization, positional encodings up to
    max_len are precomputed and stored using register_buffer. During the forward
    pass, the module selects the positional encodings that match the current
    sequence length, adds them to the token embeddings, and then applies dropout.
    """

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()

        # 中文：定义 dropout 层，用于正则化，减少过拟合。
        # English: Define a dropout layer for regularization to reduce overfitting.
        self.dropout = nn.Dropout(dropout)

        # 中文：生成位置索引 [0, 1, 2, ..., max_len-1]，
        # 并变成形状 [max_len, 1]。
        # English: Create position indices [0, 1, 2, ..., max_len-1],
        # and reshape them to [max_len, 1].
        position = torch.arange(max_len).unsqueeze(1)

        # 中文：计算每一组 sin/cos 维度对应的频率项 1 / 10000^(2i / dim)。
        # English: Compute the frequency terms 1 / 10000^(2i / dim)
        # for each sin/cos dimension pair.
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )

        # 中文：创建一个全零矩阵，用来存储所有位置的位置编码，
        # 形状为 [max_len, dim]。
        # English: Create a zero matrix to store positional encodings
        # for all positions, with shape [max_len, dim].
        pe = torch.zeros(max_len, dim)

        # 中文：偶数维度使用 sin 函数计算位置编码。
        # English: Use the sine function to compute positional encodings
        # for even dimensions.
        pe[:, 0::2] = torch.sin(position * div_term)

        # 中文：奇数维度使用 cos 函数计算位置编码；
        # 这里切片是为了兼容 dim 为奇数的情况。
        # English: Use the cosine function to compute positional encodings
        # for odd dimensions; the slicing handles the case where dim is odd.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # 中文：把 pe 注册为 buffer，而不是可训练参数；
        # unsqueeze(0) 后形状变为 [1, max_len, dim]，方便和 batch 输入相加。
        # English: Register pe as a buffer instead of a trainable parameter;
        # after unsqueeze(0), its shape becomes [1, max_len, dim],
        # making it easy to add to batched inputs.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文：x 的形状通常是 [batch, seq_len, dim]。
        # English: x usually has shape [batch, seq_len, dim].

        # 中文：根据当前序列长度 x.size(1)，取出对应长度的位置编码，
        # 并加到 token embedding 上。
        # English: Select positional encodings according to the current
        # sequence length x.size(1), and add them to the token embeddings.
        x = x + self.pe[:, : x.size(1)]

        # 中文：对加入位置编码后的结果应用 dropout，并返回输出。
        # English: Apply dropout to the result after adding positional
        # encodings, and return the output.
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    中文说明：
    这是一个可学习的位置编码模块。它不像正弦位置编码那样使用固定的 sin/cos 公式，
    而是通过 nn.Embedding(max_len, dim) 创建一个可训练的位置向量表。
    对于序列中的每一个位置 p, 模型都会学习一个对应的位置向量 P_p。
    在前向传播时，先根据当前输入序列长度生成位置编号 [0, 1, ..., seq_len-1],
    然后通过 self.position(positions) 查表得到每个位置的位置向量，
    最后将这些位置向量加到输入 token embedding 上，使模型能够感知 token 的顺序信息。

    English explanation:
    This is a learned positional encoding module. Unlike sinusoidal positional encoding,
    which uses fixed sin/cos functions, this module uses nn.Embedding(max_len, dim)
    to create a trainable position embedding table. For each position p in the sequence,
    the model learns a corresponding position vector P_p. During the forward pass,
    it first creates position indices [0, 1, ..., seq_len-1] according to the current
    sequence length, then uses self.position(positions) to look up the position vectors,
    and finally adds them to the input token embeddings so that the model can capture
    the order information of tokens.
    """

    def __init__(self, max_len: int, dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        # 中文：创建一个可学习的位置 embedding 表，大小为 [max_len, dim]。
        # English: Create a trainable position embedding table with shape [max_len, dim].
        self.position = nn.Embedding(max_len, dim)

        # 中文：Dropout 用于正则化，防止模型过拟合。
        # English: Dropout is used for regularization to reduce overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文：x 的形状为 [batch, seq_len, dim]。
        # English: x has shape [batch, seq_len, dim].
        batch, seq_len, _ = x.shape

        # 中文：生成位置编号 [0, 1, ..., seq_len - 1]，
        # 并扩展成 [batch, seq_len]，让 batch 中每个样本使用相同的位置编号。
        # English: Create position indices [0, 1, ..., seq_len - 1],
        # and expand them to [batch, seq_len] so each sample in the batch uses the same positions.
        positions = torch.arange(seq_len, device=x.device).expand(batch, seq_len)

        # 中文：通过 self.position(positions) 查表得到位置向量，
        # 然后加到 token embedding x 上，最后经过 dropout。
        # English: Use self.position(positions) to look up position vectors,
        # add them to the token embeddings x, and then apply dropout.
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

