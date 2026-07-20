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
        # and expand them to [batch, seq_len] so each sample in the batch
        # uses the same positions.
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
    """
    中文说明：
    这个函数实现了 RoPE (Rotary Positional Embedding,旋转位置编码)。
    RoPE 不像传统位置编码那样把位置向量直接加到 token embedding 上，
    而是根据 token 的位置对 attention 中的 query 和 key 向量进行二维旋转。
    具体来说，它会把 q 和 k 的 head_dim 维度两两配对，例如 (0,1)、
    (2,3)、(4,5)，然后对每一对维度按照位置 p 和频率 theta_i 生成的角度
    p * theta_i 进行旋转。这样在后续 attention 计算 q 和 k 的点积时，
    score 中会自然包含相对位置信息 t - p, 从而让模型感知 token 之间的
    相对距离。

    English explanation:
    This function implements RoPE, short for Rotary Positional Embedding.
    Unlike traditional positional encoding methods that add position vectors
    directly to token embeddings, RoPE injects positional information by
    rotating the query and key vectors in attention. Specifically, it pairs
    the head_dim dimensions of q and k, such as (0,1), (2,3), and (4,5), and
    applies a 2D rotation to each pair using an angle p * theta_i determined
    by the token position p and the frequency theta_i. After this rotation,
    the dot product between q and k naturally contains relative positional
    information t - p, allowing the model to capture relative distances
    between tokens.
    """

    # 中文：获取 attention head 的维度，也就是 q 和 k 最后一维的大小。
    # English: Get the attention head dimension, which is the size of the
    # last dimension of q and k.
    head_dim = q.size(-1)

    # 中文：RoPE 需要把维度两两配对做二维旋转，所以 head_dim 必须是偶数。
    # English: RoPE pairs dimensions for 2D rotation, so head_dim must be even.
    if head_dim % 2 != 0:

        # 中文：如果 head_dim 不是偶数，就报错。
        # English: Raise an error if head_dim is not even.
        raise ValueError("RoPE requires an even head_dim")

    # 中文：获取 q 所在的设备，例如 CPU 或 GPU，
    # 确保后面创建的张量在同一个设备上。
    # English: Get the device of q, such as CPU or GPU, so newly created
    # tensors are on the same device.
    device = q.device

    # 中文：获取序列长度；q 的形状是 (batch, heads, seq_len, head_dim)，
    # 所以倒数第二维是 seq_len。
    # English: Get the sequence length. Since q has shape
    # (batch, heads, seq_len, head_dim), the second-to-last dimension is seq_len.
    seq_len = q.size(-2)

    # 中文：计算每一对维度对应的频率 theta_i = 1 / 10000^(2i / head_dim)。
    # English: Compute the frequency for each dimension pair:
    # theta_i = 1 / 10000^(2i / head_dim).
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # 中文：生成位置编号 [0, 1, 2, ..., seq_len - 1]。
    # English: Create position indices [0, 1, 2, ..., seq_len - 1].
    positions = torch.arange(seq_len, device=device).float()

    # 中文：计算每个位置 p 和每个频率 theta_i 的乘积，
    # 得到旋转角度 p * theta_i。
    # English: Compute the product of each position p and each frequency theta_i,
    # giving the rotation angle p * theta_i.
    freqs = torch.einsum("i,j->ij", positions, theta)

    # 中文：对旋转角度取 sin，并扩展成形状 (1, 1, seq_len, head_dim / 2)，
    # 方便和 q、k 广播相乘。
    # English: Apply sin to the rotation angles and reshape to
    # (1, 1, seq_len, head_dim / 2) for broadcasting with q and k.
    sin = freqs.sin()[None, None, :, :]

    # 中文：对旋转角度取 cos，并扩展成形状 (1, 1, seq_len, head_dim / 2)，
    # 方便和 q、k 广播相乘。
    # English: Apply cos to the rotation angles and reshape to
    # (1, 1, seq_len, head_dim / 2) for broadcasting with q and k.
    cos = freqs.cos()[None, None, :, :]

    # 中文：定义内部函数 rotate，用来对输入张量 x 的每两个维度做 RoPE 旋转。
    # English: Define an inner function rotate to apply RoPE rotation to every
    # pair of dimensions in x.
    def rotate(x: torch.Tensor) -> torch.Tensor:

        # 中文：取出偶数维度，例如 x_0, x_2, x_4, ...，
        # 作为每个二维向量的第一个分量。
        # English: Select even dimensions, such as x_0, x_2, x_4, ...,
        # as the first component of each 2D vector.
        x1 = x[..., 0::2]

        # 中文：取出奇数维度，例如 x_1, x_3, x_5, ...，
        # 作为每个二维向量的第二个分量。
        # English: Select odd dimensions, such as x_1, x_3, x_5, ...,
        # as the second component of each 2D vector.
        x2 = x[..., 1::2]

        # 中文：对每一对维度做二维旋转：
        # x'_1 = x1 * cos - x2 * sin,
        # x'_2 = x1 * sin + x2 * cos。
        # torch.stack(..., dim=-1) 会把旋转后的两个分量重新放到一起。
        # English: Apply 2D rotation to each dimension pair:
        # x'_1 = x1 * cos - x2 * sin,
        # x'_2 = x1 * sin + x2 * cos.
        # torch.stack(..., dim=-1) groups the two rotated components back together.
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

        # 中文：把最后两个维度合并，把形状从 (..., head_dim / 2, 2)
        # 还原成 (..., head_dim)。
        # English: Flatten the last two dimensions, converting the shape from
        # (..., head_dim / 2, 2) back to (..., head_dim).
        return rotated.flatten(-2)

    # 中文：分别对 q 和 k 应用 RoPE 旋转，并返回旋转后的位置编码版本。
    # English: Apply RoPE rotation to q and k separately, and return their
    # position-encoded versions.
    return rotate(q), rotate(k)

