"""Normalization layers.

LayerNorm is the standard choice in Transformers. RMSNorm is a simpler variant
used by several modern decoder-only language models.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    中文说明：
    这是 RMSNorm (均方根层归一化)模块。与 LayerNorm 不同，它不会先减去
    均值, 而是直接计算输入在最后一维上的均方根 (RMS)，再用 x / RMS 做
    缩放归一化，最后乘一个可学习的 weight 参数进行仿射变换。这样既保留了
    归一化的稳定训练效果，又比 LayerNorm 更省计算，因此在 LLaMA 等现代
    大语言模型中被广泛使用。

    English explanation:
    This is an RMSNorm (Root Mean Square Layer Normalization) module. Unlike
    LayerNorm, it does not subtract the mean first. Instead, it computes the
    root mean square (RMS) along the last dimension, normalizes with x / RMS,
    and then applies a learnable weight for an affine transform. This keeps
    training stable while being slightly cheaper than LayerNorm, which is why
    it is widely used in modern LLMs such as LLaMA.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        # 中文：保存一个很小的 eps，防止除零时分母为 0。
        # English: Store a small eps to avoid division by zero in the denominator.
        self.eps = eps

        # 中文：创建可学习的缩放参数 weight，初始化为全 1，形状为 [dim]。
        # English: Create a learnable scaling parameter weight, initialized to ones,
        # with shape [dim].
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文：x 的形状为 (..., dim)，最后一维是特征维度。
        # English: x has shape (..., dim), where the last dimension is the feature dimension.

        # 中文：对 x 平方，在最后一维求均值，加上 eps 后开方，得到 RMS。
        # English: Square x, take the mean over the last dimension, add eps,
        # and take the square root to get RMS.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # 中文：先除以 RMS 做归一化，再乘以 weight 做可学习缩放，并返回结果。
        # English: Divide by RMS to normalize, multiply by weight for learnable scaling,
        # and return the result.
        return self.weight * (x / rms)


def build_norm(dim: int, norm_type: str = "layernorm") -> nn.Module:
    """Small factory so model files can switch normalization style easily."""

    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")

