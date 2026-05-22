from dataclasses import dataclass


@dataclass
class DiTConfig:
    image_size: int = 32  # 图片高宽 / Image height and width.
    patch_size: int = 4  # Patch 边长 / Side length of each image patch.
    in_channels: int = 3  # 输入图片通道数 / Number of input image channels.
    dim: int = 256  # Transformer 隐藏层维度 / Transformer hidden feature dimension.
    num_layers: int = 4  # Transformer 层数 / Number of Transformer blocks.
    num_heads: int = 4  # 注意力头数 / Number of attention heads.
    mlp_dim: int = 1024  # 前馈网络中间层维度 / Feed-forward network inner dimension.
    dropout: float = 0.1  # Dropout 概率 / Dropout probability for regularization.
    num_diffusion_steps: int = 1000  # 扩散时间步总数 / Total number of diffusion timesteps.

