from dataclasses import dataclass


@dataclass
class DiTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 1024
    dropout: float = 0.1
    num_diffusion_steps: int = 1000

