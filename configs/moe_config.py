from dataclasses import dataclass


@dataclass
class MoEConfig:
    vocab_size: int = 32000
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 1024
    num_experts: int = 4
    max_len: int = 128
    num_classes: int = 2
    dropout: float = 0.1

