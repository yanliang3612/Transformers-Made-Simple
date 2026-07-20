from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 1024
    max_len: int = 128
    dropout: float = 0.1
    pad_id: int = 0

