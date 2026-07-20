from dataclasses import dataclass


@dataclass
class BertConfig:
    vocab_size: int = 30522
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 1024
    max_len: int = 128
    num_classes: int = 2
    dropout: float = 0.1
    pad_id: int = 0
    mask_id: int = 103

