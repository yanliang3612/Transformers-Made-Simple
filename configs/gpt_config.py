from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32000  # 词表大小 / Vocabulary size: number of token ids the model can represent.
    dim: int = 256  # Transformer 隐藏层维度 / Transformer hidden feature dimension.
    num_layers: int = 4  # Decoder 层数 / Number of Transformer decoder blocks.
    num_heads: int = 4  # 注意力头数 / Number of self-attention heads.
    mlp_dim: int = 1024  # 前馈网络中间层维度 / Feed-forward network inner dimension.
    max_len: int = 128  # 最大序列长度 / Maximum supported input sequence length.
    dropout: float = 0.1  # Dropout 概率 / Dropout probability for regularization.
    pad_id: int = 0  # Padding token 的 id / Token id used for padding positions.

