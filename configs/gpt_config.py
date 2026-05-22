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

    # Q: 为什么 vocab_size 可以是 32000？GPT 这么强，只预测 32000 个 token 真的够吗？
    # A: 是的，因为 GPT 预测的是 token，不一定是完整的词。tokenizer 会把文本拆成有限数量的
    #    基础片段，比如常见词、子词、字符、符号等。一个复杂词或新词可以由多个 token 组合出来。
    #    这类似于英文字母只有 26 个，但可以组合出非常多单词。
    #    Yes. GPT predicts tokens, not necessarily whole words. A tokenizer splits
    #    text into a fixed vocabulary of pieces such as common words, subwords,
    #    characters, and symbols. Rare or new words can be represented as multiple
    #    tokens, similar to how 26 English letters can form many words.
    #
    # Q: 那 32000 个 token 可以表达任何词汇吗？/ Can 32,000 tokens express any word?
    # A: 更准确地说，只要 tokenizer 有兜底拆分机制，就几乎可以表示任意文本；
    #    但生僻词、代码、网址等可能会被拆得更碎，序列更长，效率更低。
    #    More precisely, if the tokenizer has a fallback splitting strategy, it can
    #    represent almost any text. However, rare words, code, or URLs may be split
    #    into more tokens, making the sequence longer and less efficient.

