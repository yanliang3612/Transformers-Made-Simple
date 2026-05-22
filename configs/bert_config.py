from dataclasses import dataclass


@dataclass
class BertConfig:
    vocab_size: int = 30522  # 词表大小 / Vocabulary size: number of token ids the model can represent.
    dim: int = 256  # 隐藏层维度 / Hidden size: embedding and Transformer feature dimension.
    num_layers: int = 4  # Encoder 层数 / Number of Transformer encoder blocks.
    num_heads: int = 4  # 注意力头数 / Number of self-attention heads.
    mlp_dim: int = 1024  # 前馈网络中间层维度 / Feed-forward network inner dimension.
    max_len: int = 128  # 最大序列长度 / Maximum input sequence length.
    num_classes: int = 2  # 分类类别数 / Number of output classes for classification.
    dropout: float = 0.1  # Dropout 概率 / Dropout probability used for regularization.
    pad_id: int = 0  # Padding token 的 id / Token id used for padding positions.
    mask_id: int = 103  # [MASK] token 的 id / Token id used to replace masked tokens in MLM.

    # Q1: BERT 是 encoder-only 吗？/ Is BERT encoder-only?
    # A1: 是的。BERT 只使用 Transformer 的 Encoder 部分，没有 Decoder。
    #    Yes. BERT only uses Transformer encoder blocks and does not have a decoder.

    # Q2: MLM 是句子分类吗？/ Is MLM sentence classification?
    # A2: 不是。MLM 是 masked language modeling，目标是预测被遮住的 token；
    #    句子分类是另一个任务，通常使用 [CLS] 向量预测整个句子的类别。
    #    No. MLM predicts masked tokens, while sentence classification predicts
    #    one label for the whole sentence, usually from the [CLS] vector.

    # Q3: 既然 BERT 是 encoder-only，为什么还需要 mask？/ Why does encoder-only BERT still need masks?
    # A3: Encoder-only 只是模型结构，MLM 是训练目标。BERT 的 self-attention 是双向的；
    #    如果不遮住目标 token，模型会直接看到答案，所以需要用 [MASK] 隐藏原 token。
    #    Encoder-only describes the architecture, while MLM describes the training
    #    objective. Because BERT uses bidirectional self-attention, the target
    #    token must be hidden to avoid leaking the answer.

    # Q4: 为什么 MLM 需要 mask_id？/ Why does MLM need mask_id?
    # A4: mask_id 是 [MASK] token 的编号。MLM 需要一个明确的占位符告诉模型：
    #    这个位置原来的 token 被遮住了，请根据左右上下文预测它。
    #    mask_id is the id of the [MASK] token. MLM needs this explicit placeholder
    #    to mark a hidden token position that should be predicted from context.

    # Q5: 为什么不能直接删除被 mask 的 token？/ Why not delete the masked token?
    # A5: 删除 token 会改变句子长度和位置，模型也不知道原来这里有一个词需要预测；
    #    [MASK] 的作用是保留位置，但隐藏答案。
    #    Deleting the token changes sequence length and positions. [MASK] keeps
    #    the position while hiding the answer.

    # Q6: 为什么不能保留原 token 让模型预测？/ Why not keep the original token?
    # A6: 因为 BERT 是双向 attention，模型会直接看到原 token，相当于泄露答案。
    #    Since BERT attends bidirectionally, keeping the original token would
    #    reveal the answer to the model.

    # Q7: pad_id 是什么？/ What is pad_id?
    # A7: pad_id 是 padding token 的编号。batch 中句子长度不同时，用 padding 补齐到相同长度；
    #    attention mask 会让模型忽略这些不是真实文本的 fake positions。
    #    pad_id marks padding positions. They are added only to make sequences in
    #    a batch the same length, and attention masks tell the model to ignore them.

    # Q8: dropout 是什么，它为什么能降低过拟合？/ What is dropout, and why can it reduce overfitting?
    # A8: dropout 是训练时随机丢弃一部分激活值的概率。数学上，如果某层激活是 h，
    #    dropout 会采样一个 Bernoulli mask: m_i ~ Bernoulli(1 - p)，然后使用
    #    h'_i = h_i * m_i / (1 - p)，其中 p 是 dropout 概率。除以 (1 - p) 是为了
    #    让训练时的期望保持不变：E[h'_i] = h_i。
    #    Dropout randomly drops activations during training. Mathematically, for
    #    activation h, it samples a Bernoulli mask m_i ~ Bernoulli(1 - p) and uses
    #    h'_i = h_i * m_i / (1 - p), where p is the dropout probability. Dividing
    #    by (1 - p) keeps the expected activation unchanged: E[h'_i] = h_i.
    #
    #    它能降低过拟合，是因为每次训练都相当于在一个不同的“子网络”上优化参数；
    #    同一个特征不能总是假设其他特征一定存在，所以模型不容易记住训练集里的偶然模式。
    #    这类似于训练许多共享参数的模型并做近似集成，从而降低方差、减少 co-adaptation。
    #    It reduces overfitting because every step trains a different sub-network.
    #    Features cannot rely on other specific features always being present, so
    #    the model is less likely to memorize accidental patterns in the training
    #    set. This behaves like an approximate ensemble of many shared-parameter
    #    models, reducing variance and feature co-adaptation.
    #
    #    在 eval / inference 模式下，PyTorch 会自动关闭 dropout，使用完整网络做预测。
    #    In eval/inference mode, PyTorch disables dropout and uses the full network.

