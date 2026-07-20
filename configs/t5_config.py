from dataclasses import dataclass


@dataclass
class T5Config:
    src_vocab_size: int = 32000  # 输入端词表大小 / Source vocabulary size for encoder input tokens.
    tgt_vocab_size: int = 32000  # 输出端词表大小 / Target vocabulary size for decoder output tokens.
    dim: int = 256  # Transformer 隐藏层维度 / Transformer hidden feature dimension.
    num_layers: int = 4  # Encoder 和 Decoder 的层数 / Number of encoder layers and decoder layers.
    num_heads: int = 4  # 注意力头数 / Number of attention heads.
    mlp_dim: int = 1024  # 前馈网络中间层维度 / Feed-forward network inner dimension.
    max_len: int = 128  # 最大序列长度 / Maximum supported sequence length.
    dropout: float = 0.1  # Dropout 概率 / Dropout probability for regularization.
    pad_id: int = 0  # Padding token 的 id / Token id used for padding positions.

    # Q1: T5 是什么模型？/ What kind of model is T5?
    # A1: T5 是 Text-to-Text Transfer Transformer 的缩写，是 Google 提出的一个
    #    encoder-decoder Transformer 模型。
    #    T5 stands for Text-to-Text Transfer Transformer. It is an
    #    encoder-decoder Transformer model proposed by Google.
    #
    #    它的核心思想是：所有 NLP 任务都统一成 “文本输入 -> 文本输出”。
    #    也就是 text-to-text。
    #    Its core idea is to convert all NLP tasks into "text input -> text output",
    #    which is called text-to-text.
    #
    #    比如翻译、摘要、情感分类、问答，都可以写成文本到文本：
    #    For example, translation, summarization, sentiment classification, and
    #    question answering can all be written as text-to-text tasks:
    #        翻译 / translation:
    #        input:  translate English to German: I love cats
    #        output: Ich liebe Katzen
    #
    #        摘要 / summarization:
    #        input:  summarize: long article...
    #        output: short summary
    #
    #        情感分类 / sentiment classification:
    #        input:  sentiment: this movie is great
    #        output: positive
    #
    #        问答 / question answering:
    #        input:  question: ... context: ...
    #        output: answer text
    #
    #    所以 T5 和 BERT/GPT 的结构区别是：
    #    So the architecture difference between T5, BERT, and GPT is:
    #        BERT: encoder-only
    #        GPT: decoder-only
    #        T5: encoder-decoder
    #
    #    T5 的结构大概是：输入文本 -> Encoder 编码输入语义 ->
    #    Decoder 根据 Encoder 输出逐步生成目标文本。
    #    The rough structure of T5 is: input text -> Encoder encodes the input
    #    meaning -> Decoder generates the target text step by step from the
    #    Encoder output.
    #
    #    其中 Encoder 读完整输入，比如原文、问题、上下文；Decoder 一步步生成输出，
    #    比如翻译结果、摘要、答案；Cross-attention 让 decoder 读取 encoder 的信息。
    #    The Encoder reads the full input, such as the source text, question, or
    #    context. The Decoder generates the output step by step, such as a
    #    translation, summary, or answer. Cross-attention lets the decoder read
    #    information from the encoder.
    #
    #    在这个项目里，configs/t5_config.py 对应的是一个简化版的 T5/翻译类模型：
    #    models/encoder_decoder.py。它不是完整工业级 T5，只是用来学习 T5 这种
    #    encoder-decoder / seq2seq 架构。
    #    In this project, configs/t5_config.py corresponds to a simplified
    #    T5/translation-style model: models/encoder_decoder.py. It is not a full
    #    production T5 model; it is for learning the encoder-decoder / seq2seq
    #    architecture.

