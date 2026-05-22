# Transformers Made Simple

这是一个面向入门学习的 Transformer 仓库。目标不是复刻工业级大模型训练框架，而是把常见 Transformer 家族拆成清楚、可运行、方便改动的小模块。

## 目录结构

```text
modules/
  attention.py              多头注意力、线性注意力
  feedforward.py            普通 MLP、SwiGLU、MoE MLP
  positional_encoding.py    正弦位置编码、可学习位置编码、RoPE
  embeddings.py             Token embedding、图像 patch embedding
  normalization.py          LayerNorm/RMSNorm 工厂
  masks.py                  causal mask、padding mask

models/
  encoder_only.py           BERT 类 encoder-only
  decoder_only.py           GPT 类 decoder-only
  encoder_decoder.py        T5/翻译类 encoder-decoder
  vision_transformer.py     ViT
  efficient_transformer.py  线性注意力示例
  moe_transformer.py        Mixture-of-Experts Transformer
  multimodal_transformer.py 文本 + 图像融合 Transformer
  diffusion_transformer.py  DiT 风格扩散去噪器
  retrieval_transformer.py  检索增强 Transformer
  hybrid_transformer.py     CNN + Attention 混合模型

configs/                    小模型配置
train/                      可运行的玩具训练脚本
utils/                      采样、checkpoint、简单 tokenizer
```

## 安装

建议先创建虚拟环境，然后安装 PyTorch：

```bash
pip install torch
```

如果你用的是 Apple Silicon，也可以直接用 PyTorch 官方命令安装对应版本。

## 快速运行

每个模型都有一个同名训练脚本。所有脚本都使用脚本内编造的小数据集，重点是帮你看懂训练流程。

Encoder-only / BERT 类文本分类：

```bash
python -m train.train_encoder_only
```

Decoder-only / GPT 类语言模型：

```bash
python -m train.train_decoder_only
```

Encoder-decoder / 翻译类反转序列任务：

```bash
python -m train.train_encoder_decoder
```

Vision Transformer / 小图像分类：

```bash
python -m train.train_vision_transformer
```

Efficient Transformer / 较长文本分类：

```bash
python -m train.train_efficient_transformer
```

MoE Transformer / 专家路由文本分类：

```bash
python -m train.train_moe_transformer
```

Multimodal Transformer / 文本 + 图像分类：

```bash
python -m train.train_multimodal_transformer
```

Diffusion Transformer / 图像去噪：

```bash
python -m train.train_diffusion_transformer
```

Retrieval Transformer / 检索增强语言模型：

```bash
python -m train.train_retrieval_transformer
```

Hybrid Transformer / CNN + Attention 文本分类：

```bash
python -m train.train_hybrid_transformer
```

这些脚本使用短文本或随机数据，重点是帮助你理解训练流程、loss 形状、输入输出，而不是训练出高质量模型。

## 推荐学习路线

1. 先读 `modules/attention.py`，弄清楚 query/key/value 的形状变化。
2. 再读 `models/encoder_only.py`，理解 self-attention + feed-forward + residual 的基本块。
3. 读 `models/decoder_only.py`，重点看 causal mask 如何防止模型偷看未来 token。
4. 读 `models/encoder_decoder.py`，理解 cross-attention 如何让 decoder 读取 encoder 的结果。
5. 最后看 ViT、MoE、DiT、多模态、检索增强这些变体。

## 常见张量形状

文本 token：

```text
token_ids: (batch, seq_len)
hidden:    (batch, seq_len, dim)
logits:    (batch, seq_len, vocab_size)
```

注意力内部：

```text
q/k/v:     (batch, heads, seq_len, head_dim)
scores:    (batch, heads, query_len, key_len)
mask:      (batch, 1, query_len, key_len) 或可广播到这个形状
```

图像 patch：

```text
images:    (batch, channels, height, width)
patches:   (batch, num_patches, dim)
```

## 下一步可以怎么改

- 把玩具数据换成真实文本数据集。
- 把 `CharTokenizer` 换成 BPE 或 SentencePiece tokenizer。
- 给 MoE 增加 load balancing loss。
- 给 decoder-only 模型加 KV cache，提高生成速度。
- 给 DiT 加完整 diffusion scheduler。
