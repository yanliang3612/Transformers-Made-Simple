from dataclasses import dataclass


@dataclass
class MoEConfig:
    vocab_size: int = 32000  # 词表大小 / Vocabulary size: number of token ids the model can represent.
    dim: int = 256  # Transformer 隐藏层维度 / Transformer hidden feature dimension.
    num_layers: int = 4  # MoE Transformer 层数 / Number of MoE Transformer blocks.
    num_heads: int = 4  # 注意力头数 / Number of self-attention heads.
    mlp_dim: int = 1024  # 每个 expert 的前馈网络中间层维度 / Feed-forward inner dimension for each expert.
    num_experts: int = 4  # Expert 数量 / Number of feed-forward experts in each MoE layer.
    max_len: int = 128  # 最大序列长度 / Maximum supported input sequence length.
    num_classes: int = 2  # 分类类别数 / Number of output classes for classification.
    dropout: float = 0.1  # Dropout 概率 / Dropout probability for regularization.

    # Q1: MoE Transformer block 的层数和传统 Transformer block 的层数有什么区别？
    #    What is the difference between MoE Transformer layers and standard Transformer layers?
    # A1: num_layers 都表示 Transformer block 堆叠的层数。传统 Transformer block 每层通常包含
    #    LayerNorm、self-attention、residual connection、LayerNorm、普通 FeedForward/MLP、
    #    residual connection。MoE Transformer block 的 self-attention 部分基本一样，
    #    但把普通 FeedForward/MLP 替换成 Router/Gate + 多个 Expert FeedForward。
    #    In both cases, num_layers means the number of stacked Transformer blocks.
    #    A standard Transformer block usually contains LayerNorm, self-attention,
    #    residual connection, LayerNorm, a regular FeedForward/MLP, and another
    #    residual connection. An MoE Transformer block keeps the self-attention
    #    part mostly the same, but replaces the regular FeedForward/MLP with a
    #    Router/Gate plus multiple Expert FeedForward networks.

    # Q2: 传统 Transformer block 每层包含哪些？/ What does each standard Transformer block contain?
    # A2: 通常是：输入 x -> LayerNorm -> Self-Attention -> Dropout -> Residual Add
    #    -> LayerNorm -> FeedForward/MLP -> Dropout -> Residual Add -> 输出 x。
    #    Usually: input x -> LayerNorm -> Self-Attention -> Dropout -> Residual Add
    #    -> LayerNorm -> FeedForward/MLP -> Dropout -> Residual Add -> output x.

    # Q3: MoE Transformer block 每层包含哪些？/ What does each MoE Transformer block contain?
    # A3: 通常是：输入 x -> LayerNorm -> Self-Attention -> Dropout -> Residual Add
    #    -> LayerNorm -> Router/Gate -> 选择 Expert -> MoE FeedForward -> Dropout
    #    -> Residual Add -> 输出 x。
    #    Usually: input x -> LayerNorm -> Self-Attention -> Dropout -> Residual Add
    #    -> LayerNorm -> Router/Gate -> choose Expert -> MoE FeedForward -> Dropout
    #    -> Residual Add -> output x.

    # Q4: 最核心的区别是什么？/ What is the key difference?
    # A4: 传统 Transformer 里所有 token 都经过同一个 FeedForward；MoE Transformer 里，
    #    router 会为不同 token 选择不同 expert。也就是：普通 FFN 被替换成
    #    Router + 多个 Expert FFN。
    #    In a standard Transformer, all tokens go through the same FeedForward.
    #    In an MoE Transformer, the router can send different tokens to different
    #    experts. In short, the regular FFN is replaced by Router + multiple Expert FFNs.

    # Q5: router_logits = self.gate(x); expert_ids = router_logits.argmax(dim=-1)
    #    这里的 self.gate(x) 一般是什么函数？
    #    What kind of function is self.gate(x) usually?
    # A5: 在这个项目里，self.gate(x) 是一个线性层，也就是 nn.Linear(dim, num_experts)。
    #    In this project, self.gate(x) is a linear layer, nn.Linear(dim, num_experts).
    #
    #    所以 router_logits = self.gate(x) 本质上是在做：
    #    So router_logits = self.gate(x) is essentially:
    #        router_logits = x @ W + b
    #
    #    其中 x 的 shape 是 (batch, seq_len, dim)，W 的 shape 大概是
    #    (dim, num_experts)，输出 router_logits 的 shape 是
    #    (batch, seq_len, num_experts)。
    #    Here, x has shape (batch, seq_len, dim), W is roughly
    #    (dim, num_experts), and router_logits has shape
    #    (batch, seq_len, num_experts).
    #
    #    也就是说，每个 token 的 hidden state 都会被线性层打分，得到它对每个 expert 的分数。
    #    That means each token hidden state is scored by the linear layer against each expert.
    #
    #    比如 num_experts=4，某个 token 经过 gate 后可能得到：
    #    For example, if num_experts=4, one token may get:
    #        router_logits = [1.2, -0.3, 2.5, 0.7]
    #
    #    然后 expert_ids = router_logits.argmax(dim=-1) 会选分数最大的 expert，
    #    也就是 expert_id = 2，所以这个 token 会走第 2 个 expert。
    #    Then expert_ids = router_logits.argmax(dim=-1) selects the expert with
    #    the largest score, expert_id = 2, so this token goes to expert 2.
    #
    #    一般 MoE 里的 gate 最常见是 Linear layer + Softmax + Top-k：
    #    In common MoE models, the gate is usually Linear layer + Softmax + Top-k:
    #        router_logits = gate(x)
    #        router_probs = softmax(router_logits)
    #        topk_experts = topk(router_probs)
    #
    #    你的项目里为了简单，省略了 softmax，直接用 argmax 选 top-1 expert。
    #    因为 softmax 不改变最大值的位置，所以如果只是选 top-1 expert，
    #    直接对 logits 做 argmax 也能选出同一个 expert。
    #    This project omits softmax for simplicity and directly uses argmax to
    #    choose the top-1 expert. Since softmax does not change which value is
    #    largest, argmax on logits selects the same top-1 expert.
    #
    #    更完整的 MoE router 通常还会有 softmax、top-k routing、load balancing loss、
    #    capacity limit、router noise 等机制。
    #    A fuller MoE router often also includes softmax, top-k routing,
    #    load balancing loss, capacity limits, router noise, and related mechanisms.
