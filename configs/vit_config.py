from dataclasses import dataclass


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 1024
    dropout: float = 0.1

    # Q1: Vision Transformer 是 decoder-only 吗，还是哪种模型？
    #    Is Vision Transformer decoder-only, or what kind of model is it?
    # A1: Vision Transformer（ViT）不是 decoder-only，它更接近 encoder-only Transformer。
    #    Vision Transformer (ViT) is not decoder-only. It is closer to an
    #    encoder-only Transformer.
    #
    #    在这个项目里，models/vision_transformer.py 明确复用了 EncoderBlock：
    #    In this project, models/vision_transformer.py explicitly reuses EncoderBlock:
    #        from models.encoder_only import EncoderBlock
    #
    #    并且每一层是：
    #    And each layer is:
    #        self.layers = nn.ModuleList(
    #            [EncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
    #        )
    #
    #    所以这个 ViT 的结构是：
    #    So the structure of this ViT is:
    #        image
    #        -> 切成 patches / split into patches
    #        -> patch embedding
    #        -> 加 cls token / add cls token
    #        -> 加 position embedding / add position embedding
    #        -> 多层 encoder-style Transformer blocks / multiple encoder-style Transformer blocks
    #        -> 取 cls token / take cls token
    #        -> 分类 head / classification head
    #
    #    也就是：ViT = encoder-only style Transformer for images。
    #    In short: ViT = encoder-only style Transformer for images.
    #
    #    它和 BERT 类似，都是 encoder-only 风格，只不过 BERT 的输入是文本 token，
    #    ViT 的输入是图像 patch token。
    #    It is similar to BERT because both are encoder-only style models. The
    #    difference is that BERT takes text tokens, while ViT takes image patch tokens.
    #
    #    对比一下：
    #    Comparison:
    #        BERT: 文本 token -> encoder blocks -> [CLS] 分类/MLM
    #              text tokens -> encoder blocks -> [CLS] classification/MLM
    #        ViT:  图片 patch -> encoder blocks -> [CLS] 图像分类
    #              image patches -> encoder blocks -> [CLS] image classification
    #        GPT:  文本 token -> decoder-only causal blocks -> 下一个 token 预测
    #              text tokens -> decoder-only causal blocks -> next-token prediction
    #        T5:   encoder + decoder -> seq2seq 生成
    #              encoder + decoder -> seq2seq generation
    #
    #    ViT 不需要像 GPT 那样用 causal mask，因为它不是从左到右生成下一个 token；
    #    它通常是看整张图片的所有 patch，然后做分类。
    #    ViT does not need a causal mask like GPT, because it is not generating
    #    the next token from left to right. It usually looks at all image patches
    #    and then performs classification.
    #
    #    所以它不是 decoder-only，而是 encoder-only / encoder-style 模型。
    #    Therefore, it is not decoder-only; it is an encoder-only / encoder-style model.

    # Q2: -> 切成 patches -> patch embedding -> 加 cls token
    #    这里的具体流程是啥，cls token 是什么东西？
    #    What is the concrete flow of splitting into patches, patch embedding,
    #    and adding a cls token? What is a cls token?
    # A2: 这里说的是 ViT 把一张图片变成 Transformer 能处理的 token 序列。
    #    This describes how ViT converts an image into a token sequence that a
    #    Transformer can process.
    #
    #    具体流程是：
    #    The concrete flow is:
    #        图片 / image
    #        -> 切成小块 patches / split into small patches
    #        -> 每个 patch 变成一个向量 patch embedding / convert each patch into a vector
    #        -> 在最前面加一个可学习的 [CLS] token / add a learnable [CLS] token at the front
    #        -> 送进 Transformer encoder / send into the Transformer encoder
    #        -> 最后取 [CLS] token 的输出做分类 / use the final [CLS] output for classification
    #
    #    假设 image_size = 32，patch_size = 4，in_channels = 3，
    #    也就是一张 3 x 32 x 32 的 RGB 图片。每个 patch 是 4 x 4，
    #    所以一边可以切 32 / 4 = 8 个 patch，整张图可以切成 8 * 8 = 64 个 patch。
    #    每个 patch 的原始像素数量是 3 * 4 * 4 = 48。
    #    Suppose image_size = 32, patch_size = 4, and in_channels = 3, meaning
    #    a 3 x 32 x 32 RGB image. Each patch is 4 x 4, so each side has
    #    32 / 4 = 8 patches, and the whole image has 8 * 8 = 64 patches.
    #    Each patch contains 3 * 4 * 4 = 48 raw pixel values.
    #
    #    Transformer 不能直接处理二维图片，它处理的是 token 向量序列。
    #    所以 ViT 会把每个 patch 映射成一个 dim 维向量。比如 dim = 256，
    #    每个 4x4 RGB patch 会变成一个 256 维向量。
    #    A Transformer does not directly process a 2D image; it processes a
    #    sequence of token vectors. Therefore, ViT maps each patch into a
    #    dim-dimensional vector. For example, if dim = 256, each 4x4 RGB patch
    #    becomes a 256-dimensional vector.
    #
    #    在这个项目里，PatchEmbedding 用 Conv2d 实现：
    #    In this project, PatchEmbedding is implemented with Conv2d:
    #        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
    #
    #    这相当于用一个卷积一次性完成：切 patch + 线性投影成 embedding。
    #    输出形状大概是 (batch, num_patches, dim)，也就是 (batch, 64, 256)。
    #    This is equivalent to using one convolution to do both patch splitting
    #    and linear projection into embeddings. The output shape is roughly
    #    (batch, num_patches, dim), for example (batch, 64, 256).
    #
    #    cls_token 是一个可学习的特殊 token，不是图片里真实存在的 patch。
    #    The cls_token is a learnable special token; it is not a real patch from the image.
    #
    #    在代码里：
    #    In the code:
    #        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
    #
    #    它的 shape 是 (1, 1, dim)，训练时会被模型学习更新。
    #    Its shape is (1, 1, dim), and it is learned during training.
    #
    #    forward 里会把它复制到每个 batch，然后拼到 patch token 前面：
    #    In forward, it is expanded for each batch and concatenated before patch tokens:
    #        cls = self.cls_token.expand(batch, -1, -1)
    #        x = torch.cat([cls, patches], dim=1)
    #
    #    如果原来有 64 个 patch token，加上 cls token 后，序列长度变成 65：
    #    If there were originally 64 patch tokens, after adding the cls token,
    #    the sequence length becomes 65:
    #        patches shape = (batch, 64, dim)
    #        x shape = (batch, 65, dim)
    #
    #    为什么要加 cls token？因为图片分类需要一个“整张图的表示”。
    #    Transformer encoder 处理完后，每个 token 都会和其他 token 交互。
    #    cls_token 放在最前面，它通过 self-attention 可以关注所有 patch token，
    #    逐渐聚合整张图片的信息。
    #    Why add a cls token? Image classification needs one representation for
    #    the whole image. After the Transformer encoder, every token has interacted
    #    with other tokens. The cls_token is placed at the front and can attend to
    #    all patch tokens through self-attention, gradually aggregating whole-image information.
    #
    #    最后代码里取第 0 个 token 做分类：
    #    Finally, the code takes token 0 for classification:
    #        return self.head(self.norm(x[:, 0]))
    #
    #    这里的 x[:, 0] 就是处理后的 cls_token。
    #    Here, x[:, 0] is the processed cls_token.
    #
    #    所以 cls token = 用来收集全局图像信息的分类 token。
    #    So, cls token = a classification token used to collect global image information.
    #
    #    它类似 BERT 里的 [CLS] token：
    #    It is similar to the [CLS] token in BERT:
    #        BERT: [CLS] 收集整句文本信息，用来文本分类
    #              [CLS] collects whole-sentence information for text classification
    #        ViT:  [CLS] 收集整张图片信息，用来图像分类
    #              [CLS] collects whole-image information for image classification
    #
    #    一句话总结：patch embedding 把图片块变成 token；cls token 是额外加进去的
    #    可学习 token，最终用它的输出代表整张图来做分类。
    #    In one sentence: patch embedding turns image patches into tokens; the
    #    cls token is an extra learnable token, and its final output represents
    #    the whole image for classification.

    # Q3: 还是没看懂 CLS token，可以举一个具体的例子吗？
    #    I still do not understand CLS token. Can you give a concrete example?
    # A3: 可以。把 CLS token 想成一个“班长 token”：它自己不代表图片里的某一块，
    #    但它负责和所有图片块交流，最后拿它的状态去做分类。
    #    Sure. Think of the CLS token as a "class leader token": it does not
    #    represent any real image patch itself, but it communicates with all image
    #    patches, and its final state is used for classification.
    #
    #    假设一张猫图被切成 4 个 patch：
    #    Suppose a cat image is split into 4 patches:
    #        patch1 = 左上角：猫耳朵 / top-left: cat ears
    #        patch2 = 右上角：背景 / top-right: background
    #        patch3 = 左下角：猫身体 / bottom-left: cat body
    #        patch4 = 右下角：猫尾巴 / bottom-right: cat tail
    #
    #    ViT 不直接输入“猫图”，而是输入 token 序列：
    #    ViT does not directly input the "cat image"; it inputs a token sequence:
    #        [CLS], patch1, patch2, patch3, patch4
    #
    #    一开始：
    #    At the beginning:
    #        [CLS] 只是一个可学习向量，还不知道图片内容
    #              [CLS] is only a learnable vector and does not know the image content yet
    #        patch1 包含猫耳朵信息 / patch1 contains cat-ear information
    #        patch2 包含背景信息 / patch2 contains background information
    #        patch3 包含猫身体信息 / patch3 contains cat-body information
    #        patch4 包含猫尾巴信息 / patch4 contains cat-tail information
    #
    #    进入 Transformer 后，self-attention 会让每个 token 看其他 token。
    #    于是 [CLS] 可以去关注所有 patch：
    #    After entering the Transformer, self-attention lets each token look at
    #    other tokens. So [CLS] can attend to all patches:
    #        [CLS] attention:
    #        patch1 猫耳朵 -> 很重要 / cat ears -> very important
    #        patch2 背景   -> 不太重要 / background -> less important
    #        patch3 猫身体 -> 很重要 / cat body -> very important
    #        patch4 猫尾巴 -> 有用 / cat tail -> useful
    #
    #    经过多层 Transformer 后，[CLS] 的向量就从“空白班长”变成了
    #    “总结了整张图片的向量”：
    #    After multiple Transformer layers, the [CLS] vector changes from a
    #    "blank class leader" into a vector that summarizes the whole image:
    #        [CLS] ~= 这张图里有猫的整体特征
    #        [CLS] ~= global features indicating that this image contains a cat
    #
    #    最后分类 head 只看 [CLS]：
    #    Finally, the classification head only looks at [CLS]:
    #        classifier([CLS]) -> cat
    #
    #    也就是代码里的：
    #    This is the code:
    #        return self.head(self.norm(x[:, 0]))
    #
    #    这里 x[:, 0] 就是序列第一个 token，也就是 [CLS]。
    #    Here, x[:, 0] is the first token in the sequence, which is [CLS].
    #
    #    为什么不用所有 patch 平均一下？也可以，有些 ViT 变体就是 mean pooling。
    #    但原始 ViT/BERT 风格常用 [CLS]，好处是：让模型自己学习该从哪些 patch 汇总信息，
    #    而不是简单平均。
    #    Why not simply average all patches? That is also possible, and some ViT
    #    variants use mean pooling. But the original ViT/BERT style commonly uses
    #    [CLS]. The benefit is that the model can learn which patches to summarize
    #    from, instead of using a simple average.
    #
    #    一个很小的数值例子：
    #    A tiny numerical example:
    #        输入序列 / input sequence:
    #        [CLS], 耳朵patch, 背景patch, 身体patch, 尾巴patch
    #        [CLS], ear patch, background patch, body patch, tail patch
    #
    #        某一层 attention 后 [CLS] 可能学到：
    #        After one attention layer, [CLS] may learn:
    #        [CLS] = 0.40 * 耳朵patch / ear patch
    #              + 0.05 * 背景patch / background patch
    #              + 0.35 * 身体patch / body patch
    #              + 0.20 * 尾巴patch / tail patch
    #
    #    这样 [CLS] 就更关注对分类有用的区域。多层之后，它会变成整张图片的全局表示。
    #    In this way, [CLS] focuses more on regions useful for classification.
    #    After multiple layers, it becomes a global representation of the image.
    #
    #    一句话：CLS token 不是图片的一部分，而是模型额外放进去的“汇总 token”；
    #    它通过 self-attention 读取所有 patch，最后用它来代表整张图做分类。
    #    In one sentence: the CLS token is not part of the image; it is an extra
    #    "summary token" inserted by the model. It reads all patches through
    #    self-attention, and its final state represents the whole image for classification.

    # Q4: patch embedding 输出形状 (batch, 64, 256) 里的 64 是啥？
    #    所有 patch embedding 的意思是不是类似于之前一个 token 的 embedding，
    #    然后一张图片所有 patch 的 embedding 组合到一起，类似一个句子所有 token
    #    的 embedding 组合到一起？
    #    What does the 64 in patch embedding output shape (batch, 64, 256) mean?
    #    Are patch embeddings similar to token embeddings, where all patch
    #    embeddings of one image are combined like all token embeddings of one sentence?
    # A4: 对，这个理解基本完全正确。
    #    Yes, that understanding is basically correct.
    #
    #    这里的 64 是一张图片被切出来的 patch 数量，也就是：
    #    Here, 64 is the number of patches cut from one image:
    #        num_patches = (image_size / patch_size) * (image_size / patch_size)
    #
    #    如果 image_size = 32，patch_size = 4，那么图片每条边可以切成：
    #    If image_size = 32 and patch_size = 4, each image side is split into:
    #        32 / 4 = 8
    #
    #    整张图就是：
    #    So the whole image has:
    #        8 * 8 = 64 个 patch / 64 patches
    #
    #    所以输出 (batch, 64, 256) 的含义是：
    #    Therefore, output shape (batch, 64, 256) means:
    #        batch: 一批里有多少张图片 / how many images are in the batch
    #        64: 每张图片有 64 个 patch token / each image has 64 patch tokens
    #        256: 每个 patch token 的 embedding 维度 / embedding dimension of each patch token
    #
    #    你说的类比是对的：
    #    Your analogy is correct:
    #        文本 Transformer / text Transformer:
    #        一个句子 = 多个 token / one sentence = multiple tokens
    #        每个 token -> token embedding / each token -> token embedding
    #        所有 token embedding 组合成序列 / all token embeddings form a sequence
    #
    #        ViT:
    #        一张图片 = 多个 patch / one image = multiple patches
    #        每个 patch -> patch embedding / each patch -> patch embedding
    #        所有 patch embedding 组合成序列 / all patch embeddings form a sequence
    #
    #    比如文本里：
    #    For example, in text:
    #        "I love cats"
    #        -> token1, token2, token3
    #        -> embedding1, embedding2, embedding3
    #        -> shape: (batch, seq_len, dim)
    #
    #    ViT 里：
    #    In ViT:
    #        一张 32x32 图片 / one 32x32 image
    #        -> 64 个 patch / 64 patches
    #        -> 64 个 patch embedding / 64 patch embeddings
    #        -> shape: (batch, 64, dim)
    #
    #    如果再加上 CLS token，就变成：
    #    If the CLS token is added, it becomes:
    #        (batch, 65, 256)
    #
    #    因为：
    #    Because:
    #        1 个 CLS token + 64 个 patch token = 65 个 token
    #        1 CLS token + 64 patch tokens = 65 tokens
    #
    #    一句话：patch embedding 就是图像版的 token embedding；
    #    一张图片的所有 patch embedding 拼在一起，就像一句话的所有 token embedding 拼在一起。
    #    In one sentence: patch embedding is the image version of token embedding;
    #    all patch embeddings of one image are combined just like all token
    #    embeddings of one sentence are combined.
