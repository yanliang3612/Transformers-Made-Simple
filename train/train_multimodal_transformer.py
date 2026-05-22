"""Train multimodal_transformer.py on tiny text + image pairs.

Run:
    python -m train.train_multimodal_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.multimodal_transformer import MultimodalTransformer


def make_multimodal_batch(
    batch_size: int = 16,
    seq_len: int = 10,
    vocab_size: int = 64,
    image_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create paired text and image examples.

    Label rule:
        class 0: text contains token 11 and image has upper-left square.
        class 1: text contains token 22 and image has lower-right square.
    """

    tokens = torch.randint(3, vocab_size, (batch_size, seq_len))
    images = torch.randn(batch_size, 3, image_size, image_size) * 0.05
    labels = torch.randint(0, 2, (batch_size,))

    for i, label in enumerate(labels):
        if label.item() == 0:
            tokens[i, 2:5] = 11
            images[i, :, 4:12, 4:12] += 1.0
        else:
            tokens[i, 2:5] = 22
            images[i, :, 20:28, 20:28] += 1.0
    return tokens, images, labels


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 64
    model = MultimodalTransformer(
        vocab_size=vocab_size,
        image_size=32,
        patch_size=4,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        num_classes=2,
        max_text_len=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(40):
        tokens, images, labels = make_multimodal_batch(vocab_size=vocab_size)
        logits = model(tokens, images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} multimodal_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

