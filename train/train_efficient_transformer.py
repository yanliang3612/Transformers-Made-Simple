"""Train efficient_transformer.py on a tiny long-ish text classification task.

Run:
    python -m train.train_efficient_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.efficient_transformer import EfficientTransformer
from modules.masks import padding_mask


def make_long_text_batch(batch_size: int = 16, seq_len: int = 64, vocab_size: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    """Create longer sequences for the linear-attention example.

    Label rule:
        class 1 if a marker token appears near the end, class 0 otherwise.
    """

    tokens = torch.randint(4, vocab_size, (batch_size, seq_len))
    tokens[:, 0] = 1
    labels = torch.randint(0, 2, (batch_size,))
    for i, label in enumerate(labels):
        marker = 9 if label.item() == 1 else 5
        tokens[i, -8:-4] = marker
    return tokens, labels


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 96
    model = EfficientTransformer(
        vocab_size=vocab_size,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        max_len=128,
        num_classes=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(40):
        tokens, labels = make_long_text_batch(vocab_size=vocab_size)
        logits = model(tokens, padding_mask(tokens, pad_id=0))
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} efficient_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

