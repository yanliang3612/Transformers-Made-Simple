"""Train encoder_only.py on a tiny synthetic text classification task.

Run:
    python -m train.train_encoder_only
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.encoder_only import EncoderOnlyTransformer
from modules.masks import padding_mask


def make_tiny_text_batch(batch_size: int = 16, seq_len: int = 12, vocab_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Create toy sentences.

    Label rule:
        class 1 if token 7 appears more often than token 3, else class 0.
    This gives the encoder a simple global pattern to learn.
    """

    tokens = torch.randint(4, vocab_size, (batch_size, seq_len))
    tokens[:, 0] = 1  # Pretend token 1 is [CLS].
    labels = torch.randint(0, 2, (batch_size,))
    for i, label in enumerate(labels):
        if label.item() == 1:
            tokens[i, 2:5] = 7
        else:
            tokens[i, 2:5] = 3
    return tokens, labels


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 64
    model = EncoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        max_len=32,
        num_classes=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(40):
        tokens, labels = make_tiny_text_batch(vocab_size=vocab_size)
        outputs = model(tokens, padding_mask(tokens, pad_id=0))
        loss = F.cross_entropy(outputs["class_logits"], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} encoder_only_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

