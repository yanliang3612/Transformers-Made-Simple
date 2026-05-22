"""Train hybrid_transformer.py on a tiny local-pattern text classification task.

Run:
    python -m train.train_hybrid_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.hybrid_transformer import HybridTransformer
from modules.masks import padding_mask


def make_local_pattern_batch(batch_size: int = 16, seq_len: int = 20, vocab_size: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """Create data where local n-gram patterns determine the label."""

    tokens = torch.randint(5, vocab_size, (batch_size, seq_len))
    tokens[:, 0] = 1
    labels = torch.randint(0, 2, (batch_size,))
    for i, label in enumerate(labels):
        if label.item() == 0:
            tokens[i, 8:11] = torch.tensor([10, 11, 12])
        else:
            tokens[i, 8:11] = torch.tensor([20, 21, 22])
    return tokens, labels


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 80
    model = HybridTransformer(
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
        tokens, labels = make_local_pattern_batch(vocab_size=vocab_size)
        logits = model(tokens, padding_mask(tokens, pad_id=0))
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} hybrid_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

