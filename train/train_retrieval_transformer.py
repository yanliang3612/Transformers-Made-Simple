"""Train retrieval_transformer.py on a tiny retrieval-conditioned LM task.

Run:
    python -m train.train_retrieval_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.retrieval_transformer import RetrievalTransformer


def make_retrieval_batch(
    batch_size: int = 16,
    seq_len: int = 12,
    retrieved_len: int = 4,
    vocab_size: int = 70,
    dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create token sequences plus retrieved vectors.

    The retrieved vectors carry a topic signal. The target language-model tokens
    are biased toward one topic token, so cross-attention has useful context.
    """

    topic = torch.randint(0, 2, (batch_size,))
    tokens = torch.randint(4, vocab_size, (batch_size, seq_len + 1))
    retrieved = torch.randn(batch_size, retrieved_len, dim) * 0.05
    for i, t in enumerate(topic):
        topic_token = 13 if t.item() == 0 else 29
        tokens[i, 2:6] = topic_token
        retrieved[i, :, 0] = -1.0 if t.item() == 0 else 1.0

    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, retrieved, y


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 70
    dim = 64
    model = RetrievalTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        max_len=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(40):
        tokens, retrieved, labels = make_retrieval_batch(vocab_size=vocab_size, dim=dim)
        logits = model(tokens, retrieved)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} retrieval_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

