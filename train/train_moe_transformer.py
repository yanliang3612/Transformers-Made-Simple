"""Train moe_transformer.py on a tiny synthetic text classification task.

Run:
    python -m train.train_moe_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.moe_transformer import MoETransformer
from modules.masks import padding_mask


def make_expert_batch(batch_size: int = 16, seq_len: int = 16, vocab_size: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Create data with two token patterns.

    The MoE layer can learn to route different token patterns to different experts.
    """

    tokens = torch.randint(6, vocab_size, (batch_size, seq_len))
    tokens[:, 0] = 1
    labels = torch.randint(0, 2, (batch_size,))
    for i, label in enumerate(labels):
        if label.item() == 0:
            tokens[i, 3:7] = torch.tensor([6, 7, 6, 7])
        else:
            tokens[i, 3:7] = torch.tensor([8, 9, 8, 9])
    return tokens, labels


def router_balance_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Small auxiliary loss so routing does not collapse to one expert.

    router_logits shape:
        (num_layers, batch, seq_len, num_experts)
    """

    probs = router_logits.softmax(dim=-1)
    mean_probs = probs.mean(dim=(0, 1, 2))
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    return F.mse_loss(mean_probs, uniform)


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 100
    model = MoETransformer(
        vocab_size=vocab_size,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        num_experts=4,
        max_len=32,
        num_classes=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(40):
        tokens, labels = make_expert_batch(vocab_size=vocab_size)
        outputs = model(tokens, padding_mask(tokens, pad_id=0))
        task_loss = F.cross_entropy(outputs["logits"], labels)
        loss = task_loss + 0.01 * router_balance_loss(outputs["router_logits"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} moe_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

