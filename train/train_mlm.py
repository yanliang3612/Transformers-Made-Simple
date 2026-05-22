"""Train a tiny BERT-like model with masked language modeling."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from configs.bert_config import BertConfig
from models.encoder_only import EncoderOnlyTransformer
from modules.masks import padding_mask


def mask_tokens(x: torch.Tensor, mask_id: int, mask_prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    labels = x.clone()
    mask = torch.rand_like(x.float()) < mask_prob
    x = x.clone()
    x[mask] = mask_id
    labels[~mask] = -100
    return x, labels


def main() -> None:
    cfg = BertConfig(vocab_size=200, dim=128, num_layers=2, num_heads=4, mlp_dim=512, mask_id=199)
    model = EncoderOnlyTransformer(
        cfg.vocab_size,
        cfg.dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.mlp_dim,
        cfg.max_len,
        cfg.num_classes,
        cfg.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(30):
        clean = torch.randint(1, cfg.vocab_size - 1, (16, 32))
        masked, labels = mask_tokens(clean, cfg.mask_id)
        outputs = model(masked, padding_mask(masked, cfg.pad_id))
        loss = F.cross_entropy(outputs["mlm_logits"].view(-1, cfg.vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step={step} mlm_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

