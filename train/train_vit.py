"""Train a tiny ViT on random images to demonstrate the training loop."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from configs.vit_config import ViTConfig
from models.vision_transformer import VisionTransformer


def main() -> None:
    cfg = ViTConfig(dim=128, num_layers=2, num_heads=4, mlp_dim=512)
    model = VisionTransformer(**cfg.__dict__)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(20):
        images = torch.randn(16, cfg.in_channels, cfg.image_size, cfg.image_size)
        labels = torch.randint(0, cfg.num_classes, (16,))
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"step={step} vit_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

