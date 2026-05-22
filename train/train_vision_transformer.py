"""Train vision_transformer.py on tiny synthetic images.

Run:
    python -m train.train_vision_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.vision_transformer import VisionTransformer


def make_image_batch(batch_size: int = 16, image_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Create simple two-class images.

    Class 0 has a bright square in the upper-left.
    Class 1 has a bright square in the lower-right.
    """

    images = torch.randn(batch_size, 3, image_size, image_size) * 0.05
    labels = torch.randint(0, 2, (batch_size,))
    for i, label in enumerate(labels):
        if label.item() == 0:
            images[i, :, 4:12, 4:12] += 1.0
        else:
            images[i, :, 20:28, 20:28] += 1.0
    return images, labels


def main() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=2,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(40):
        images, labels = make_image_batch()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} vision_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

