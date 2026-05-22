"""Train diffusion_transformer.py on tiny synthetic denoising data.

Run:
    python -m train.train_diffusion_transformer
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.diffusion_transformer import DiffusionTransformer


def make_clean_images(batch_size: int = 8, image_size: int = 32) -> torch.Tensor:
    """Create clean images with simple bright squares."""

    images = torch.zeros(batch_size, 3, image_size, image_size)
    for i in range(batch_size):
        if i % 2 == 0:
            images[i, :, 6:14, 6:14] = 1.0
        else:
            images[i, :, 18:26, 18:26] = 1.0
    return images


def main() -> None:
    torch.manual_seed(0)
    model = DiffusionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(30):
        clean = make_clean_images()
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, 1000, (clean.size(0),))
        noise_strength = timesteps.float().view(-1, 1, 1, 1) / 1000.0
        noised = clean + noise_strength * noise

        predicted_noise = model(noised, timesteps)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} diffusion_transformer_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

