"""Train a tiny DiT-style denoiser on random image noise."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from configs.dit_config import DiTConfig
from models.diffusion_transformer import DiffusionTransformer


def main() -> None:
    cfg = DiTConfig(dim=128, num_layers=2, num_heads=4, mlp_dim=512)
    model = DiffusionTransformer(
        cfg.image_size,
        cfg.patch_size,
        cfg.in_channels,
        cfg.dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.mlp_dim,
        cfg.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(20):
        images = torch.randn(8, cfg.in_channels, cfg.image_size, cfg.image_size)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, cfg.num_diffusion_steps, (8,))
        noised = images + 0.1 * noise

        predicted_noise = model(noised, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"step={step} dit_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

