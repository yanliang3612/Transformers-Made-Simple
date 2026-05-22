"""Sampling utilities for decoder-only language models."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k largest logits and set the rest to -inf."""

    if k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """Autoregressively generate tokens from a decoder-only model."""

    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :] / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids

