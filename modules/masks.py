"""Mask helpers used by different Transformer variants.

In PyTorch attention, a mask is usually broadcast to:
    (batch, heads, query_length, key_length)

This project uses boolean masks where True means "can attend" and False means
"must be hidden". The attention module converts False positions to -inf.
"""

from __future__ import annotations

import torch


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a lower-triangular mask for autoregressive decoding.

    Shape:
        (1, 1, seq_len, seq_len)
    """

    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def padding_mask(tokens: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Create a mask that hides padding tokens.

    Args:
        tokens: token ids with shape (batch, seq_len)
        pad_id: id used for padding

    Returns:
        Boolean mask with shape (batch, 1, 1, seq_len)
    """

    return (tokens != pad_id).unsqueeze(1).unsqueeze(2)


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Combine masks with logical AND while ignoring None values."""

    valid_masks = [mask for mask in masks if mask is not None]
    if not valid_masks:
        return None

    combined = valid_masks[0]
    for mask in valid_masks[1:]:
        combined = combined & mask
    return combined

