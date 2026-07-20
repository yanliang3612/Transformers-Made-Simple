"""Feed-forward networks used inside Transformer blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """Standard Transformer MLP.

    Each token is processed independently:
        dim -> hidden_dim -> dim
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        x = self.dropout(x)
        return self.fc2(x)


class GatedFeedForward(nn.Module):
    """SwiGLU-style gated feed-forward layer used in many modern LLMs."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        return self.down(self.dropout(x))


class MoEFeedForward(nn.Module):
    """Tiny Mixture-of-Experts feed-forward layer.

    This is intentionally simple: each token chooses the top-1 expert. Production
    MoE models add load balancing losses and distributed routing.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, dim)
        router_logits = self.gate(x)
        expert_ids = router_logits.argmax(dim=-1)

        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            token_mask = expert_ids == expert_id
            if token_mask.any():
                output[token_mask] = expert(x[token_mask])

        return output, router_logits

