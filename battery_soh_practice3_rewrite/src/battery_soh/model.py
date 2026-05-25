from __future__ import annotations

import torch
from torch import nn


class SOHRegressor(nn.Module):
    """Small MLP for tabular SOH regression."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.05):
        super().__init__()
        h1, h2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

