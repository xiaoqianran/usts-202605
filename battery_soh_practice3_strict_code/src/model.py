from __future__ import annotations

import torch
from torch import nn


class SOHMLP(nn.Module):
    """MLP regression model for tabular SOH prediction."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
