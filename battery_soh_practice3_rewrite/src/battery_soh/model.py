from __future__ import annotations

import torch
from torch import nn


class SOHRegressor(nn.Module):
    """用于表格数据 SOH 回归的小型 MLP 模型。

    结构：Input → Linear → ReLU → Dropout → Linear → ReLU → Linear → Output
    隐藏层默认 64→32，输出为单个 SOH 百分比（回归）。
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.05):
        """初始化 MLP 回归器。

        Args:
            input_dim: 输入特征维度（即 Top-K 特征数量）。
            hidden_dims: 两个隐藏层神经元数，默认 (64, 32)。
            dropout: Dropout 概率，默认 0.05。
        """
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
        """前向传播，返回 shape 为 (batch,) 的 SOH 预测值（已 squeeze 最后一维）。"""
        return self.network(x).squeeze(-1)

