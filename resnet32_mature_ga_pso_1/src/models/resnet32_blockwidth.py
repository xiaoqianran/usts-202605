"""Block-level variable-width ResNet32.

This model keeps the ResNet32 depth fixed but searches the output channels of
15 residual blocks:
    stage1: 5 blocks
    stage2: 5 blocks
    stage3: 5 blocks

When block input/output dimensions differ, a 1x1 projection shortcut is used.
This makes arbitrary block-level channel configurations feasible and stable.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BLOCK_CHANNELS = [16] * 5 + [32] * 5 + [64] * 5


class ProjectionBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        if stride == 1 and in_planes == out_planes:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class BlockWidthResNet32(nn.Module):
    def __init__(self, block_channels: Sequence[int] | None = None, num_classes: int = 10) -> None:
        super().__init__()
        if block_channels is None:
            block_channels = DEFAULT_BLOCK_CHANNELS
        if len(block_channels) != 15:
            raise ValueError("block_channels must contain 15 integers: 5 per stage.")
        if min(int(c) for c in block_channels) <= 0:
            raise ValueError("all channels must be positive.")

        self.block_channels = [int(c) for c in block_channels]
        self.blocks_per_stage = 5

        in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        blocks = []
        for i, out_planes in enumerate(self.block_channels):
            stride = 2 if i in (5, 10) else 1
            blocks.append(ProjectionBasicBlock(in_planes, out_planes, stride=stride))
            in_planes = out_planes
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.blocks(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def block_width_resnet32(block_channels: Sequence[int] | None = None, num_classes: int = 10) -> BlockWidthResNet32:
    return BlockWidthResNet32(block_channels=block_channels, num_classes=num_classes)
