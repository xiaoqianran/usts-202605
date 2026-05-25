"""Block-level variable-width CIFAR-10 ResNet32.

The depth is still ResNet32:
    conv1 + 15 residual blocks + global average pooling + fc

The search vector has 15 dimensions:
    [c1_1, ..., c1_5, c2_1, ..., c2_5, c3_1, ..., c3_5]

This is a structured channel-width search. It changes block output channels but
keeps the number of residual blocks unchanged.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


BASELINE_BLOCK_CHANNELS = [16] * 5 + [32] * 5 + [64] * 5


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class FlexibleShortcut(nn.Module):
    """Parameter-free residual alignment for CIFAR ResNet.

    It handles both spatial downsampling and channel mismatch by strided slicing,
    zero-padding, or channel truncation. This keeps the baseline block-channel
    model parameter-equivalent to the original option-A CIFAR ResNet32.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x[:, :, :: self.stride, :: self.stride]
        diff = self.out_planes - self.in_planes
        if diff > 0:
            left = diff // 2
            right = diff - left
            out = F.pad(out, (0, 0, 0, 0, left, right), "constant", 0)
        elif diff < 0:
            out = out[:, : self.out_planes, :, :]
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 1 and in_planes == planes:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = FlexibleShortcut(in_planes, planes, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class BlockWidthResNet32(nn.Module):
    def __init__(self, block_channels: Sequence[int] = BASELINE_BLOCK_CHANNELS, num_classes: int = 10) -> None:
        super().__init__()
        if len(block_channels) != 15:
            raise ValueError("block_channels must contain 15 integers: 5 blocks per stage x 3 stages.")
        channels = [int(c) for c in block_channels]
        if min(channels) <= 0:
            raise ValueError("all block channels must be positive.")
        self.block_channels = channels
        self.blocks_per_stage = 5

        self.in_planes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        self.stage1 = self._make_stage(channels[0:5], first_stride=1)
        self.stage2 = self._make_stage(channels[5:10], first_stride=2)
        self.stage3 = self._make_stage(channels[10:15], first_stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)
        self._init_weights()

    def _make_stage(self, channels: Sequence[int], first_stride: int) -> nn.Sequential:
        layers = []
        for i, out_planes in enumerate(channels):
            stride = first_stride if i == 0 else 1
            layers.append(BasicBlock(self.in_planes, int(out_planes), stride=stride))
            self.in_planes = int(out_planes) * BasicBlock.expansion
        return nn.Sequential(*layers)

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
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def block_width_resnet32(block_channels: Sequence[int] = BASELINE_BLOCK_CHANNELS, num_classes: int = 10) -> BlockWidthResNet32:
    return BlockWidthResNet32(block_channels=block_channels, num_classes=num_classes)
