"""Variable-width CIFAR-10 ResNet32 used for channel-configuration search.

Important: the topology is still ResNet32.
- depth = 6n + 2 = 32
- n = 5 residual blocks per stage
- only the stage channel numbers are changed

This is a stage-level structured channel compression model. It is intentionally
simple and stable for coursework experiments before moving to true channel-index
pruning based on BN-gamma or weight norms.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1 and in_planes == planes:
            self.shortcut: nn.Module = nn.Identity()
        else:
            # CIFAR ResNet option A: downsample by slicing and pad channels.
            # This remains parameter-free, matching the baseline implementation.
            def option_a(x: torch.Tensor) -> torch.Tensor:
                out = x[:, :, ::stride, ::stride]
                channel_pad = planes - in_planes
                if channel_pad > 0:
                    left = channel_pad // 2
                    right = channel_pad - left
                    out = F.pad(out, (0, 0, 0, 0, left, right), "constant", 0)
                elif channel_pad < 0:
                    # This branch is not expected for the default monotonic search space,
                    # but keeps the module robust for custom channel settings.
                    keep = planes
                    out = out[:, :keep, :, :]
                return out

            self.shortcut = LambdaLayer(option_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class WidthResNet32(nn.Module):
    """CIFAR-10 ResNet32 with searchable stage channels.

    Args:
        stage_channels: three integers for stage1/stage2/stage3, e.g. [16, 24, 48].
        num_classes: number of output classes.
    """

    def __init__(self, stage_channels: Sequence[int] = (16, 32, 64), num_classes: int = 10) -> None:
        super().__init__()
        if len(stage_channels) != 3:
            raise ValueError("stage_channels must contain exactly 3 integers, e.g. [16, 32, 64].")
        c1, c2, c3 = [int(c) for c in stage_channels]
        if min(c1, c2, c3) <= 0:
            raise ValueError("all stage channels must be positive.")

        self.stage_channels = [c1, c2, c3]
        self.blocks_per_stage = 5
        self.in_planes = c1

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.stage1 = self._make_layer(c1, num_blocks=5, stride=1)
        self.stage2 = self._make_layer(c2, num_blocks=5, stride=2)
        self.stage3 = self._make_layer(c3, num_blocks=5, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
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
        out = self.fc(out)
        return out


def width_resnet32(stage_channels: Sequence[int] = (16, 32, 64), num_classes: int = 10) -> WidthResNet32:
    return WidthResNet32(stage_channels=stage_channels, num_classes=num_classes)
