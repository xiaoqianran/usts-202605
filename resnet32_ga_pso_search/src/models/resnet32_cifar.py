"""Standard CIFAR-10 ResNet32.

This file intentionally contains only ResNet32.
No width scaling, no compressed variants, no ResNet20/44/56/110.

CIFAR ResNet depth rule: depth = 6n + 2.
For ResNet32, n = 5, so each of the three stages has 5 residual blocks.

Architecture:
    conv1: 3x3, 16 channels
    stage1: 5 blocks, 16 channels
    stage2: 5 blocks, 32 channels, first block stride=2
    stage3: 5 blocks, 64 channels, first block stride=2
    global average pooling
    fc: 10 classes
"""
from __future__ import annotations

from typing import Callable

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
            # Original CIFAR ResNet option A shortcut:
            # downsample spatial size by strided slicing and pad channels with zeros.
            # This adds no trainable parameters.
            def option_a(x: torch.Tensor) -> torch.Tensor:
                out = x[:, :, ::stride, ::stride]
                channel_pad = planes - in_planes
                if channel_pad > 0:
                    left = channel_pad // 2
                    right = channel_pad - left
                    out = F.pad(out, (0, 0, 0, 0, left, right), "constant", 0)
                return out

            self.shortcut = LambdaLayer(option_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(planes=16, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(planes=32, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(planes=64, num_blocks=5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet32(num_classes: int = 10) -> ResNet32:
    return ResNet32(num_classes=num_classes)
