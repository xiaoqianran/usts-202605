"""标准 CIFAR-10 ResNet32 实现。

本文件仅包含 ResNet32，不包含宽度缩放、压缩变体或其他深度变体（ResNet20/44/56/110）。

CIFAR ResNet 深度规则：depth = 6n + 2。
对于 ResNet32，n = 5，因此三个阶段各有 5 个残差块。

网络架构：
    conv1: 3×3 卷积，16 个通道
    stage1: 5 个残差块，16 个通道
    stage2: 5 个残差块，32 个通道，第一个块 stride=2（下采样）
    stage3: 5 个残差块，64 个通道，第一个块 stride=2（下采样）
    全局平均池化
    fc: 全连接层，输出 10 个类别
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """Lambda 包装层：将任意可调用对象封装为 nn.Module，便于在 nn.Sequential 中使用。"""

    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    """ResNet 基本残差块（适用于 CIFAR 系列，expansion=1）。

    结构：
        conv1 (3×3) → BN → ReLU → conv2 (3×3) → BN → + shortcut → ReLU

    当 stride=1 且输入输出通道数相同时，shortcut 为恒等映射；
    否则使用原始 CIFAR ResNet 的 Option A：空间维度通过步长切片降采样，
    通道维度通过零填充对齐，不引入额外可训练参数。
    """

    expansion = 1  # 每个块的输出通道数 = planes × expansion

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        # 第一层卷积：可能执行空间下采样（stride > 1）
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二层卷积：保持空间尺寸不变
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
            # 维度完全匹配，直接使用恒等映射
            self.shortcut: nn.Module = nn.Identity()
        else:
            # 原始 CIFAR ResNet Option A 快捷连接：
            # 1) 通过步长切片对特征图进行空间下采样
            # 2) 通过零填充对齐通道数差异
            # 不引入任何可训练参数
            def option_a(x: torch.Tensor) -> torch.Tensor:
                # 空间下采样：每隔 stride 个像素取样
                out = x[:, :, ::stride, ::stride]
                # 通道维度零填充
                channel_pad = planes - in_planes
                if channel_pad > 0:
                    left = channel_pad // 2
                    right = channel_pad - left
                    # F.pad 参数格式：(W左, W右, H上, H下, C前, C后)
                    out = F.pad(out, (0, 0, 0, 0, left, right), "constant", 0)
                return out

            self.shortcut = LambdaLayer(option_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径：conv1 → BN → ReLU → conv2 → BN
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # 残差连接：主路径输出 + shortcut 输出
        out = out + self.shortcut(x)
        # 残差连接后再过 ReLU
        out = F.relu(out, inplace=True)
        return out


class ResNet32(nn.Module):
    """ResNet32 网络，专用于 CIFAR-10（32×32 彩色图像，10 个类别）。

    整体流程：
        输入 (B, 3, 32, 32)
        → conv1 (3×3, 16通道) + BN + ReLU
        → layer1 (5×BasicBlock, 16通道, stride=1)
        → layer2 (5×BasicBlock, 32通道, 首块 stride=2 下采样)
        → layer3 (5×BasicBlock, 64通道, 首块 stride=2 下采样)
        → 全局平均池化 → 展平 → 全连接层 → 输出 (B, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # in_planes 跟踪当前通道数，_make_layer 中会更新
        self.in_planes = 16

        # 初始卷积层：CIFAR ResNet 使用 3×3 卷积（而非 ImageNet ResNet 的 7×7）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 三个残差阶段
        self.layer1 = self._make_layer(planes=16, num_blocks=5, stride=1)   # 32×32 → 32×32
        self.layer2 = self._make_layer(planes=32, num_blocks=5, stride=2)   # 32×32 → 16×16
        self.layer3 = self._make_layer(planes=64, num_blocks=5, stride=2)   # 16×16 → 8×8
        # 自适应全局平均池化：将任意空间尺寸压缩为 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类全连接层
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """构建一个残差阶段。

        Args:
            planes: 该阶段中每个 BasicBlock 的输出通道数。
            num_blocks: 该阶段包含的 BasicBlock 数量（ResNet32 每阶段 5 个）。
            stride: 第一个 BasicBlock 的步长（用于空间下采样），后续块均为 stride=1。

        Returns:
            包含 num_blocks 个 BasicBlock 的 nn.Sequential。
        """
        # 仅第一个块使用指定 stride，其余块 stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, current_stride))
            # 更新 in_planes 为当前块的输出通道数
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """权重初始化。

        - Conv2d: Kaiming 正态初始化（fan_out 模式，适配 ReLU）
        - BatchNorm2d: 权重初始化为 1，偏置初始化为 0
        - Linear: 权重使用均值 0、标准差 0.01 的正态分布，偏置初始化为 0
        """
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
        """前向传播。

        Args:
            x: 输入张量，形状 (B, 3, 32, 32)。

        Returns:
            logits: 输出张量，形状 (B, num_classes)，未经 softmax。
        """
        # 初始特征提取
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # 三个残差阶段
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 全局平均池化：(B, 64, H, W) → (B, 64, 1, 1)
        out = self.avgpool(out)
        # 展平：(B, 64, 1, 1) → (B, 64)
        out = torch.flatten(out, 1)
        # 分类层：(B, 64) → (B, 10)
        out = self.fc(out)
        return out


def resnet32(num_classes: int = 10) -> ResNet32:
    """构造并返回一个 ResNet32 模型实例。

    Args:
        num_classes: 分类类别数，默认为 10（CIFAR-10）。

    Returns:
        ResNet32 模型实例。
    """
    return ResNet32(num_classes=num_classes)
