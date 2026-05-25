"""可变宽度 CIFAR-10 ResNet32，用于通道配置搜索。

重要说明：网络拓扑仍然是 ResNet32。
- 深度 = 6n + 2 = 32
- 每个阶段 n = 5 个残差块
- 仅修改各阶段的通道数（宽度）

这是一种阶段级结构化通道压缩模型。设计上刻意保持简单和稳定，
适用于课程实验阶段，之后再过渡到基于 BN-gamma 或权重范数的真正通道索引剪枝。
"""
from __future__ import annotations

from typing import Callable, Sequence

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
    """ResNet 基本残差块（expansion=1），支持输入输出通道数不匹配的情况。

    结构：
        conv1 (3×3) → BN → ReLU → conv2 (3×3) → BN → + shortcut → ReLU

    当 stride=1 且 in_planes == planes 时，shortcut 为恒等映射；
    否则使用 CIFAR ResNet Option A 快捷连接：
        - 空间维度通过步长切片降采样
        - 通道维度通过零填充对齐（通道数增加时）或截断（通道数减少时）
        - 不引入任何可训练参数
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
            # CIFAR ResNet Option A 快捷连接：
            # 通过步长切片进行空间下采样，通过填充/截断对齐通道数。
            # 不引入额外参数，与基线实现保持一致。
            def option_a(x: torch.Tensor) -> torch.Tensor:
                # 空间下采样：每隔 stride 个像素取样
                out = x[:, :, ::stride, ::stride]
                channel_pad = planes - in_planes
                if channel_pad > 0:
                    # 通道数增加：在通道维度两端对称补零
                    left = channel_pad // 2
                    right = channel_pad - left
                    # F.pad 参数格式：(W左, W右, H上, H下, C前, C后)
                    out = F.pad(out, (0, 0, 0, 0, left, right), "constant", 0)
                elif channel_pad < 0:
                    # 通道数减少：截断多余的通道（仅保留前 planes 个通道）
                    # 在默认的单调递增搜索空间中不会触发此分支，
                    # 但保留此逻辑以支持自定义通道配置时的鲁棒性。
                    keep = planes
                    out = out[:, :keep, :, :]
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


class WidthResNet32(nn.Module):
    """可搜索阶段通道数的 CIFAR-10 ResNet32。

    与标准 ResNet32 的唯一区别在于：三个阶段的通道数不再固定为 (16, 32, 64)，
    而是通过 stage_channels 参数灵活指定，从而支持通道配置搜索实验。

    Args:
        stage_channels: 三个整数，分别对应 stage1/stage2/stage3 的通道数，
                        例如 [16, 24, 48]。
        num_classes: 输出类别数。
    """

    def __init__(self, stage_channels: Sequence[int] = (16, 32, 64), num_classes: int = 10) -> None:
        super().__init__()
        # 参数校验
        if len(stage_channels) != 3:
            raise ValueError("stage_channels 必须恰好包含 3 个整数，例如 [16, 32, 64]。")
        c1, c2, c3 = [int(c) for c in stage_channels]
        if min(c1, c2, c3) <= 0:
            raise ValueError("所有阶段通道数必须为正整数。")

        self.stage_channels = [c1, c2, c3]
        self.blocks_per_stage = 5  # 每个阶段的残差块数（ResNet32: n=5）
        # in_planes 跟踪当前通道数，_make_layer 中会更新
        self.in_planes = c1

        # 初始卷积层：3×3 卷积，输出通道数 = 第一阶段通道数 c1
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        # 三个残差阶段
        self.stage1 = self._make_layer(c1, num_blocks=5, stride=1)   # 32×32 → 32×32
        self.stage2 = self._make_layer(c2, num_blocks=5, stride=2)   # 32×32 → 16×16
        self.stage3 = self._make_layer(c3, num_blocks=5, stride=2)   # 16×16 → 8×8
        # 自适应全局平均池化：将任意空间尺寸压缩为 1×1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类全连接层：输入维度 = 第三阶段通道数 c3
        self.fc = nn.Linear(c3, num_classes)

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
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
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
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        # 全局平均池化：(B, c3, H, W) → (B, c3, 1, 1)
        out = self.avg_pool(out)
        # 展平：(B, c3, 1, 1) → (B, c3)
        out = torch.flatten(out, 1)
        # 分类层：(B, c3) → (B, num_classes)
        out = self.fc(out)
        return out


def width_resnet32(stage_channels: Sequence[int] = (16, 32, 64), num_classes: int = 10) -> WidthResNet32:
    """构造并返回一个可变宽度 ResNet32 模型实例。

    Args:
        stage_channels: 三个阶段的通道数，例如 (16, 32, 64)。
        num_classes: 分类类别数，默认为 10（CIFAR-10）。

    Returns:
        WidthResNet32 模型实例。
    """
    return WidthResNet32(stage_channels=stage_channels, num_classes=num_classes)
