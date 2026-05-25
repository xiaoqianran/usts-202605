"""训练与评估工具函数。

包含：
    - AverageMeter  : 运行时指标（损失、准确率等）的滑动平均追踪器
    - accuracy      : Top-K 准确率计算
    - count_parameters : 模型参数量统计
    - human_number  : 数值可读化格式输出
    - measure_flops : 基于前向钩子的近似 FLOPs 估算
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AverageMeter:
    """滑动平均指标追踪器。

    用于在训练/验证循环中跟踪损失、准确率等标量指标。
    支持按批次动态更新，并自动维护加权平均值。

    用法示例：
        meter = AverageMeter()
        for x, y in loader:
            loss = criterion(model(x), y)
            meter.update(loss.item(), n=x.size(0))
        print(f"平均损失: {meter.avg:.4f}")
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """重置所有统计量。通常在每个 epoch 开始时调用。"""
        self.val = 0.0    # 最近一次更新的值
        self.avg = 0.0    # 加权平均值
        self.sum = 0.0    # 所有值的加权总和
        self.count = 0    # 累计样本数

    def update(self, val: float, n: int = 1) -> None:
        """更新统计量。

        Args:
            val: 当前批次的指标值。
            n: 当前批次的样本数（用于加权平均，默认为 1）。
        """
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """计算 Top-K 准确率。

    Args:
        output: 模型输出 logits，形状 (B, C)，B 为批次大小，C 为类别数。
        target: 真实标签，形状 (B,)，每个元素为类别索引。
        topk: 需要计算的 Top-K 元组，例如 (1, 5) 表示同时计算 Top-1 和 Top-5。

    Returns:
        准确率列表（百分比），与 topk 中的 K 一一对应。
        例如 topk=(1, 5) 时返回 [top1_acc, top5_acc]。
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # 取出每个样本得分最高的 maxk 个预测类别
    # _: 未使用的值，pred: 对应的类别索引，形状 (B, maxk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    # 转置为 (maxk, B)，便于按 K 逐行切片
    pred = pred.t()
    # 与真实标签比较，correct 形状 (maxk, B)，True/False
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    results = []
    for k in topk:
        # 取前 k 行，展平后求和，得到 Top-K 正确预测数
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # 转换为百分比准确率
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def count_parameters(model: nn.Module) -> int:
    """统计模型的总参数量（所有可训练参数的元素总数）。

    Args:
        model: PyTorch 模型。

    Returns:
        参数总数（标量）。
    """
    return sum(p.numel() for p in model.parameters())


def human_number(n: float) -> str:
    """将大数值格式化为人类可读的字符串。

    规则：
        >= 1e9  → 以 G（十亿）为单位
        >= 1e6  → 以 M（百万）为单位
        >= 1e3  → 以 K（千）为单位
        < 1e3   → 原样输出整数

    Args:
        n: 待格式化的数值。

    Returns:
        格式化后的字符串，例如 "1.500M"、"470.000K"。
    """
    if abs(n) >= 1e9:
        return f"{n / 1e9:.3f}G"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.3f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.3f}K"
    return f"{n:.0f}"


@torch.no_grad()
def measure_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 32, 32),
    device: str | torch.device = "cpu",
) -> int:
    """估算模型单张图片的近似乘加 FLOPs。

    仅统计 Conv2d 和 Linear 层的计算量。
    BatchNorm、ReLU、残差加法等忽略不计——对于课程实验的基线对比已经足够。

    原理：
        1) 为每个 Conv2d / Linear 模块注册前向钩子（forward hook）
        2) 用一张随机虚拟输入执行前向传播
        3) 钩子在每次前向时根据输出尺寸和卷积核大小累加 FLOPs
        4) 前向结束后移除所有钩子，返回累计值

    Args:
        model: 待测量的 PyTorch 模型。
        input_size: 输入尺寸元组 (C, H, W)，不含批次维度，默认 CIFAR-10 的 3×32×32。
        device: 运算设备。

    Returns:
        单张图片的近似 FLOPs 总数（整数）。
    """
    model = model.to(device)
    model.eval()
    # 使用字典在闭包中共享状态，累加所有层的 FLOPs
    flops: Dict[str, int] = {"total": 0}
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        """Conv2d 的 FLOPs 计算钩子。

        计算公式：
            FLOPs = out_h × out_w × out_channels × (kH × kW × in_channels / groups)

        即：每个输出像素需要 kH×kW×(in_channels/groups) 次乘加，
        共有 out_h × out_w × out_channels 个输出像素。
        """
        out = output[0] if isinstance(output, (tuple, list)) else output
        batch_size = out.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        # 每个输出通道对应的乘加操作数：kH × kW × (in_channels / groups)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        # 总操作数（含批次），再除以 batch_size 归一化为单张图片
        total_ops = batch_size * out_h * out_w * module.out_channels * kernel_ops
        flops["total"] += int(total_ops / batch_size)

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        """Linear 的 FLOPs 计算钩子。

        计算公式：FLOPs = in_features × out_features
        （矩阵乘法的乘加次数）
        """
        batch_size = inputs[0].shape[0]
        total_ops = batch_size * module.in_features * module.out_features
        flops["total"] += int(total_ops / batch_size)

    # 遍历模型所有子模块，为卷积层和全连接层注册钩子
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    # 构造虚拟输入并执行一次前向传播，触发所有钩子
    dummy = torch.randn(1, *input_size, device=device)
    model(dummy)

    # 清理：移除所有已注册的钩子，避免内存泄漏和重复计数
    for hook in hooks:
        hook.remove()

    return flops["total"]
