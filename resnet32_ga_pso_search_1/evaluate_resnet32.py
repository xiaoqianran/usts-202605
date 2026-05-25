#!/usr/bin/env python3
"""评估标准 ResNet32 在 CIFAR-10 测试集上的性能。

加载训练好的 ResNet32 checkpoint，在 CIFAR-10 测试集上计算损失与准确率，
同时统计模型的参数量和推理 FLOPs，并将结果输出为 JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from src.models import resnet32                          # 自定义 ResNet32 模型
from src.utils.checkpoint import load_checkpoint          # checkpoint 加载工具
from src.utils.metrics import (
    AverageMeter,       # 滑动平均计量器
    accuracy,           # top-k 准确率计算
    count_parameters,   # 统计可训练参数数量
    human_number,       # 将大数值格式化为可读字符串（如 1.2M）
    measure_flops,      # 测量模型 FLOPs
)


def get_test_loader(data_dir: str, batch_size: int, num_workers: int):
    """构建 CIFAR-10 测试集的 DataLoader。

    使用 CIFAR-10 官方推荐的均值和标准差进行归一化。

    Args:
        data_dir:    数据集存放根目录（不存在时自动下载）。
        batch_size:  每个 mini-batch 的样本数。
        num_workers: 数据加载的子进程数。

    Returns:
        配置好的 torch.utils.data.DataLoader。
    """
    # CIFAR-10 三个通道的均值和标准差（官方推荐）
    # 用于 图像归一化（Normalization）：
    mean = [0.4914, 0.4822, 0.4465]   # R, G, B 三个通道的均值
    std  = [0.2023, 0.1994, 0.2010]   # R, G, B 三个通道的标准差

    # 测试集只需 ToTensor + Normalize，不做数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,                              # 测试集不需要打乱顺序
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),       # GPU 可用时启用锁页内存以加速传输
    )


@torch.no_grad()  # 评估阶段关闭梯度计算，节省显存与计算
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集上评估模型的损失和 Top-1 准确率。

    Args:
        model:     待评估的模型。
        loader:    测试集 DataLoader。
        criterion: 损失函数（通常为 CrossEntropyLoss）。
        device:    计算设备。

    Returns:
        (平均损失, Top-1 准确率 %) 的元组。
    """
    model.eval()                    # 切换到评估模式（关闭 Dropout / 更新 BN 统计量等）
    losses = AverageMeter()         # 累计并平均损失
    top1 = AverageMeter()           # 累计并平均 Top-1 准确率

    for images, targets in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)       # non_blocking 与 pin_memory 配合加速
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)                # 前向传播
        loss = criterion(outputs, targets)     # 计算损失
        acc1 = accuracy(outputs, targets, topk=(1,))[0]  # 计算 Top-1 准确率

        losses.update(float(loss.item()), images.size(0))  # 按当前 batch 大小加权更新
        top1.update(float(acc1.item()), images.size(0))

    return losses.avg, top1.avg


def main() -> None:
    """程序入口：解析参数 → 加载模型 → 评估 → 统计参数/FLOPs → 输出结果。"""

    # ── 命令行参数解析 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Evaluate standard ResNet32 checkpoint on CIFAR-10"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="模型 checkpoint 文件路径（.pth）",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="CIFAR-10 数据集存放目录",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="评估时的 batch size",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="数据加载的 worker 数量",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（cuda / cpu）",
    )
    parser.add_argument(
        "--output", default=None,
        help="（可选）评估结果的 JSON 输出路径",
    )
    args = parser.parse_args()

    # ── 设备选择 ────────────────────────────────────────────────────
    # 若用户指定 cuda 但环境不可用，自动回退到 cpu
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # ── 加载 checkpoint ─────────────────────────────────────────────
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")  # 先加载到 CPU 再迁移到目标设备

    # ── 构建模型并加载权重 ──────────────────────────────────────────
    model = resnet32(num_classes=10)            # CIFAR-10 共 10 个类别
    model.load_state_dict(ckpt["model"])        # 从 checkpoint 中恢复模型参数
    model = model.to(device)                    # 迁移到目标设备

    # ── 数据加载 ────────────────────────────────────────────────────
    loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)

    # ── 评估 ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device)

    # ── 统计模型复杂度 ──────────────────────────────────────────────
    params = count_parameters(model)                            # 可训练参数总量
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)  # 单样本推理 FLOPs

    # ── 汇总结果 ────────────────────────────────────────────────────
    result = {
        "checkpoint": str(args.checkpoint),
        "model": "ResNet32",
        "dataset": "CIFAR-10",
        "test_loss": loss,
        "test_acc": acc,            # 百分比形式，如 93.52
        "params": params,
        "flops": flops,
    }

    # 打印到终端（JSON + 一行摘要）
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Params: {human_number(params)} | FLOPs: {human_number(flops)} | Acc: {acc:.2f}%")

    # ── 写入文件（可选） ────────────────────────────────────────────
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)  # 自动创建输出目录
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
