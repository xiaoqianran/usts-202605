#!/usr/bin/env python3
"""
CIFAR-10 模型评估脚本
用于加载可变宽度 ResNet32 的检查点，在 CIFAR-10 测试集上评估模型性能，
并统计参数量与 FLOPs。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_cifar10_test_loader
from src.models import width_resnet32
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops


def parse_channels(s: str) -> list[int]:
    """
    解析通道数参数字符串，支持逗号和短横线分隔。
    例如: "16,24,48" 或 "16-24-48" 都将返回 [16, 24, 48]。
    必须恰好包含 3 个整数，分别对应三个阶段的通道宽度。
    """
    # 将短横线替换为逗号，再按逗号分割并转换为整数列表
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    # 验证必须是 3 个阶段的通道数
    if len(values) != 3:
        raise argparse.ArgumentTypeError("channels must contain 3 integers, e.g. 16,24,48")
    return values


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    """
    在测试集上评估模型性能。
    返回: (平均损失, Top-1 准确率百分比)
    """
    # 切换到评估模式（关闭 dropout、batch norm 的训练行为等）
    model.eval()
    losses = AverageMeter()  # 损失统计器
    top1 = AverageMeter()    # Top-1 准确率统计器

    # 遍历测试数据批次
    for images, targets in tqdm(loader, desc="eval"):
        # 将数据转移到目标设备（GPU/CPU），non_blocking 允许异步传输以加速
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 前向传播，获取模型输出
        outputs = model(images)
        # 计算交叉熵损失
        loss = criterion(outputs, targets)
        # 计算 Top-1 准确率（返回的是元组，取第一个元素）
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        # 按当前批次大小加权更新统计值
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))

    # 返回整个测试集上的平均损失和平均准确率
    return losses.avg, top1.avg


def main() -> None:
    """主函数：解析参数 → 加载模型 → 评估 → 输出结果"""

    # ========================
    # 1. 命令行参数解析
    # ========================
    parser = argparse.ArgumentParser(
        description="Evaluate variable-width ResNet32 checkpoint on CIFAR-10"
    )
    # 必需参数：检查点文件路径
    parser.add_argument("--checkpoint", required=True, help="模型检查点文件路径")
    # 可选参数：三阶段通道宽度；若省略则从检查点的 config 中读取
    parser.add_argument("--channels", type=parse_channels, default=None,
                        help="optional; if omitted, read from checkpoint config")
    # 数据集存放目录
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 数据集目录")
    # 测试时的批大小
    parser.add_argument("--batch-size", type=int, default=256, help="测试批大小")
    # 数据加载的工作线程数
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    # 限制测试样本数（0 表示使用全部测试集）
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="限制测试样本数，0 表示使用全部")
    # 随机种子（保证可复现性）
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # 计算设备（有 GPU 默认用 cuda，否则用 cpu）
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    # 结果输出的 JSON 文件路径（可选）
    parser.add_argument("--output", default=None, help="结果输出 JSON 文件路径")

    args = parser.parse_args()

    # ========================
    # 2. 设备选择与检查点加载
    # ========================
    # 确保设备有效：若指定 cuda 但不可用则回退到 cpu
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # 从磁盘加载检查点到 CPU（避免 GPU 内存占用）
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    # 通道配置优先级：命令行参数 > 检查点中保存的配置
    channels = args.channels or ckpt.get("config", {}).get("stage_channels")
    if channels is None:
        raise ValueError("channels not provided and not found in checkpoint config")

    # ========================
    # 3. 模型构建与权重加载
    # ========================
    # 根据通道配置构建可变宽度 ResNet32（10 类分类任务）
    model = width_resnet32(stage_channels=channels, num_classes=10)
    # 将检查点中保存的模型权重加载到模型中
    model.load_state_dict(ckpt["model"])
    # 将模型迁移到目标设备
    model = model.to(device)

    # ========================
    # 4. 测试数据加载
    # ========================
    loader = get_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    # 损失函数：交叉熵（标准分类损失）
    criterion = nn.CrossEntropyLoss()

    # ========================
    # 5. 执行评估
    # ========================
    loss, acc = evaluate(model, loader, criterion, device)

    # ========================
    # 6. 统计模型信息
    # ========================
    # 计算可训练参数总量
    params = count_parameters(model)
    # 以 CIFAR-10 的 32×32 RGB 图像为输入，计算浮点运算量（FLOPs）
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)

    # ========================
    # 7. 汇总结果
    # ========================
    result = {
        "checkpoint": str(args.checkpoint),       # 检查点路径
        "model": "WidthResNet32",                  # 模型名称
        "dataset": "CIFAR-10",                     # 数据集名称
        "stage_channels": list(channels),          # 三阶段通道配置
        "test_loss": loss,                         # 测试集损失
        "test_acc": acc,                           # 测试集 Top-1 准确率 (%)
        "params": params,                          # 参数量
        "flops": flops,                            # FLOPs
    }

    # 在终端打印格式化的 JSON 结果和摘要信息
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Params: {human_number(params)} | FLOPs: {human_number(flops)} | Acc: {acc:.2f}%")

    # ========================
    # 8. 可选：将结果写入文件
    # ========================
    if args.output:
        # 自动创建输出目录（含父目录）
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
