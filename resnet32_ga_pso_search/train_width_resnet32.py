#!/usr/bin/env python3
"""
在 CIFAR-10 上训练可变宽度 ResNet32 候选网络。

用法示例:
    python train.py --channels 16,24,48 --epochs 80 --amp
    python train.py --channels 32,48,96 --epochs 120 --resume runs/xxx/last.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 训练/评估循环中的进度条库

# ──────────────────────────── 项目内部模块 ────────────────────────────
from src.data import get_cifar10_loaders           # 获取 CIFAR-10 训练/测试 DataLoader
from src.models import width_resnet32               # 可变宽度 ResNet32 模型构建函数
from src.utils.checkpoint import load_checkpoint, save_checkpoint  # 检查点保存与加载
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed                 # 随机种子设置工具


# ══════════════════════════════════════════════════════════════════════
#  命令行参数解析 辅助函数
# ══════════════════════════════════════════════════════════════════════

def parse_channels(s: str) -> list[int]:
    """
    将字符串解析为长度为 3 的通道数列表。

    支持两种分隔格式:
        "16,24,48"  或  "16-24-48"
    分别对应 ResNet 三个阶段(stage)的输出通道数。
    """
    # 统一把 "-" 替换成 ","，再按 "," 分割并转为 int
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "channels must contain 3 integers, e.g. 16,24,48 or 16-24-48"
        )
    return values


def parse_milestones(s: str) -> list[int]:
    """
    将逗号分隔的字符串解析为学习率调度的里程碑(milestone)列表。
    例如 "40,60" → [40, 60]，在第 40 和第 60 个 epoch 降低学习率。
    空字符串返回空列表。
    """
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ══════════════════════════════════════════════════════════════════════
#  CSV 日志辅助函数
# ══════════════════════════════════════════════════════════════════════

def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    """
    如果 CSV 文件尚不存在，创建文件并写入表头。
    存在时不做任何操作，避免覆盖已有数据。
    """
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()


def append_csv(path: Path, row: dict) -> None:
    """向 CSV 文件末尾追加一行记录。"""
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════
#  核心训练与评估逻辑
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool) -> tuple[float, float]:
    """
    训练模型一个完整的 epoch。

    参数:
        model      : 待训练的神经网络模型
        loader     : 训练数据 DataLoader
        criterion  : 损失函数（CrossEntropyLoss）
        optimizer  : 优化器（SGD）
        device     : 计算设备（"cuda" 或 "cpu"）
        use_amp    : 是否启用混合精度训练（AMP）

    返回:
        (平均训练损失, 平均训练 Top-1 准确率)
    """
    model.train()  # 切换到训练模式（启用 Dropout / BatchNorm 的训练行为）

    losses = AverageMeter()  # 累计损失统计器
    top1 = AverageMeter()    # 累计 Top-1 准确率统计器

    # AMP 梯度缩放器：在混合精度下防止梯度下溢
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    pbar = tqdm(loader, desc="train", leave=False)  # 训练进度条
    for images, targets in pbar:
        # 将数据异步传输到目标设备，non_blocking=True 允许与计算重叠
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 清除上一步的梯度；set_to_none=True 将梯度设为 None（比 zero_ 更节省内存）
        optimizer.zero_grad(set_to_none=True)

        # ── 前向传播（混合精度上下文）──
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)          # 前向推理，得到 logits
            loss = criterion(outputs, targets)  # 计算交叉熵损失

        # ── 反向传播 + 参数更新 ──
        scaler.scale(loss).backward()   # 缩放损失后反向传播
        scaler.step(optimizer)          # 用缩放后的梯度更新参数
        scaler.update()                 # 更新缩放因子

        # 计算当前 batch 的 Top-1 准确率并更新统计
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))

        # 在进度条上实时显示当前平均 loss 和 acc
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}")

    return losses.avg, top1.avg


@torch.no_grad()  # 评估时禁止梯度计算，节省显存和加速
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """
    在测试集上评估模型性能。

    参数:
        model     : 待评估模型
        loader    : 测试数据 DataLoader
        criterion : 损失函数
        device    : 计算设备

    返回:
        (平均测试损失, 平均测试 Top-1 准确率)
    """
    model.eval()  # 切换到评估模式（冻结 BatchNorm / 关闭 Dropout）

    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(loader, desc="eval", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))

        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}")

    return losses.avg, top1.avg


# ══════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    # ────────────────── 1. 命令行参数定义 ──────────────────
    parser = argparse.ArgumentParser(
        description="Train a variable-width ResNet32 candidate on CIFAR-10"
    )
    parser.add_argument("--channels", type=parse_channels, default=[16, 24, 48],
                        help="三个阶段的通道数，如 16,24,48 或 16-24-48")
    parser.add_argument("--data-dir", default="data",
                        help="数据集存放目录（自动下载 CIFAR-10）")
    parser.add_argument("--save-dir", default="runs",
                        help="实验输出根目录")
    parser.add_argument("--run-name", default=None,
                        help="本次实验名称；默认根据通道数自动生成")
    parser.add_argument("--epochs", type=int, default=80,
                        help="总训练轮数")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="每张 GPU / CPU 的批大小")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="初始学习率")
    parser.add_argument("--milestones", type=parse_milestones, default=[40, 60],
                        help="学习率衰减里程碑，如 40,60")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="学习率衰减倍率（乘以 gamma）")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD 动量")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 权重衰减系数")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="数据加载的子进程数")
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="限制训练样本数（>0 时生效），用于快速实验")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="限制测试样本数（>0 时生效），用于快速实验")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证可复现性")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备（cuda / cpu）")
    parser.add_argument("--amp", action="store_true",
                        help="启用自动混合精度（AMP）训练")
    parser.add_argument("--resume", default=None,
                        help="从指定检查点路径恢复训练")
    args = parser.parse_args()

    # ────────────────── 2. 环境初始化 ──────────────────
    set_seed(args.seed, deterministic=False)  # 设置全局随机种子

    # 设备选择：优先 CUDA，不可用时回退到 CPU
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    # AMP 仅在 CUDA 设备上有效
    use_amp = bool(args.amp and device.type == "cuda")

    # ────────────────── 3. 输出目录与文件路径 ──────────────────
    channel_tag = "-".join(map(str, args.channels))  # 例: "16-24-48"
    run_name = args.run_name or f"width_resnet32_{channel_tag}"
    out_dir = Path(args.save_dir) / run_name          # 例: runs/width_resnet32_16-24-48/
    out_dir.mkdir(parents=True, exist_ok=True)         # 递归创建目录
    metrics_csv = out_dir / "metrics.csv"              # 每 epoch 指标记录文件

    # ────────────────── 4. 数据加载 ──────────────────
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,  # 0 表示使用全部数据
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    # ────────────────── 5. 模型构建与统计 ──────────────────
    model = width_resnet32(stage_channels=args.channels, num_classes=10).to(device)
    params = count_parameters(model)                              # 可训练参数总量
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)  # 前向 FLOPs

    print("Model: variable-width CIFAR-10 ResNet32")
    print(f"Stage channels: {channel_tag}")
    print("Blocks per stage: 5")
    print(f"Params: {params} ({human_number(params)})")
    print(f"FLOPs : {flops} ({human_number(flops)})")

    # ────────────────── 6. 损失函数 / 优化器 / 调度器 ──────────────────
    criterion = nn.CrossEntropyLoss()  # 标准交叉熵损失

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 多步学习率调度：在指定 epoch 乘以 gamma 衰减
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    # ────────────────── 7. 恢复训练（可选）──────────────────
    start_epoch = 0
    best_acc = 0.0      # 历史最佳测试准确率
    best_epoch = -1      # 达到最佳准确率的 epoch

    # 保存一份完整的训练配置，写入检查点以备后续查阅
    config = vars(args).copy()
    config["model"] = "WidthResNet32"
    config["dataset"] = "CIFAR-10"
    config["stage_channels"] = args.channels
    config["blocks_per_stage"] = 5

    if args.resume:
        # 从检查点恢复模型权重、优化器状态、调度器状态及训练进度
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1      # 从下一个 epoch 继续
        best_acc = float(ckpt.get("best_acc", 0.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ────────────────── 8. 训练循环 ──────────────────
    # CSV 表头：若文件不存在则创建
    fields = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"]
    write_csv_header_if_needed(metrics_csv, fields)

    total_start = time.time()  # 记录总训练开始时间

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 获取当前学习率（用于日志显示）
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={lr:.6f}")

        # ── 训练一个 epoch ──
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp
        )

        # ── 在测试集上评估 ──
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # ── 更新学习率调度器 ──
        scheduler.step()

        # 计算本 epoch 耗时
        epoch_time = time.time() - epoch_start

        # ── 记录指标到 CSV ──
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.4f}",
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.4f}",
            "epoch_time_sec": f"{epoch_time:.2f}",
        }
        append_csv(metrics_csv, row)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f} | time={epoch_time:.1f}s"
        )

        # ── 判断是否为历史最佳 ──
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch + 1

        # ── 保存检查点 ──
        # 每个 epoch 都保存 last.pt（覆盖写），用于断点续训
        # 仅在准确率刷新记录时保存 best.pt
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),           # 模型权重
            "optimizer": optimizer.state_dict(),   # 优化器状态（动量等）
            "scheduler": scheduler.state_dict(),   # 学习率调度器状态
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": config,                      # 训练超参数快照
            "params": params,
            "flops": flops,
        }
        save_checkpoint(out_dir / "last.pt", **payload)
        if is_best:
            save_checkpoint(out_dir / "best.pt", **payload)
            print(f"Saved new best checkpoint: acc={best_acc:.2f}%")

    # ────────────────── 9. 训练完成，输出汇总 ──────────────────
    total_time = time.time() - total_start

    summary = {
        "run_name": run_name,
        "model": "WidthResNet32",
        "dataset": "CIFAR-10",
        "stage_channels": args.channels,
        "blocks_per_stage": 5,
        "epochs": args.epochs,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "params": params,
        "flops": flops,
        "total_train_time_sec": total_time,
        "checkpoint_best": str(out_dir / "best.pt"),
        "checkpoint_last": str(out_dir / "last.pt"),
    }

    # 将实验摘要写入 JSON 文件，方便后续对比分析
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
