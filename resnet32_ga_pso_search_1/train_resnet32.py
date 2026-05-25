#!/usr/bin/env python3
"""在 CIFAR-10 上训练标准 ResNet32 基线模型。

功能概览：
  - 支持学习率多阶段衰减（MultiStepLR）
  - 支持混合精度训练（AMP）
  - 支持断点续训（resume）
  - 每个 epoch 记录指标到 CSV，训练结束后输出 summary.json
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
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from src.models import resnet32
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed


# ── 辅助函数 ────────────────────────────────────────────────────────

def parse_milestones(s: str) -> list[int]:
    """将逗号分隔的字符串解析为学习率衰减的 epoch 列表。

    例如 "100,150" → [100, 150]，表示在第 100 和第 150 个 epoch 衰减学习率。

    Args:
        s: 逗号分隔的 epoch 字符串。

    Returns:
        整数列表；空字符串返回空列表。
    """
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """构建 CIFAR-10 的训练集和测试集 DataLoader。

    训练集应用数据增强（随机裁剪 + 水平翻转），测试集仅做归一化。

    Args:
        data_dir:    CIFAR-10 数据存放目录（不存在时自动下载）。
        batch_size:  每个 mini-batch 的样本数。
        num_workers: 数据加载的子进程数。

    Returns:
        (train_loader, test_loader) 元组。
    """
    # CIFAR-10 训练集上统计得到的三通道均值和标准差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # 训练集：随机裁剪（先 padding 4 像素再裁回 32×32）+ 水平翻转 + 归一化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ##  transforms.RandomCrop(32, padding=4),
    # 操作：先在原32×32图像四周各填充4个像素（变成40×40），然后从中随机裁剪出一个32×32的区域。
    # 目的：引入位置偏移，让模型对物体位置不那么敏感（提升平移不变性）。
    # 效果：相当于对物体做了轻微的“随机移动”，防止模型只记住固定位置的特征。

    ## transforms.RandomHorizontalFlip()
    # 操作：以 50% 概率 左右翻转图像。
    # 目的：增加数据多样性。
    # 为什么有效：CIFAR-10 中的飞机、汽车、鸟、猫狗等，很多左右翻转后依然是合理的样本，几乎不增加任何计算成本。

    ## transforms.ToTensor() 【ToTensor() 只做 0-1 缩放】
    # 把 PIL Image 对象转为 torch.Tensor。
    # 同时把像素值从 [0, 255] 缩放到 [0.0, 1.0]。

    ## transforms.Normalize(mean, std)
    # 使用 CIFAR-10 的全局统计值进行归一化：
    # 分别为每个通道减去它自己的均值，除以它自己的标准差，让三个通道的数值分布都尽量接近 均值≈0，标准差≈1。
    # 目的：让输入图像的每个通道均值接近0，标准差接近1。
    # 好处：加速神经网络收敛，防止梯度爆炸/消失，训练更稳定。
    # 所有特征处于相似尺度上，网络可以同时、均衡地学习不同通道的特征，收敛速度明显变快（通常能快几倍）。



    # 测试集：仅 ToTensor + 归一化，不做增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 创建 Dataset
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    # 创建 Dataset
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,                                   # 训练集每个 epoch 打乱顺序
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),            # GPU 可用时启用锁页内存加速
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,                                  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# ── 训练与评估 ──────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """训练模型一个 epoch，返回平均损失和 Top-1 准确率。

    Args:
        model:     待训练模型。
        loader:    训练集 DataLoader。
        criterion: 损失函数。
        optimizer: 优化器。
        device:    计算设备。
        use_amp:   是否使用自动混合精度（AMP）。

    Returns:
        (平均训练损失, 训练 Top-1 准确率 %) 元组。
    """
    model.train()                               # 训练模式：启用 Dropout、更新 BN 统计量
    losses = AverageMeter()
    top1 = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # AMP 梯度缩放器，防止 FP16 梯度下溢

    # 什么是 AMP + GradScaler？
    # 传统训练：全部用 FP32（32位浮点），占显存多，速度较慢。
    # AMP 混合精度：大部分运算用 FP16（半精度），部分敏感运算仍保留 FP32，既快又省显存。
    
    # 但 FP16 数值范围小，很容易出现梯度下溢（grad underflow） —— 梯度变得极小接近0，导致模型学不到东西。
    # 不开启 AMP 纯 FP32 也可以跑，但浪费时间和显存，属于“不会玩”。

    pbar = tqdm(loader, desc="train", leave=False)

    # 训练 batch 流程：
    for images, targets in pbar:

        # 1. 数据搬 GPU（异步传输加速）
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 2. 清梯度（更高效的 None 方式，节省显存）
        optimizer.zero_grad(set_to_none=True)   # 将梯度设为 None 而非填零，节省显存

        # 3. AMP 前向传播（自动使用 FP16 加速）
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # 4. AMP 反向传播：先缩放损失再反传，防止 FP16 梯度精度不足
        scaler.scale(loss).backward()           # 先缩放 loss 防止 FP16 下溢
        scaler.step(optimizer)                  # 先 unscale 梯度，再执行 optimizer.step
        scaler.update()                         # 更新缩放因子

        # 5. 计算指标 + 更新统计 + 刷新进度条
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}")

    return losses.avg, top1.avg


@torch.no_grad()  # 评估阶段关闭梯度计算
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集上评估模型，返回平均损失和 Top-1 准确率。

    Args:
        model:     待评估模型。
        loader:    测试集 DataLoader。
        criterion: 损失函数。
        device:    计算设备。

    Returns:
        (平均测试损失, 测试 Top-1 准确率 %) 元组。
    """
    # 切换到评估模式
    model.eval()                    # 评估模式：关闭 Dropout，BN 使用训练时积累的全局统计量
    
    # 初始化统计器
    losses = AverageMeter()         # 累计平均损失
    top1 = AverageMeter()           # 累计平均 Top-1 准确率

    # 创建进度条
    pbar = tqdm(loader, desc="eval", leave=False)

    for images, targets in pbar:
        # 1. 数据搬运到 GPU（异步传输加速）
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 2. 前向传播（纯推理，无梯度）
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # 3. 计算指标
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        
        # 4. 更新统计（加权平均）
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
        
        # 5. 实时显示进度
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}")

    # 返回整个评估集的平均结果
    return losses.avg, top1.avg


# ── CSV 日志工具 ────────────────────────────────────────────────────

def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    """如果 CSV 文件不存在则创建并写入表头，避免重复写入。"""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()


def append_csv(path: Path, row: dict) -> None:
    """向 CSV 文件追加一行数据（用于每个 epoch 结束后记录指标）。"""
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# ── 主流程 ──────────────────────────────────────────────────────────

def main() -> None:
    """程序入口：配置 → 数据 → 模型 → 训练循环 → 保存结果。"""

    # ── 1. 命令行参数 ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train standard ResNet32 on CIFAR-10")
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 数据目录")
    parser.add_argument("--save-dir", default="runs", help="输出根目录")
    parser.add_argument("--run-name", default="resnet32_baseline", help="本次实验名称（子目录）")
    parser.add_argument("--epochs", type=int, default=200, help="总训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="mini-batch 大小")
    parser.add_argument("--lr", type=float, default=0.1, help="初始学习率")
    parser.add_argument("--milestones", type=parse_milestones, default=[100, 150],
                        help="学习率衰减的 epoch 节点，如 100,150")
    parser.add_argument("--gamma", type=float, default=0.1, help="每次衰减的倍率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD 动量")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减（L2 正则化）")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--amp", action="store_true", help="启用 CUDA 混合精度训练")
    parser.add_argument("--resume", default=None, help="从指定 checkpoint 断点续训")
    args = parser.parse_args()

    # ── 2. 环境准备 ────────────────────────────────────────────────
    set_seed(args.seed, deterministic=False)            # 固定随机种子，保证可复现

    # 设备回退：请求 cuda 但不可用时自动切换到 cpu
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    # AMP 仅在 CUDA 设备上有加速效果
    use_amp = bool(args.amp and device.type == "cuda")

    # 创建本次实验的输出目录
    out_dir = Path(args.save_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    # ── 3. 记录配置 ────────────────────────────────────────────────
    config = vars(args).copy()
    config["model"] = "ResNet32"
    config["dataset"] = "CIFAR-10"
    config["stage_channels"] = [16, 32, 64]     # ResNet32 三个阶段的输出通道数
    config["blocks_per_stage"] = 5              # 每个阶段的残差块数量

    # ── 4. 数据加载 ────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    # ── 5. 模型构建 ────────────────────────────────────────────────
    model = resnet32(num_classes=10).to(device)

    # 统计模型复杂度
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    print("Model: standard CIFAR-10 ResNet32")
    print("Stage channels: 16-32-64")
    print("Blocks per stage: 5")
    print(f"Params: {params} ({human_number(params)})")
    print(f"FLOPs : {flops} ({human_number(flops)})")

    # ── 6. 优化器、损失函数、学习率调度器 ──────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # MultiStepLR：在 milestones 指定的 epoch 将学习率乘以 gamma
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    # ── 7. 断点续训（可选） ────────────────────────────────────────
    start_epoch = 0
    best_acc = 0.0
    best_epoch = -1

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1       # 从下一个 epoch 继续
        best_acc = float(ckpt.get("best_acc", 0.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── 8. CSV 日志初始化 ──────────────────────────────────────────
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "epoch_time_sec",
    ]
    write_csv_header_if_needed(metrics_csv, fields)

    # ── 9. 训练循环 ────────────────────────────────────────────────
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]        # 记录当前学习率（step 前）
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={lr:.6f}")

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp
        )

        # 在测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 学习率调度：每个 epoch 结束后调用 step
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # 记录本轮指标到 CSV
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

        # ── 保存 checkpoint ────────────────────────────────────────
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch + 1

        # checkpoint 包含恢复训练所需的全部状态
        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": config,
            "params": params,
            "flops": flops,
        }

        # 始终保存最新 checkpoint（覆盖写入）
        save_checkpoint(out_dir / "last.pt", **ckpt_payload)

        # 仅在准确率刷新历史最佳时保存 best checkpoint
        if is_best:
            save_checkpoint(out_dir / "best.pt", **ckpt_payload)
            print(f"Saved new best checkpoint: acc={best_acc:.2f}%")

    # ── 10. 训练结束，输出 summary ─────────────────────────────────
    total_time = time.time() - total_start

    summary = {
        "run_name": args.run_name,
        "model": "ResNet32",
        "dataset": "CIFAR-10",
        "stage_channels": [16, 32, 64],
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

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
