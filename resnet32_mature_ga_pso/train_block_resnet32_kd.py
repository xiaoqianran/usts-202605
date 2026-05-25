#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.data import get_cifar10_loaders
from src.models import block_width_resnet32, resnet32
from src.search import candidate_key
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed


def parse_block_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) != 15:
        raise argparse.ArgumentTypeError("block-channels must contain 15 integers")
    return values


def parse_milestones(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(fields)).writeheader()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)


def kd_loss_fn(student_logits, teacher_logits, targets, temperature: float, alpha: float):
    ce = F.cross_entropy(student_logits, targets)
    if alpha <= 0 or teacher_logits is None:
        return ce, ce.detach(), torch.tensor(0.0, device=student_logits.device)
    t = temperature
    kd = F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)
    loss = (1 - alpha) * ce + alpha * kd
    return loss, ce.detach(), kd.detach()


def train_one_epoch(student, teacher, loader, optimizer, device, use_amp: bool, temperature: float, alpha: float):
    student.train()
    if teacher is not None:
        teacher.eval()
    losses, ce_meter, kd_meter, top1 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images) if teacher is not None else None
        with torch.cuda.amp.autocast(enabled=use_amp):
            student_logits = student(images)
            loss, ce_loss, kd_loss = kd_loss_fn(student_logits, teacher_logits, targets, temperature, alpha)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(student_logits.detach(), targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        ce_meter.update(float(ce_loss.item()), images.size(0))
        kd_meter.update(float(kd_loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
    return losses.avg, ce_meter.avg, kd_meter.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
    return losses.avg, top1.avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final block-width ResNet32 with optional KD")
    parser.add_argument("--block-channels", type=parse_block_channels, required=True)
    parser.add_argument("--teacher-checkpoint", default=None, help="standard ResNet32 best.pt; if omitted, train without KD")
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--milestones", type=parse_milestones, default=[40, 60])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    tag = candidate_key(args.block_channels)
    run_name = args.run_name or f"final_block_{tag}"
    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers,
        max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples, seed=args.seed,
    )
    student = block_width_resnet32(args.block_channels, num_classes=10).to(device)
    teacher = None
    if args.teacher_checkpoint:
        teacher = resnet32(num_classes=10).to(device)
        ckpt = load_checkpoint(args.teacher_checkpoint, map_location="cpu")
        teacher.load_state_dict(ckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"KD enabled: teacher={args.teacher_checkpoint}, alpha={args.kd_alpha}, T={args.kd_temperature}")
    else:
        print("KD disabled: training compressed model with CE only")

    params = count_parameters(student)
    flops = measure_flops(student, input_size=(3, 32, 32), device=device)
    print(f"Student block channels={tag}\nParams={human_number(params)} FLOPs={human_number(flops)}")

    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    fields = ["epoch", "lr", "loss", "ce_loss", "kd_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"]
    write_csv_header_if_needed(metrics_csv, fields)
    best_acc, best_epoch = 0.0, -1
    total_start = time.time()
    config = vars(args).copy()
    config.update({"model": "BlockWidthResNet32", "dataset": "CIFAR-10"})
    for epoch in range(args.epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={lr:.6f}")
        loss, ce_loss, kd_loss, train_acc = train_one_epoch(student, teacher, train_loader, optimizer, device, use_amp, args.kd_temperature, args.kd_alpha if teacher is not None else 0.0)
        test_loss, test_acc = evaluate(student, test_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        append_csv(metrics_csv, {
            "epoch": epoch + 1, "lr": lr, "loss": f"{loss:.6f}", "ce_loss": f"{ce_loss:.6f}", "kd_loss": f"{kd_loss:.6f}",
            "train_acc": f"{train_acc:.4f}", "test_loss": f"{test_loss:.6f}", "test_acc": f"{test_acc:.4f}", "epoch_time_sec": f"{epoch_time:.2f}",
        })
        is_best = test_acc > best_acc
        if is_best:
            best_acc, best_epoch = test_acc, epoch + 1
        payload = {
            "epoch": epoch, "model": student.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_acc": best_acc, "best_epoch": best_epoch, "config": config, "params": params, "flops": flops,
        }
        save_checkpoint(out_dir / "last.pt", **payload)
        if is_best:
            save_checkpoint(out_dir / "best.pt", **payload)
        print(f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} best={best_acc:.2f} time={epoch_time:.1f}s")

    summary = {
        "run_name": run_name, "model": "BlockWidthResNet32", "dataset": "CIFAR-10", "block_channels": args.block_channels,
        "kd_enabled": bool(args.teacher_checkpoint), "kd_alpha": args.kd_alpha if args.teacher_checkpoint else 0.0,
        "kd_temperature": args.kd_temperature if args.teacher_checkpoint else None,
        "epochs": args.epochs, "best_acc": best_acc, "best_epoch": best_epoch, "params": params, "flops": flops,
        "total_train_time_sec": time.time() - total_start, "checkpoint_best": str(out_dir / "best.pt"), "checkpoint_last": str(out_dir / "last.pt"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
