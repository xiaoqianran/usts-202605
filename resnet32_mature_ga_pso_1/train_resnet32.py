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
import torch.optim as optim
from tqdm import tqdm

from src.data import get_cifar10_loaders
from src.models import resnet32
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed


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


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool) -> tuple[float, float]:
    model.train()
    losses, top1 = AverageMeter(), AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}")
    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standard ResNet32 on CIFAR-10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default="resnet32_baseline")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--milestones", type=parse_milestones, default=[100, 150])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    out_dir = Path(args.save_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    model = resnet32(num_classes=10).to(device)
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    print(f"Model: standard ResNet32 | Params={human_number(params)} | FLOPs={human_number(flops)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    start_epoch, best_acc, best_epoch = 0, 0.0, -1
    config = vars(args).copy()
    config.update({"model": "ResNet32", "dataset": "CIFAR-10", "stage_channels": [16, 32, 64]})

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))
        best_epoch = int(ckpt.get("best_epoch", -1))

    fields = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_sec"]
    write_csv_header_if_needed(metrics_csv, fields)
    total_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={lr:.6f}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        append_csv(metrics_csv, {
            "epoch": epoch + 1, "lr": lr, "train_loss": f"{train_loss:.6f}", "train_acc": f"{train_acc:.4f}",
            "test_loss": f"{test_loss:.6f}", "test_acc": f"{test_acc:.4f}", "epoch_time_sec": f"{epoch_time:.2f}",
        })
        print(f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} time={epoch_time:.1f}s")
        is_best = test_acc > best_acc
        if is_best:
            best_acc, best_epoch = test_acc, epoch + 1
        payload = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_acc": best_acc, "best_epoch": best_epoch,
            "config": config, "params": params, "flops": flops,
        }
        save_checkpoint(out_dir / "last.pt", **payload)
        if is_best:
            save_checkpoint(out_dir / "best.pt", **payload)

    summary = {
        "run_name": args.run_name, "model": "ResNet32", "dataset": "CIFAR-10",
        "stage_channels": [16, 32, 64], "epochs": args.epochs,
        "best_acc": best_acc, "best_epoch": best_epoch, "params": params, "flops": flops,
        "total_train_time_sec": time.time() - total_start,
        "checkpoint_best": str(out_dir / "best.pt"), "checkpoint_last": str(out_dir / "last.pt"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
