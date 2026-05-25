#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_cifar10_test_loader
from src.models import resnet32
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops



@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    for images, targets in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
    return losses.avg, top1.avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standard ResNet32 checkpoint on CIFAR-10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")

    model = resnet32(num_classes=10)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    loader = get_cifar10_test_loader(args.data_dir, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device)

    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    result = {
        "checkpoint": str(args.checkpoint),
        "model": "ResNet32",
        "dataset": "CIFAR-10",
        "test_loss": loss,
        "test_acc": acc,
        "params": params,
        "flops": flops,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Params: {human_number(params)} | FLOPs: {human_number(flops)} | Acc: {acc:.2f}%")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
