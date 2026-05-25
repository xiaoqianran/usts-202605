#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_cifar10_test_loader
from src.models import block_width_resnet32, BASELINE_BLOCK_CHANNELS
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops


def parse_block_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) == 3:
        return [values[0]] * 5 + [values[1]] * 5 + [values[2]] * 5
    if len(values) != 15:
        raise argparse.ArgumentTypeError("block channels must contain 15 integers, or 3 stage integers to expand")
    return values


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
    parser = argparse.ArgumentParser(description="Evaluate block-level variable-width ResNet32 checkpoint on CIFAR-10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block-channels", type=parse_block_channels, default=None, help="optional; if omitted, read block_channels from checkpoint config")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    channels = args.block_channels or ckpt.get("config", {}).get("block_channels") or ckpt.get("config", {}).get("stage_channels")
    if channels is None:
        raise ValueError("block_channels not provided and not found in checkpoint config")
    channels = list(channels)
    if len(channels) == 3:
        channels = [channels[0]] * 5 + [channels[1]] * 5 + [channels[2]] * 5

    model = block_width_resnet32(block_channels=channels, num_classes=10)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    loader = get_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device)

    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    result = {
        "checkpoint": str(args.checkpoint),
        "model": "BlockWidthResNet32",
        "dataset": "CIFAR-10",
        "block_channels": list(channels),
        "stage_channels": [list(channels)[0], list(channels)[5], list(channels)[10]],
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
