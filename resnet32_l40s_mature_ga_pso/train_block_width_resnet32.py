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
from src.models import BASELINE_BLOCK_CHANNELS, block_width_resnet32, resnet32
from src.utils.accelerate import (
    autocast_context,
    make_grad_scaler,
    maybe_channels_last,
    maybe_compile,
    move_images,
    setup_torch_fast,
)
from src.utils.checkpoint import load_checkpoint, load_sliced_baseline_weights, save_checkpoint
from src.utils.kd import FeatureCapture, attention_transfer_loss, kd_loss
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed


def parse_block_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) == 3:
        return [values[0]] * 5 + [values[1]] * 5 + [values[2]] * 5
    if len(values) != 15:
        raise argparse.ArgumentTypeError(
            "block channels must contain 15 integers, or 3 stage integers to expand, e.g. 16,16,... or 16,24,48"
        )
    return values


def parse_milestones(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def load_teacher_model(checkpoint_path: str | Path, device: torch.device, channels_last: bool) -> nn.Module:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    teacher = resnet32(num_classes=10)
    teacher.load_state_dict(unwrap_state_dict(ckpt), strict=True)
    teacher = teacher.to(device)
    teacher = maybe_channels_last(teacher, channels_last)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[KD] loaded teacher from {ckpt_path}")
    return teacher


def build_feature_captures(student: nn.Module, teacher: nn.Module, kd_mode: str):
    if kd_mode != "logits_at":
        return None, None
    # Student block model uses stage1/2/3. Standard teacher uses layer1/2/3.
    student_capture = FeatureCapture(student, ["stage1", "stage2", "stage3"], detach=False)
    teacher_capture = FeatureCapture(teacher, ["layer1", "layer2", "layer3"], detach=True)
    return student_capture, teacher_capture


def train_one_epoch(
    model: nn.Module,
    loader,
    ce_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    channels_last: bool,
    teacher: nn.Module | None = None,
    kd_mode: str = "none",
    kd_alpha: float = 0.7,
    kd_temperature: float = 4.0,
    at_weight: float = 0.0,
    student_capture: FeatureCapture | None = None,
    teacher_capture: FeatureCapture | None = None,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()

    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    at_meter = AverageMeter()
    top1 = AverageMeter()
    scaler = make_grad_scaler(device, use_amp, amp_dtype)

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = move_images(images, device, channels_last)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        teacher_logits = None
        if teacher is not None and kd_mode != "none":
            if teacher_capture is not None:
                teacher_capture.clear()
            with torch.no_grad():
                with autocast_context(device, use_amp, amp_dtype):
                    teacher_logits = teacher(images)

        if student_capture is not None:
            student_capture.clear()
        with autocast_context(device, use_amp, amp_dtype):
            outputs = model(images)
            if teacher_logits is not None and kd_mode in {"logits", "logits_at"}:
                loss, ce_loss, kd_part = kd_loss(
                    outputs,
                    teacher_logits,
                    targets,
                    ce_criterion,
                    alpha=kd_alpha,
                    temperature=kd_temperature,
                )
                if kd_mode == "logits_at" and student_capture is not None and teacher_capture is not None and at_weight > 0:
                    at_part = attention_transfer_loss(student_capture.features, teacher_capture.features)
                    loss = loss + float(at_weight) * at_part
                else:
                    at_part = torch.zeros((), device=outputs.device)
            else:
                ce_loss = ce_criterion(outputs, targets)
                kd_part = torch.zeros((), device=outputs.device)
                at_part = torch.zeros((), device=outputs.device)
                loss = ce_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        n = images.size(0)
        total_meter.update(float(loss.item()), n)
        ce_meter.update(float(ce_loss.item()), n)
        kd_meter.update(float(kd_part.item()), n)
        at_meter.update(float(at_part.item()), n)
        top1.update(float(acc1.item()), n)
        pbar.set_postfix(loss=f"{total_meter.avg:.4f}", ce=f"{ce_meter.avg:.4f}", kd=f"{kd_meter.avg:.4f}", acc=f"{top1.avg:.2f}")

    return {
        "loss": total_meter.avg,
        "ce_loss": ce_meter.avg,
        "kd_loss": kd_meter.avg,
        "at_loss": at_meter.avg,
        "acc": top1.avg,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, amp_dtype: str, channels_last: bool) -> tuple[float, float]:
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    pbar = tqdm(loader, desc="eval", leave=False)
    for images, targets in pbar:
        images = move_images(images, device, channels_last)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, use_amp, amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{top1.avg:.2f}")
    return losses.avg, top1.avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 15-D block-width ResNet32 candidate on CIFAR-10, with optional KD")
    parser.add_argument("--block-channels", type=parse_block_channels, default=BASELINE_BLOCK_CHANNELS, help="15 block channels, or 3 stage channels expanded to 15")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--milestones", type=parse_milestones, default=[40, 60], help="e.g. 40,60")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="use torch.compile; disabled automatically for attention-transfer KD")
    parser.add_argument("--baseline-ckpt", default=None, help="optional standard ResNet32 checkpoint for sliced weight inheritance")
    parser.add_argument("--resume", default=None)

    # Knowledge distillation options. Keep KD in final training, not GA/PSO proxy search.
    parser.add_argument("--kd-mode", default="none", choices=["none", "logits", "logits_at"], help="KD mode for final compressed-model training")
    parser.add_argument("--teacher-ckpt", default=None, help="teacher standard ResNet32 checkpoint; normally runs/resnet32_baseline/best.pt")
    parser.add_argument("--kd-alpha", type=float, default=0.7, help="weight of logit KD term; CE weight is 1-alpha")
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--at-weight", type=float, default=0.0, help="attention-transfer loss weight; use with --kd-mode logits_at")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    setup_torch_fast(tf32=True, benchmark=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    if args.kd_mode != "none" and not args.teacher_ckpt:
        args.teacher_ckpt = args.baseline_ckpt
    if args.kd_mode != "none" and not args.teacher_ckpt:
        raise ValueError("KD requires --teacher-ckpt, or --baseline-ckpt can be reused as teacher.")
    if args.kd_mode == "logits_at" and args.compile:
        print("[KD] --compile disabled for logits_at because feature hooks and torch.compile can interact poorly.")
        args.compile = False

    channel_tag = "-".join(map(str, args.block_channels))
    run_name = args.run_name or f"block_width_resnet32_{channel_tag}"
    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    model = block_width_resnet32(block_channels=args.block_channels, num_classes=10).to(device)
    if args.baseline_ckpt:
        load_sliced_baseline_weights(model, args.baseline_ckpt, verbose=True)
    model = maybe_channels_last(model, args.channels_last)
    setattr(model, "_channels_last", args.channels_last)
    setattr(model, "_amp_dtype", args.amp_dtype)

    teacher = None
    student_capture = teacher_capture = None
    if args.kd_mode != "none":
        teacher = load_teacher_model(args.teacher_ckpt, device=device, channels_last=args.channels_last)
        student_capture, teacher_capture = build_feature_captures(model, teacher, args.kd_mode)
        print(
            f"[KD] mode={args.kd_mode}, alpha={args.kd_alpha}, "
            f"T={args.kd_temperature}, at_weight={args.at_weight}"
        )

    train_model = maybe_compile(model, args.compile)
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)

    print("Model: block-level variable-width CIFAR-10 ResNet32")
    print(f"Block channels: {channel_tag}")
    print("Blocks per stage: 5")
    print(f"Params: {params} ({human_number(params)})")
    print(f"FLOPs : {flops} ({human_number(flops)})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    start_epoch = 0
    best_acc = 0.0
    best_epoch = -1

    config = vars(args).copy()
    config["model"] = "BlockWidthResNet32"
    config["dataset"] = "CIFAR-10"
    config["block_channels"] = args.block_channels
    config["stage_channels"] = [args.block_channels[0], args.block_channels[5], args.block_channels[10]]
    config["blocks_per_stage"] = 5

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    fields = [
        "epoch", "lr", "train_loss", "train_ce_loss", "train_kd_loss", "train_at_loss", "train_acc",
        "test_loss", "test_acc", "epoch_time_sec",
    ]
    write_csv_header_if_needed(metrics_csv, fields)

    total_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch + 1}/{args.epochs} | lr={lr:.6f}")

            train_stats = train_one_epoch(
                train_model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_amp,
                args.amp_dtype,
                args.channels_last,
                teacher=teacher,
                kd_mode=args.kd_mode,
                kd_alpha=args.kd_alpha,
                kd_temperature=args.kd_temperature,
                at_weight=args.at_weight,
                student_capture=student_capture,
                teacher_capture=teacher_capture,
            )
            test_loss, test_acc = evaluate(train_model, test_loader, criterion, device, use_amp, args.amp_dtype, args.channels_last)
            scheduler.step()
            epoch_time = time.time() - epoch_start

            row = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": f"{train_stats['loss']:.6f}",
                "train_ce_loss": f"{train_stats['ce_loss']:.6f}",
                "train_kd_loss": f"{train_stats['kd_loss']:.6f}",
                "train_at_loss": f"{train_stats['at_loss']:.6f}",
                "train_acc": f"{train_stats['acc']:.4f}",
                "test_loss": f"{test_loss:.6f}",
                "test_acc": f"{test_acc:.4f}",
                "epoch_time_sec": f"{epoch_time:.2f}",
            }
            append_csv(metrics_csv, row)
            print(
                f"train_loss={train_stats['loss']:.4f} ce={train_stats['ce_loss']:.4f} "
                f"kd={train_stats['kd_loss']:.4f} at={train_stats['at_loss']:.4f} "
                f"train_acc={train_stats['acc']:.2f} | test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.2f} | time={epoch_time:.1f}s"
            )

            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                best_epoch = epoch + 1

            payload = {
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
            save_checkpoint(out_dir / "last.pt", **payload)
            if is_best:
                save_checkpoint(out_dir / "best.pt", **payload)
                print(f"Saved new best checkpoint: acc={best_acc:.2f}%")
    finally:
        if student_capture is not None:
            student_capture.close()
        if teacher_capture is not None:
            teacher_capture.close()

    total_time = time.time() - total_start
    summary = {
        "run_name": run_name,
        "model": "BlockWidthResNet32",
        "dataset": "CIFAR-10",
        "block_channels": args.block_channels,
        "stage_channels": [args.block_channels[0], args.block_channels[5], args.block_channels[10]],
        "blocks_per_stage": 5,
        "epochs": args.epochs,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "params": params,
        "flops": flops,
        "kd_mode": args.kd_mode,
        "teacher_ckpt": args.teacher_ckpt,
        "kd_alpha": args.kd_alpha,
        "kd_temperature": args.kd_temperature,
        "at_weight": args.at_weight,
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
