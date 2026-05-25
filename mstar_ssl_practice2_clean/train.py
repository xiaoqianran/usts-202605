#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from src.mstar_ssl.data import build_fixmatch, build_supervised
from src.mstar_ssl.eval import evaluate, save_confusion_matrix
from src.mstar_ssl.models import build_model, describe_model
from src.mstar_ssl.utils import append_csv, ensure_dir, get_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSTAR supervised and FixMatch training")
    parser.add_argument("--mode", choices=["supervised", "fixmatch"], required=True)
    parser.add_argument("--data-root", default="data/MSTAR")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["smallresnet", "smallcnn"], default="smallresnet")
    parser.add_argument("--label-ratio", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mu", type=int, default=4, help="FixMatch unlabeled batch multiplier")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, drop_last: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def train_supervised(args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict:
    train_set, test_set, split = build_supervised(args.data_root, args.img_size, args.label_ratio, args.seed, out_dir)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)

    model = build_model(args.model, num_classes=len(test_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0.0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for images, targets in tqdm(train_loader, desc=f"supervised {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch = targets.size(0)
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(1) == targets).sum().item()
            total += batch

        scheduler.step()
        result = evaluate(model, test_loader, criterion, device, desc="test")
        if result["acc"] > best_acc:
            best_acc = result["acc"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "best.pt")
            save_confusion_matrix(result["targets"], result["preds"], test_set.classes, out_dir / "confusion_matrix.csv")

        append_csv(
            out_dir / "history.csv",
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": total_loss / max(total, 1),
                "train_acc": total_correct / max(total, 1),
                "test_loss": result["loss"],
                "test_acc": result["acc"],
                "best_acc": best_acc,
            },
        )

    return {
        "method": "supervised",
        "label_ratio": args.label_ratio,
        "num_labeled": split["num_labeled"],
        "num_unlabeled_unused": split["num_unlabeled"],
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "time_seconds": round(time.time() - start, 2),
    }


def train_fixmatch(args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict:
    labeled_set, unlabeled_set, test_set, split = build_fixmatch(
        args.data_root, args.img_size, args.label_ratio, args.seed, out_dir
    )
    labeled_sampler = RandomSampler(labeled_set, replacement=True, num_samples=max(len(unlabeled_set), len(labeled_set)))
    labeled_loader = DataLoader(
        labeled_set,
        batch_size=args.batch_size,
        sampler=labeled_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    unlabeled_loader = make_loader(unlabeled_set, args.batch_size * args.mu, True, args.num_workers, drop_last=True)
    test_loader = make_loader(test_set, args.batch_size * args.mu, False, args.num_workers)

    model = build_model(args.model, num_classes=len(test_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0.0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        labeled_iter = itertools.cycle(labeled_loader)
        loss_sum = loss_x_sum = loss_u_sum = mask_sum = 0.0
        correct_labeled = total_labeled = steps = 0

        for weak_u, strong_u in tqdm(unlabeled_loader, desc=f"fixmatch {epoch}/{args.epochs}"):
            labeled_x, labeled_y = next(labeled_iter)
            labeled_x = labeled_x.to(device, non_blocking=True)
            labeled_y = labeled_y.to(device, non_blocking=True)
            weak_u = weak_u.to(device, non_blocking=True)
            strong_u = strong_u.to(device, non_blocking=True)

            logits_x = model(labeled_x)
            loss_x = criterion(logits_x, labeled_y)
            with torch.no_grad():
                probs = torch.softmax(model(weak_u), dim=1)
                confidence, pseudo_label = probs.max(dim=1)
                mask = confidence.ge(args.threshold).float()
            loss_u = (F.cross_entropy(model(strong_u), pseudo_label, reduction="none") * mask).mean()
            loss = loss_x + args.lambda_u * loss_u

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            steps += 1
            batch = labeled_y.size(0)
            total_labeled += batch
            correct_labeled += (logits_x.argmax(1) == labeled_y).sum().item()
            loss_sum += loss.item()
            loss_x_sum += loss_x.item()
            loss_u_sum += loss_u.item()
            mask_sum += mask.mean().item()

        scheduler.step()
        result = evaluate(model, test_loader, criterion, device, desc="test")
        if result["acc"] > best_acc:
            best_acc = result["acc"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "best.pt")
            save_confusion_matrix(result["targets"], result["preds"], test_set.classes, out_dir / "confusion_matrix.csv")

        append_csv(
            out_dir / "history.csv",
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / max(steps, 1),
                "loss_x": loss_x_sum / max(steps, 1),
                "loss_u": loss_u_sum / max(steps, 1),
                "pseudo_label_used_rate": mask_sum / max(steps, 1),
                "labeled_train_acc": correct_labeled / max(total_labeled, 1),
                "test_loss": result["loss"],
                "test_acc": result["acc"],
                "best_acc": best_acc,
            },
        )

    return {
        "method": "fixmatch",
        "label_ratio": args.label_ratio,
        "num_labeled": split["num_labeled"],
        "num_unlabeled": split["num_unlabeled"],
        "threshold": args.threshold,
        "lambda_u": args.lambda_u,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "time_seconds": round(time.time() - start, 2),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out)
    save_json(vars(args), out_dir / "args.json")
    save_json(describe_model(), out_dir / "model_summary.json")
    device = get_device(args.device)
    metrics = train_supervised(args, out_dir, device) if args.mode == "supervised" else train_fixmatch(args, out_dir, device)
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

