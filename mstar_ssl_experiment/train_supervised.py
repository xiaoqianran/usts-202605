#!/usr/bin/env python3
"""Supervised training for full-label and limited-label MSTAR experiments."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_supervised_datasets
from src.models import build_model
from src.utils import (
    append_csv,
    ensure_dir,
    evaluate,
    get_device,
    save_confusion_matrix,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MSTAR")
    parser.add_argument("--out", default="runs/supervised")
    parser.add_argument("--model", default="smallresnet", choices=["smallresnet", "smallcnn"])
    parser.add_argument("--label-ratio", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    save_json(vars(args), out_dir / "args.json")

    train_set, test_set, split = build_supervised_datasets(
        data_root=args.data_root,
        img_size=args.img_size,
        label_ratio=args.label_ratio,
        seed=args.seed,
        out_dir=out_dir,
    )
    save_json({k: v for k, v in split.items() if k not in {"labeled_indices", "unlabeled_indices"}}, out_dir / "split_summary.json")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = get_device()
    model = build_model(args.model, num_classes=len(test_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total += bs
            pbar.set_postfix(loss=total_loss / max(total, 1), acc=total_correct / max(total, 1))

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)
        eval_result = evaluate(model, test_loader, criterion, device, desc="test")
        test_acc = eval_result["acc"]

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "best_acc": best_acc}, out_dir / "best.pt")
            save_confusion_matrix(eval_result["targets"], eval_result["preds"], test_set.classes, out_dir / "confusion_matrix.csv")

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": eval_result["loss"],
            "test_acc": test_acc,
            "best_acc": best_acc,
        }
        append_csv(out_dir / "history.csv", row)
        print(row)

    metrics = {
        "method": "supervised",
        "label_ratio": args.label_ratio,
        "num_labeled": split["num_labeled"],
        "num_unlabeled_unused": split["num_unlabeled"],
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "time_seconds": round(time.time() - t0, 2),
    }
    save_json(metrics, out_dir / "metrics.json")
    print("Finished:", metrics)


if __name__ == "__main__":
    main()
