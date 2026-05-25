"""Training utilities."""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv(path: str | Path, row: Dict) -> None:
    path = Path(path)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, desc: str = "eval"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == targets).sum().item()
        total += bs
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
        "targets": all_targets,
        "preds": all_preds,
    }


def save_confusion_matrix(targets: List[int], preds: List[int], class_names: List[str], out_path: str | Path) -> None:
    n = len(class_names)
    mat = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        mat[t, p] += 1

    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for i, row in enumerate(mat):
            writer.writerow([class_names[i]] + row.tolist())
