from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import write_csv_rows


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, desc: str = "eval") -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    targets_all: list[int] = []
    preds_all: list[int] = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        batch = targets.size(0)
        total_loss += loss.item() * batch
        total_correct += (preds == targets).sum().item()
        total += batch
        targets_all.extend(targets.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
        "targets": targets_all,
        "preds": preds_all,
    }


def confusion_matrix(targets: list[int], preds: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for target, pred in zip(targets, preds):
        matrix[target, pred] += 1
    return matrix


def save_confusion_matrix(targets: list[int], preds: list[int], class_names: list[str], path) -> None:
    matrix = confusion_matrix(targets, preds, len(class_names))
    rows = ([class_names[i], *matrix[i].tolist()] for i in range(len(class_names)))
    write_csv_rows(path, ["true/pred", *class_names], rows)

