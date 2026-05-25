from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset, Subset

from .transforms import build_transforms
from .utils import load_json, save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageFolderGray(Dataset):
    """Small grayscale ImageFolder replacement to avoid a torchvision dependency."""

    def __init__(self, root: str | Path, transform: Any = None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not self.classes:
            raise FileNotFoundError(f"No class folders found under {self.root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []
        for class_name in self.classes:
            for path in sorted((self.root / class_name).rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(path), self.class_to_idx[class_name]))
        if not self.samples:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class TwoViewUnlabeledDataset(Dataset):
    """Returns weak and strong views of the same unlabeled image for FixMatch."""

    def __init__(self, root: str | Path, indices: list[int], weak_transform: Any, strong_transform: Any):
        base = ImageFolderGray(root)
        self.samples = [base.samples[index] for index in indices]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        image = Image.open(path).convert("L")
        return self.weak_transform(image), self.strong_transform(image)


def resolve_mstar_dirs(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root)
    train_dir = root / "mstar-train"
    test_dir = root / "mstar-test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected {train_dir} and {test_dir}")
    return train_dir, test_dir


def make_split(dataset: ImageFolderGray, label_ratio: float, seed: int, split_path: str | Path) -> dict[str, Any]:
    if not (0 < label_ratio <= 1.0):
        raise ValueError("--label-ratio must be in (0, 1].")
    split_path = Path(split_path)
    if split_path.exists():
        return load_json(split_path)

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, target) in enumerate(dataset.samples):
        by_class[target].append(index)

    rng = random.Random(seed)
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    labeled_indices: list[int] = []
    unlabeled_indices: list[int] = []
    per_class_total: dict[str, int] = {}
    per_class_labeled: dict[str, int] = {}

    for class_idx, indices in sorted(by_class.items()):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_labeled = len(shuffled) if label_ratio >= 1.0 else max(1, int(len(shuffled) * label_ratio))
        labeled_indices.extend(sorted(shuffled[:n_labeled]))
        unlabeled_indices.extend(sorted(shuffled[n_labeled:]))
        class_name = idx_to_class[class_idx]
        per_class_total[class_name] = len(indices)
        per_class_labeled[class_name] = n_labeled

    split = {
        "label_ratio": label_ratio,
        "seed": seed,
        "num_train": len(dataset),
        "num_labeled": len(labeled_indices),
        "num_unlabeled": len(unlabeled_indices),
        "class_to_idx": dataset.class_to_idx,
        "per_class_total": per_class_total,
        "per_class_labeled": per_class_labeled,
        "labeled_indices": sorted(labeled_indices),
        "unlabeled_indices": sorted(unlabeled_indices),
    }
    save_json(split, split_path)
    return split


def split_filename(label_ratio: float, seed: int) -> str:
    ratio = str(label_ratio).replace(".", "p")
    return f"split_ratio{ratio}_seed{seed}.json"


def build_supervised(data_root: str | Path, img_size: int, label_ratio: float, seed: int, out_dir: str | Path):
    train_dir, test_dir = resolve_mstar_dirs(data_root)
    weak_transform, _, eval_transform = build_transforms(img_size)
    train_full = ImageFolderGray(train_dir, transform=weak_transform)
    train_for_split = ImageFolderGray(train_dir)
    test_set = ImageFolderGray(test_dir, transform=eval_transform)
    split = make_split(train_for_split, label_ratio, seed, Path(out_dir) / split_filename(label_ratio, seed))
    return Subset(train_full, split["labeled_indices"]), test_set, split


def build_fixmatch(data_root: str | Path, img_size: int, label_ratio: float, seed: int, out_dir: str | Path):
    train_dir, test_dir = resolve_mstar_dirs(data_root)
    weak_transform, strong_transform, eval_transform = build_transforms(img_size)
    labeled_full = ImageFolderGray(train_dir, transform=weak_transform)
    train_for_split = ImageFolderGray(train_dir)
    test_set = ImageFolderGray(test_dir, transform=eval_transform)
    split = make_split(train_for_split, label_ratio, seed, Path(out_dir) / split_filename(label_ratio, seed))
    labeled_set = Subset(labeled_full, split["labeled_indices"])
    unlabeled_set = TwoViewUnlabeledDataset(train_dir, split["unlabeled_indices"], weak_transform, strong_transform)
    return labeled_set, unlabeled_set, test_set, split

