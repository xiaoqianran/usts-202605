"""Dataset, transforms, and limited-label split utilities for MSTAR.

This file intentionally avoids torchvision to reduce installation problems.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class SimpleImageFolder(Dataset):
    """Minimal replacement for torchvision.datasets.ImageFolder.

    Directory format:
      root/class_name/image.jpeg
    """

    def __init__(self, root: str | Path, transform: Any = None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise FileNotFoundError(f"No class folders found under {self.root}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for path in sorted(cls_dir.rglob("*")):
                if path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((str(path), self.class_to_idx[cls_name]))
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class Compose:
    def __init__(self, transforms: Sequence[Any]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size, self.size), resample=Image.BILINEAR)


class RandomCropWithPadding:
    def __init__(self, size: int, padding: int = 0, fill: int = 0):
        self.size = size
        self.padding = padding
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=self.fill)
        w, h = img.size
        if w == self.size and h == self.size:
            return img
        if w < self.size or h < self.size:
            padded = Image.new("L", (max(w, self.size), max(h, self.size)), color=self.fill)
            padded.paste(img, ((padded.width - w) // 2, (padded.height - h) // 2))
            img = padded
            w, h = img.size
        left = random.randint(0, w - self.size)
        top = random.randint(0, h - self.size)
        return img.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.mirror(img)
        return img


class RandomAffineLite:
    """Small rotation and translation for PIL grayscale images."""

    def __init__(self, degrees: float = 10.0, translate: Tuple[float, float] = (0.08, 0.08), p: float = 0.8):
        self.degrees = degrees
        self.translate = translate
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx = int(self.translate[0] * img.size[0])
        max_dy = int(self.translate[1] * img.size[1])
        tx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        ty = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        return img.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty), fillcolor=0)


class ToTensorNormalize:
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        return (tensor - self.mean) / self.std


class GaussianNoise:
    """Add mild Gaussian noise to a tensor image."""

    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, -1.0, 1.0)


class RandomErasingLite:
    def __init__(self, p: float = 0.25, scale: Tuple[float, float] = (0.02, 0.08), value: float = 0.0):
        self.p = p
        self.scale = scale
        self.value = value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        c, h, w = tensor.shape
        area = h * w
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        erase_size = max(1, int(erase_area ** 0.5))
        eh = min(h, erase_size)
        ew = min(w, erase_size)
        top = random.randint(0, h - eh)
        left = random.randint(0, w - ew)
        tensor = tensor.clone()
        tensor[:, top:top + eh, left:left + ew] = self.value
        return tensor


class TwoViewImageFolder(Dataset):
    """Return weak and strong augmented views for unlabeled FixMatch training."""

    def __init__(self, root: str | Path, indices: Sequence[int], weak_transform: Any, strong_transform: Any):
        self.base = SimpleImageFolder(root, transform=None)
        self.indices = list(indices)
        self.samples = [self.base.samples[i] for i in self.indices]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, _target = self.samples[idx]
        img = Image.open(path).convert("L")
        weak = self.weak_transform(img)
        strong = self.strong_transform(img)
        return weak, strong


def resolve_mstar_dirs(data_root: str | Path) -> Tuple[Path, Path]:
    root = Path(data_root)
    train_dir = root / "mstar-train"
    test_dir = root / "mstar-test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {test_dir}. Run src/extract_mstar.py first or check --data-root."
        )
    return train_dir, test_dir


def build_transforms(img_size: int = 128):
    """Build conservative augmentations for grayscale SAR images."""
    eval_tf = Compose([
        Resize(img_size),
        ToTensorNormalize(mean=0.5, std=0.5),
    ])

    weak_tf = Compose([
        Resize(img_size),
        RandomCropWithPadding(img_size, padding=8, fill=0),
        RandomHorizontalFlip(p=0.5),
        ToTensorNormalize(mean=0.5, std=0.5),
    ])

    strong_tf = Compose([
        Resize(img_size),
        RandomCropWithPadding(img_size, padding=12, fill=0),
        RandomHorizontalFlip(p=0.5),
        RandomAffineLite(degrees=10, translate=(0.08, 0.08), p=0.8),
        ToTensorNormalize(mean=0.5, std=0.5),
        GaussianNoise(std=0.03),
        RandomErasingLite(p=0.25, scale=(0.02, 0.08), value=0.0),
    ])
    return weak_tf, strong_tf, eval_tf


def _split_filename(label_ratio: float, seed: int) -> str:
    ratio_tag = str(label_ratio).replace(".", "p")
    return f"split_ratio{ratio_tag}_seed{seed}.json"


def make_or_load_split(
    train_dataset: SimpleImageFolder,
    label_ratio: float,
    seed: int,
    out_dir: str | Path,
    force: bool = False,
) -> Dict[str, Any]:
    """Create a class-balanced labeled/unlabeled split.

    For label_ratio=0.1, uses floor(n_class * 0.1), therefore the labeled set is below or equal
    to 10% for each class. For the uploaded MSTAR split this gives 268 labeled images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / _split_filename(label_ratio, seed)

    if split_path.exists() and not force:
        with split_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if not (0 < label_ratio <= 1.0):
        raise ValueError("label_ratio must be in (0, 1].")

    targets = [target for _path, target in train_dataset.samples]
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        by_class[target].append(idx)

    rng = random.Random(seed)
    labeled_indices: List[int] = []
    unlabeled_indices: List[int] = []
    per_class_labeled: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    for cls_idx, indices in sorted(by_class.items()):
        indices = indices.copy()
        rng.shuffle(indices)
        if label_ratio >= 1.0:
            n_labeled = len(indices)
        else:
            n_labeled = max(1, int(len(indices) * label_ratio))
        chosen = sorted(indices[:n_labeled])
        rest = sorted(indices[n_labeled:])
        labeled_indices.extend(chosen)
        unlabeled_indices.extend(rest)
        cls_name = idx_to_class[cls_idx]
        per_class_labeled[cls_name] = n_labeled
        per_class_total[cls_name] = len(indices)

    split = {
        "label_ratio": label_ratio,
        "seed": seed,
        "num_train": len(train_dataset),
        "num_labeled": len(labeled_indices),
        "num_unlabeled": len(unlabeled_indices),
        "class_to_idx": train_dataset.class_to_idx,
        "per_class_total": per_class_total,
        "per_class_labeled": per_class_labeled,
        "labeled_indices": sorted(labeled_indices),
        "unlabeled_indices": sorted(unlabeled_indices),
    }

    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    return split


def build_supervised_datasets(data_root: str | Path, img_size: int, label_ratio: float, seed: int, out_dir: str | Path):
    train_dir, test_dir = resolve_mstar_dirs(data_root)
    weak_tf, _strong_tf, eval_tf = build_transforms(img_size)
    train_full = SimpleImageFolder(train_dir, transform=weak_tf)
    train_for_split = SimpleImageFolder(train_dir, transform=None)
    test_set = SimpleImageFolder(test_dir, transform=eval_tf)
    split = make_or_load_split(train_for_split, label_ratio=label_ratio, seed=seed, out_dir=out_dir)
    train_set = Subset(train_full, split["labeled_indices"])
    return train_set, test_set, split


def build_fixmatch_datasets(data_root: str | Path, img_size: int, label_ratio: float, seed: int, out_dir: str | Path):
    train_dir, test_dir = resolve_mstar_dirs(data_root)
    weak_tf, strong_tf, eval_tf = build_transforms(img_size)
    labeled_full = SimpleImageFolder(train_dir, transform=weak_tf)
    train_for_split = SimpleImageFolder(train_dir, transform=None)
    test_set = SimpleImageFolder(test_dir, transform=eval_tf)
    split = make_or_load_split(train_for_split, label_ratio=label_ratio, seed=seed, out_dir=out_dir)
    labeled_set = Subset(labeled_full, split["labeled_indices"])
    unlabeled_set = TwoViewImageFolder(train_dir, split["unlabeled_indices"], weak_transform=weak_tf, strong_transform=strong_tf)
    return labeled_set, unlabeled_set, test_set, split
