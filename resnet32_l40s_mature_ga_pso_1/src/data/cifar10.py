from __future__ import annotations

import os
import pickle
import random
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split


CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


class LocalCIFAR10(Dataset):
    """CIFAR-10 reader for the extracted python version.

    Expected directory:
        data/cifar-10-batches-py/
            data_batch_1 ... data_batch_5
            test_batch

    This avoids importing torchvision, which can fail when torchvision and
    PyTorch CUDA builds do not match.
    """

    def __init__(self, root: str = "data", train: bool = True, augment: bool = False) -> None:
        self.root = Path(root)
        self.train = train
        self.augment = augment
        base = self.root / "cifar-10-batches-py"
        if not base.exists():
            tgz = self.root / "cifar-10-python.tar.gz"
            if tgz.exists():
                with tarfile.open(tgz, "r:gz") as tar:
                    tar.extractall(self.root)
            else:
                raise FileNotFoundError(
                    f"CIFAR-10 not found at {base}. Put cifar-10-batches-py under {self.root}, "
                    "or run once with a project copy that already contains the dataset."
                )

        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        data_list, labels = [], []
        for name in files:
            with (base / name).open("rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data_list.append(entry["data"])
            labels.extend(entry.get("labels", entry.get("fine_labels")))
        data = np.concatenate(data_list, axis=0).reshape(-1, 3, 32, 32)
        self.data = torch.from_numpy(data).float().div_(255.0)
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.numel())

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Random crop with padding=4 and random horizontal flip, CIFAR standard.
        x = torch.nn.functional.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
        top = random.randint(0, 8)
        left = random.randint(0, 8)
        x = x[:, top:top + 32, left:left + 32]
        if random.random() < 0.5:
            x = torch.flip(x, dims=(2,))
        return x

    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.train and self.augment:
            x = self._augment(x)
        x = (x - CIFAR10_MEAN) / CIFAR10_STD
        y = self.targets[idx]
        return x, y


def _subset(dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **kwargs)


def get_cifar10_loaders(
    data_dir: str = "data",
    batch_size: int = 1024,
    num_workers: int = 8,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
):
    train_set = LocalCIFAR10(data_dir, train=True, augment=True)
    test_set = LocalCIFAR10(data_dir, train=False, augment=False)
    train_set = _subset(train_set, max_train_samples, seed)
    test_set = _subset(test_set, max_test_samples, seed + 10000)
    return _loader(train_set, batch_size, True, num_workers), _loader(test_set, batch_size, False, num_workers)


def get_cifar10_train_val_loaders(
    data_dir: str = "data",
    batch_size: int = 2048,
    num_workers: int = 8,
    val_size: int = 5000,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    seed: int = 42,
):
    full = LocalCIFAR10(data_dir, train=True, augment=True)
    val_size = min(max(1, val_size), len(full) - 1)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set_aug = random_split(full, [train_size, val_size], generator=gen)

    # Validation must not use augmentation. Reuse the same indices on a non-augmented dataset.
    val_base = LocalCIFAR10(data_dir, train=True, augment=False)
    val_indices = val_set_aug.indices if hasattr(val_set_aug, "indices") else list(range(train_size, len(full)))
    val_set = Subset(val_base, val_indices)

    train_set = _subset(train_set, max_train_samples, seed + 1)
    val_set = _subset(val_set, max_val_samples, seed + 2)
    return _loader(train_set, batch_size, True, num_workers), _loader(val_set, batch_size, False, num_workers)


def get_cifar10_test_loader(
    data_dir: str = "data",
    batch_size: int = 256,
    num_workers: int = 2,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
):
    test_set = LocalCIFAR10(data_dir, train=False, augment=False)
    test_set = _subset(test_set, max_test_samples, seed + 10000)
    return _loader(test_set, batch_size, False, num_workers)
