#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.mstar_ssl.data import ImageFolderGray, resolve_mstar_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MSTAR")
    args = parser.parse_args()
    train_dir, test_dir = resolve_mstar_dirs(args.data_root)
    train_set = ImageFolderGray(train_dir)
    test_set = ImageFolderGray(test_dir)
    print(f"data_root: {Path(args.data_root).resolve()}")
    print(f"train images: {len(train_set)}")
    print(f"test images: {len(test_set)}")
    print(f"classes: {len(train_set.classes)}")
    for cls in train_set.classes:
        train_n = sum(1 for _, y in train_set.samples if y == train_set.class_to_idx[cls])
        test_n = sum(1 for _, y in test_set.samples if y == test_set.class_to_idx[cls])
        print(f"{cls:10s} train={train_n:4d} test={test_n:4d}")


if __name__ == "__main__":
    main()

