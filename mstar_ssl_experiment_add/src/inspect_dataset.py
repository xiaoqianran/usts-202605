#!/usr/bin/env python3
"""Inspect MSTAR train/test folders."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_split(split_dir: Path):
    class_counts = {}
    sizes = Counter()
    modes = Counter()
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        class_counts[cls_dir.name] = len(imgs)
        for img_path in imgs[:20]:
            with Image.open(img_path) as img:
                sizes[img.size] += 1
                modes[img.mode] += 1
    return class_counts, sizes, modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/MSTAR")
    args = parser.parse_args()

    root = Path(args.data_root)
    train_dir = root / "mstar-train"
    test_dir = root / "mstar-test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected mstar-train and mstar-test under data root.")

    for name, split_dir in [("train", train_dir), ("test", test_dir)]:
        counts, sizes, modes = count_split(split_dir)
        print(f"\n[{name}] total={sum(counts.values())}, classes={len(counts)}")
        for cls, n in counts.items():
            print(f"  {cls:10s}: {n}")
        print("  sampled image sizes:", sizes.most_common(10))
        print("  sampled modes      :", modes.most_common())


if __name__ == "__main__":
    main()
