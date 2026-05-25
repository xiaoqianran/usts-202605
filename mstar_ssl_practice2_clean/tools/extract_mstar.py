#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MSTAR.zip into data/MSTAR")
    parser.add_argument("--zip", default="MSTAR.zip")
    parser.add_argument("--out", default="data/MSTAR")
    args = parser.parse_args()
    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted {zip_path} -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()

