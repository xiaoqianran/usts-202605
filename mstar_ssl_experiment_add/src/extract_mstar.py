#!/usr/bin/env python3
"""Extract MSTAR.zip into a standard folder.

Expected result:
  data/MSTAR/mstar-train/...
  data/MSTAR/mstar-test/...
"""
import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to MSTAR.zip")
    parser.add_argument("--out", default="data/MSTAR", help="Output data root")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    train_dir = out_dir / "mstar-train"
    test_dir = out_dir / "mstar-test"
    if not train_dir.exists() or not test_dir.exists():
        raise RuntimeError(
            "Extraction finished, but mstar-train/mstar-test were not found. "
            "Please check whether the zip file has an extra top-level folder."
        )

    print(f"Extracted to: {out_dir.resolve()}")
    print(f"Train dir: {train_dir}")
    print(f"Test dir : {test_dir}")


if __name__ == "__main__":
    main()
