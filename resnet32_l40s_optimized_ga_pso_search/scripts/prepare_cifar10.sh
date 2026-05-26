#!/bin/bash
set -euo pipefail

TARGET_DATA_DIR="${1:-data}"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_NAME="cifar-10-python.tar.gz"
EXTRACTED_DIR="cifar-10-batches-py"

echo "Preparing CIFAR-10 under: $TARGET_DATA_DIR"
mkdir -p "$TARGET_DATA_DIR"

if [ -d "$TARGET_DATA_DIR/$EXTRACTED_DIR" ]; then
    echo "CIFAR-10 already exists: $TARGET_DATA_DIR/$EXTRACTED_DIR"
    exit 0
fi

if [ ! -f "$TARGET_DATA_DIR/$TAR_NAME" ]; then
    wget -c --show-progress "$CIFAR_URL" -O "$TARGET_DATA_DIR/$TAR_NAME"
fi

tar -xzf "$TARGET_DATA_DIR/$TAR_NAME" -C "$TARGET_DATA_DIR"
test -d "$TARGET_DATA_DIR/$EXTRACTED_DIR"
echo "CIFAR-10 is ready: $TARGET_DATA_DIR/$EXTRACTED_DIR"
