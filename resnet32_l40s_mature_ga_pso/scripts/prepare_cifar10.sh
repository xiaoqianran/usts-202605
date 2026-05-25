#!/bin/bash
# prepare_cifar10.sh
# 一键准备 CIFAR-10 数据集（供 resnet32_l40s_mature_ga_pso 及同类项目使用）
#
# 用法:
#   bash scripts/prepare_cifar10.sh                    # 默认在当前项目 data/ 目录准备
#   bash scripts/prepare_cifar10.sh /path/to/shared/data   # 指定共享数据目录
#
# 推荐做法：
#   1. 在 workspace 根目录建一个共享 data 文件夹
#   2. 每个 resnet32 项目里 ln -s /shared/path ./data
#   3. 或者直接用本脚本指定路径

set -euo pipefail

# 默认数据目录（相对当前项目）
TARGET_DATA_DIR="${1:-data}"

CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_NAME="cifar-10-python.tar.gz"
EXTRACTED_DIR="cifar-10-batches-py"

echo "========================================"
echo "准备 CIFAR-10 数据集"
echo "目标目录: $TARGET_DATA_DIR"
echo "========================================"

mkdir -p "$TARGET_DATA_DIR"

if [ -d "$TARGET_DATA_DIR/$EXTRACTED_DIR" ]; then
    echo "✅ CIFAR-10 已存在: $TARGET_DATA_DIR/$EXTRACTED_DIR"
    echo "无需重复下载/解压。"
    exit 0
fi

# 下载 tarball（如果不存在）
if [ ! -f "$TARGET_DATA_DIR/$TAR_NAME" ]; then
    echo "⬇️  正在下载 CIFAR-10 (约 163MB) ..."
    wget -c --show-progress "$CIFAR_URL" -O "$TARGET_DATA_DIR/$TAR_NAME"
    echo "下载完成。"
else
    echo "📦 已发现 tarball: $TARGET_DATA_DIR/$TAR_NAME"
fi

# 解压
echo "📦 正在解压到 $TARGET_DATA_DIR ..."
tar -xzf "$TARGET_DATA_DIR/$TAR_NAME" -C "$TARGET_DATA_DIR"

if [ -d "$TARGET_DATA_DIR/$EXTRACTED_DIR" ]; then
    echo "✅ 解压成功！"
    echo "数据位置: $TARGET_DATA_DIR/$EXTRACTED_DIR"
    echo ""
    echo "现在可以直接运行训练命令，例如："
    echo "  python train_resnet32.py --run-name xxx --batch-size 128 --amp ..."
else
    echo "❌ 解压失败，请检查 tar 文件是否完整。"
    exit 1
fi

echo "========================================"
echo "CIFAR-10 准备完成。"
echo "========================================"
