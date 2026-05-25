#!/usr/bin/env python3
"""比较基线 ResNet32 与压缩后 WidthResNet32 的模型摘要。

读取两份 summary.json（分别来自基线模型和压缩模型），计算精度差、
参数压缩率、FLOPs 缩减率等指标，以 JSON 格式输出对比结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    """从指定路径加载并返回一个 JSON 对象。

    Args:
        path: JSON 文件的路径（字符串或 Path 对象）。

    Returns:
        解析后的字典。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """程序入口：解析命令行参数，加载两份摘要，计算对比指标并输出结果。"""

    # ── 命令行参数解析 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="对比基准版 ResNet32 与压缩版 WidthResNet32 的统计结果"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="基准模型的 summary.json 文件，示例：runs/resnet32_baseline/summary.json",
    )
    parser.add_argument(
        "--compressed",
        required=True,
        help="压缩模型的 summary.json 文件，示例：runs/final_width_16-24-48/summary.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="（可选）输出对比结果的 JSON 文件路径；不指定则仅打印到终端",
    )
    args = parser.parse_args()

    # ── 加载基线与压缩模型的摘要 ────────────────────────────────────
    base = load_json(args.baseline)   # 基线模型的 summary.json
    comp = load_json(args.compressed) # 压缩模型的 summary.json

    # ── 提取关键指标 ────────────────────────────────────────────────
    #   best_acc   : 训练过程中的最高验证准确率（%）
    #   params     : 模型可训练参数总量
    #   flops      : 模型推理浮点运算量
    base_acc = float(base.get("best_acc", 0.0))
    comp_acc = float(comp.get("best_acc", 0.0))
    base_params = int(base["params"])
    comp_params = int(comp["params"])
    base_flops = int(base["flops"])
    comp_flops = int(comp["flops"])

    # ── 构建对比结果字典 ────────────────────────────────────────────
    result = {
        # 模型标识
        "baseline_run": base.get("run_name"),
        "compressed_run": comp.get("run_name"),

        # 各阶段通道配置（用于宽度剪枝的参考信息）
        "baseline_channels": base.get("stage_channels"),
        "compressed_channels": comp.get("stage_channels"),

        # 准确率对比
        "baseline_best_acc": base_acc,
        "compressed_best_acc": comp_acc,
        "accuracy_drop": base_acc - comp_acc,          # 精度下降量（正数表示有损失）

        # 参数量对比
        "baseline_params": base_params,
        "compressed_params": comp_params,
        "params_compression_rate": 1 - comp_params / base_params,  # 参数压缩率（0~1）

        # 计算量对比
        "baseline_flops": base_flops,
        "compressed_flops": comp_flops,
        "flops_reduction_rate": 1 - comp_flops / base_flops,      # FLOPs 缩减率（0~1）

        # 训练耗时对比（秒）
        "baseline_train_time_sec": base.get("total_train_time_sec"),
        "compressed_train_time_sec": comp.get("total_train_time_sec"),
    }

    # ── 输出结果 ────────────────────────────────────────────────────
    # 先打印到终端
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 如果指定了 --output，则写入文件
    if args.output:
        # 自动创建输出目录（如父目录不存在）
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
