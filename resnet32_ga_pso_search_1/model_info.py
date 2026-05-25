#!/usr/bin/env python3
"""
模型信息查看工具：打印 ResNet32 的参数量和 FLOPs。

用途：
  - 查看标准 ResNet32（16-32-64）的参数量和计算量
  - 查看任意宽度 ResNet32 的参数量和计算量
  - 自动计算与标准模型相比的参数压缩率和计算量缩减率

用法示例：
  python model_info.py                      # 查看标准 ResNet32
  python model_info.py --channels 16,24,48  # 查看自定义宽度
  python model_info.py --channels 8,16,32   # 查看更窄的模型
"""

from __future__ import annotations

import argparse
import json

import torch

from src.models import resnet32, width_resnet32
from src.utils.metrics import count_parameters, human_number, measure_flops


def parse_channels(s: str) -> list[int]:
    """将字符串解析为长度为 3 的通道数列表。

    支持逗号或短横线分隔，例如：
        "16,24,48"  →  [16, 24, 48]
        "16-24-48"  →  [16, 24, 48]

    Args:
        s: 包含 3 个整数的字符串。

    Returns:
        3 个整数组成的列表。

    Raises:
        ArgumentTypeError: 如果整数数量不是 3 个。
    """
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "channels must contain 3 integers, e.g. 16,24,48"
        )
    return values


def main() -> None:
    """程序入口：构建模型 → 统计参数量和 FLOPs → 与标准模型对比 → 输出 JSON。"""

    # ── 1. 命令行参数 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Show Params/FLOPs for standard or variable-width ResNet32"
    )
    parser.add_argument(
        "--channels", type=parse_channels, default=[16, 32, 64],
        help="三个阶段的通道数，如 16,24,48；默认 16,32,64（标准版）"
    )
    args = parser.parse_args()

    # ── 2. 设备选择 ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 3. 判断是标准模型还是可变宽度模型 ──────────────────────────
    # 标准 ResNet32 的通道数固定为 [16, 32, 64]
    # 如果用户指定的恰好是这个值，就用标准版本；否则用可变宽度版本
    is_standard = args.channels == [16, 32, 64]

    # 根据判断结果选择对应的模型构建函数
    model = (
        resnet32(num_classes=10).to(device)
        if is_standard
        else width_resnet32(stage_channels=args.channels, num_classes=10).to(device)
    )

    # ── 4. 统计当前模型的参数量和 FLOPs ───────────────────────────
    params = count_parameters(model)                                   # 可训练参数总数
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)  # 单次前向推理的浮点运算量

    # ── 5. 组装模型信息字典 ────────────────────────────────────────
    info = {
        "model": "ResNet32" if is_standard else "WidthResNet32",
        "dataset": "CIFAR-10",
        "stage_channels": args.channels,       # 三阶段通道配置
        "blocks_per_stage": 5,                 # 每个阶段的残差块数量
        "params": params,                      # 参数量（精确数字）
        "params_human": human_number(params),  # 参数量（人类可读格式，如 "460K"）
        "flops": flops,                        # FLOPs（精确数字）
        "flops_human": human_number(flops),    # FLOPs（人类可读格式，如 "70M"）
        # 以下两项仅在非标准模型时计算，标准模型自身对比无意义，设为 None
        "params_compression_rate_vs_16_32_64": None,   # 参数压缩率
        "flops_reduction_rate_vs_16_32_64": None,      # 计算量缩减率
    }

    # ── 6. 非标准模型：额外计算与标准模型的对比指标 ────────────────
    if not is_standard:
        # 构建标准 ResNet32 作为对比基线
        base = resnet32(num_classes=10).to(device)
        base_params = count_parameters(base)
        base_flops = measure_flops(base, input_size=(3, 32, 32), device=device)

        # 压缩率 = 1 - 当前参数量 / 基线参数量
        # 例如：基线 460K 参数，当前 200K 参数 → 压缩率 = 1 - 200/460 ≈ 0.565（压缩了 56.5%）
        info["params_compression_rate_vs_16_32_64"] = 1 - params / base_params

        # 计算量缩减率同理
        info["flops_reduction_rate_vs_16_32_64"] = 1 - flops / base_flops

    # ── 7. 输出结果 ────────────────────────────────────────────────
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
