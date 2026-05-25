#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from src.models import resnet32, width_resnet32
from src.utils.metrics import count_parameters, human_number, measure_flops


def parse_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("channels must contain 3 integers, e.g. 16,24,48")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Params/FLOPs for standard or variable-width ResNet32")
    parser.add_argument("--channels", type=parse_channels, default=[16, 32, 64], help="stage channels, e.g. 16,24,48")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_standard = args.channels == [16, 32, 64]
    model = resnet32(num_classes=10).to(device) if is_standard else width_resnet32(stage_channels=args.channels, num_classes=10).to(device)
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    info = {
        "model": "ResNet32" if is_standard else "WidthResNet32",
        "dataset": "CIFAR-10",
        "stage_channels": args.channels,
        "blocks_per_stage": 5,
        "params": params,
        "params_human": human_number(params),
        "flops": flops,
        "flops_human": human_number(flops),
        "params_compression_rate_vs_16_32_64": None,
        "flops_reduction_rate_vs_16_32_64": None,
    }
    if not is_standard:
        base = resnet32(num_classes=10).to(device)
        base_params = count_parameters(base)
        base_flops = measure_flops(base, input_size=(3, 32, 32), device=device)
        info["params_compression_rate_vs_16_32_64"] = 1 - params / base_params
        info["flops_reduction_rate_vs_16_32_64"] = 1 - flops / base_flops
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
