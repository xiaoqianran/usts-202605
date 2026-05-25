#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from src.models import resnet32, width_resnet32, block_width_resnet32, BASELINE_BLOCK_CHANNELS
from src.utils.metrics import count_parameters, human_number, measure_flops


def parse_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("channels must contain 3 integers, e.g. 16,24,48")
    return values


def parse_block_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) == 3:
        return [values[0]] * 5 + [values[1]] * 5 + [values[2]] * 5
    if len(values) != 15:
        raise argparse.ArgumentTypeError("block channels must contain 15 integers, or 3 stage integers to expand")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Params/FLOPs for standard, stage-width, or block-width ResNet32")
    parser.add_argument("--channels", type=parse_channels, default=None, help="stage channels, e.g. 16,24,48")
    parser.add_argument("--block-channels", type=parse_block_channels, default=None, help="15 block channels, or 3 stage channels expanded to 15")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.block_channels is not None:
        model_name = "BlockWidthResNet32"
        ch = args.block_channels
        model = block_width_resnet32(block_channels=ch, num_classes=10).to(device)
        stage_channels = [ch[0], ch[5], ch[10]]
        block_channels = ch
    elif args.channels is not None:
        model_name = "WidthResNet32"
        ch = args.channels
        model = width_resnet32(stage_channels=ch, num_classes=10).to(device)
        stage_channels = ch
        block_channels = [ch[0]] * 5 + [ch[1]] * 5 + [ch[2]] * 5
    else:
        model_name = "ResNet32"
        model = resnet32(num_classes=10).to(device)
        stage_channels = [16, 32, 64]
        block_channels = BASELINE_BLOCK_CHANNELS

    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)

    base = resnet32(num_classes=10).to(device)
    base_params = count_parameters(base)
    base_flops = measure_flops(base, input_size=(3, 32, 32), device=device)

    info = {
        "model": model_name,
        "dataset": "CIFAR-10",
        "stage_channels": stage_channels,
        "block_channels": block_channels,
        "blocks_per_stage": 5,
        "params": params,
        "params_human": human_number(params),
        "flops": flops,
        "flops_human": human_number(flops),
        "params_compression_rate_vs_standard": 1 - params / base_params,
        "flops_reduction_rate_vs_standard": 1 - flops / base_flops,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
