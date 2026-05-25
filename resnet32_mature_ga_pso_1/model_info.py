#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from src.models import DEFAULT_BLOCK_CHANNELS, block_width_resnet32, resnet32
from src.utils.metrics import count_parameters, human_number, measure_flops


def parse_block_channels(s: str) -> list[int]:
    values = [int(x.strip()) for x in s.replace("-", ",").split(",") if x.strip()]
    if len(values) != 15:
        raise argparse.ArgumentTypeError("block channels must contain 15 integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Params/FLOPs for baseline or block-width ResNet32")
    parser.add_argument("--block-channels", type=parse_block_channels, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.block_channels is None:
        model = resnet32(num_classes=10).to(device)
        name = "Standard ResNet32"
        channels = DEFAULT_BLOCK_CHANNELS
    else:
        model = block_width_resnet32(block_channels=args.block_channels, num_classes=10).to(device)
        name = "BlockWidthResNet32"
        channels = args.block_channels
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    info = {
        "model": name,
        "block_channels": channels,
        "params": params,
        "params_human": human_number(params),
        "flops": flops,
        "flops_human": human_number(flops),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
