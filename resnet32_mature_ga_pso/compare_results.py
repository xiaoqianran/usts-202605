#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline ResNet32 and compressed BlockWidthResNet32 summaries")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--search-result", default=None, help="optional best_result.json from search")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    base = load_json(args.baseline)
    comp = load_json(args.compressed)
    base_acc = float(base.get("best_acc", 0.0))
    comp_acc = float(comp.get("best_acc", 0.0))
    base_params, comp_params = int(base["params"]), int(comp["params"])
    base_flops, comp_flops = int(base["flops"]), int(comp["flops"])
    result = {
        "baseline_run": base.get("run_name"),
        "compressed_run": comp.get("run_name"),
        "baseline_channels": base.get("stage_channels", [16, 32, 64]),
        "compressed_block_channels": comp.get("block_channels"),
        "baseline_best_acc": base_acc,
        "compressed_best_acc": comp_acc,
        "accuracy_drop": base_acc - comp_acc,
        "baseline_params": base_params,
        "compressed_params": comp_params,
        "params_compression_rate": 1 - comp_params / base_params,
        "baseline_flops": base_flops,
        "compressed_flops": comp_flops,
        "flops_reduction_rate": 1 - comp_flops / base_flops,
        "baseline_train_time_sec": base.get("total_train_time_sec"),
        "compressed_train_time_sec": comp.get("total_train_time_sec"),
        "kd_enabled": comp.get("kd_enabled"),
    }
    if args.search_result:
        search = load_json(args.search_result)
        result["search_best_algorithm"] = search.get("best_algorithm")
        result["search_time_sec"] = search.get("total_search_time_sec")
        result["search_num_unique_evaluations"] = search.get("num_unique_evaluations")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
