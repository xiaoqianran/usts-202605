#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import get_cifar10_train_val_loaders
from src.models import block_width_resnet32, BASELINE_BLOCK_CHANNELS
from src.utils.accelerate import (
    autocast_context,
    make_grad_scaler,
    maybe_channels_last,
    maybe_compile,
    move_images,
    setup_torch_fast,
)
from src.utils.checkpoint import load_sliced_baseline_weights
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed

Candidate = Tuple[int, ...]

# ========================
# 15-D block-level search space（与第一版主项目保持一致，恢复 aggressive 版本）
# ========================
# 每个 block 独立一个通道数，共 15 维。
# 为了与第一版一致，这里使用 aggressive space（包含 8）。
# 注意：15 维搜索对精度非常敏感，包含小通道后建议适当放宽 allowed_short_acc_drop 或降低 penalty。
DEFAULT_STAGE_SPACE: list[list[int]] = [
    [8, 12, 16],                   # stage1 每个 block 可选
    [16, 20, 24, 28, 32],          # stage2 每个 block 可选
    [32, 40, 48, 56, 64],          # stage3 每个 block 可选
]
# 自动扩展为 15 维（每 stage 5 个 block）
DEFAULT_SPACE = [DEFAULT_STAGE_SPACE[0] for _ in range(5)] + [DEFAULT_STAGE_SPACE[1] for _ in range(5)] + [DEFAULT_STAGE_SPACE[2] for _ in range(5)]
BASELINE_CHANNELS: Candidate = tuple(BASELINE_BLOCK_CHANNELS)


@dataclass
class EvalResult:
    channels: list[int]
    fitness: float
    val_acc: float
    train_acc: float
    val_loss: float
    train_loss: float
    params: int
    flops: int
    params_ratio: float
    flops_ratio: float
    params_compression_rate: float
    flops_reduction_rate: float
    below_baseline_short_by: float
    penalty: float
    eval_time_sec: float
    epochs: int
    inherited: bool


def parse_space(s: str | None) -> list[list[int]]:
    if not s:
        return DEFAULT_SPACE
    groups = []
    for group in s.split(";"):
        vals = [int(x.strip()) for x in group.split(",") if x.strip()]
        if not vals:
            raise argparse.ArgumentTypeError("empty group in search space")
        groups.append(vals)
    if len(groups) == 3:
        # Compact form: stage1;stage2;stage3, expanded to 5 blocks per stage.
        return [groups[0] for _ in range(5)] + [groups[1] for _ in range(5)] + [groups[2] for _ in range(5)]
    if len(groups) == 15:
        return groups
    raise argparse.ArgumentTypeError("space must have either 3 stage groups or 15 block groups separated by ';'")


def candidate_to_key(channels: Sequence[int]) -> str:
    return "-".join(map(str, channels))


def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(fields)).writeheader()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def random_candidate(space: list[list[int]], rng: random.Random) -> Candidate:
    return tuple(rng.choice(g) for g in space)  # type: ignore[return-value]


def repair_candidate(channels: Sequence[int], space: list[list[int]]) -> Candidate:
    return tuple(min(g, key=lambda x: abs(x - int(round(v)))) for v, g in zip(channels, space))  # type: ignore[return-value]


def decode_position(position: Sequence[float], space: list[list[int]]) -> Candidate:
    out = []
    for p, g in zip(position, space):
        idx = max(0, min(len(g) - 1, int(round(p))))
        out.append(g[idx])
    return tuple(out)  # type: ignore[return-value]


def model_cost(channels: Sequence[int], device: torch.device) -> tuple[int, int]:
    m = block_width_resnet32(block_channels=channels, num_classes=10).to(device)
    params = count_parameters(m)
    flops = measure_flops(m, input_size=(3, 32, 32), device=device)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return params, flops


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool, amp_dtype: str, channels_last: bool):
    model.train()
    losses, top1 = AverageMeter(), AverageMeter()
    scaler = make_grad_scaler(device, use_amp, amp_dtype)
    for images, targets in tqdm(loader, desc="candidate train", leave=False):
        images = move_images(images, device, channels_last)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp, amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, amp_dtype: str, channels_last: bool):
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()
    for images, targets in tqdm(loader, desc="candidate val", leave=False):
        images = move_images(images, device, channels_last)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, use_amp, amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))
    return losses.avg, top1.avg


class FitnessEvaluator:
    def __init__(self, args, space: list[list[int]], out_dir: Path, device: torch.device) -> None:
        self.args = args
        self.space = space
        self.out_dir = out_dir
        self.device = device
        self.use_amp = bool(args.amp and device.type == "cuda")
        self.cache: Dict[str, EvalResult] = {}
        self.eval_csv = out_dir / "evaluations.csv"

        self.train_loader, self.val_loader = get_cifar10_train_val_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
        )
        self.baseline_params, self.baseline_flops = model_cost(BASELINE_CHANNELS, device)
        self.baseline_short = self._evaluate_raw_candidate(BASELINE_CHANNELS, cache=False)
        self.baseline_short_acc = self.baseline_short.val_acc

        fields = list(EvalResult.__dataclass_fields__.keys())
        write_csv_header_if_needed(self.eval_csv, ["channels", *[f for f in fields if f != "channels"]])

    def _make_model(self, channels: Sequence[int]):
        model = block_width_resnet32(block_channels=channels, num_classes=10).to(self.device)
        inherited = False
        if self.args.baseline_ckpt and Path(self.args.baseline_ckpt).exists():
            load_sliced_baseline_weights(model, self.args.baseline_ckpt, verbose=False)
            inherited = True
        model = maybe_channels_last(model, self.args.channels_last)
        return model, inherited

    def _fitness(self, val_acc: float, params_ratio: float, flops_ratio: float) -> tuple[float, float, float]:
        # Constraint is relative to same-budget short baseline, not full 200-epoch baseline.
        threshold = self.baseline_short_acc - self.args.allowed_short_acc_drop
        below = max(0.0, threshold - val_acc)
        penalty = self.args.penalty_mu * below
        fitness = val_acc - 100.0 * (self.args.lambda_params * params_ratio + self.args.lambda_flops * flops_ratio) - penalty
        return fitness, below, penalty

    def _evaluate_raw_candidate(self, channels: Sequence[int], cache: bool = True) -> EvalResult:
        channels = repair_candidate(channels, self.space)
        key = candidate_to_key(channels)
        if cache and key in self.cache:
            return self.cache[key]

        start = time.time()
        local_seed = self.args.seed + sum((i + 1) * c for i, c in enumerate(channels))
        set_seed(local_seed, deterministic=False)

        model, inherited = self._make_model(channels)
        train_model = maybe_compile(model, self.args.compile)
        params, flops = count_parameters(model), measure_flops(model, input_size=(3, 32, 32), device=self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.args.search_epochs, 1))

        train_loss = train_acc = math.nan
        for _ in range(self.args.search_epochs):
            train_loss, train_acc = train_one_epoch(train_model, self.train_loader, criterion, optimizer, self.device, self.use_amp, self.args.amp_dtype, self.args.channels_last)
            scheduler.step()
        val_loss, val_acc = evaluate(train_model, self.val_loader, criterion, self.device, self.use_amp, self.args.amp_dtype, self.args.channels_last)

        params_ratio = params / self.baseline_params
        flops_ratio = flops / self.baseline_flops
        fitness, below, penalty = self._fitness(val_acc, params_ratio, flops_ratio)
        res = EvalResult(
            channels=list(channels), fitness=float(fitness), val_acc=float(val_acc), train_acc=float(train_acc),
            val_loss=float(val_loss), train_loss=float(train_loss), params=int(params), flops=int(flops),
            params_ratio=float(params_ratio), flops_ratio=float(flops_ratio),
            params_compression_rate=float(1 - params_ratio), flops_reduction_rate=float(1 - flops_ratio),
            below_baseline_short_by=float(below), penalty=float(penalty), eval_time_sec=float(time.time() - start),
            epochs=int(self.args.search_epochs), inherited=bool(inherited),
        )
        if cache:
            self.cache[key] = res
            row = asdict(res); row["channels"] = key
            append_csv(self.eval_csv, row)
        del model, train_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    def evaluate_candidate(self, channels: Sequence[int]) -> EvalResult:
        return self._evaluate_raw_candidate(channels, cache=True)


def tournament_select(population: list[Candidate], scores: dict[Candidate, float], rng: random.Random, k: int = 3) -> Candidate:
    return max([rng.choice(population) for _ in range(k)], key=lambda c: scores[c])


def run_ga(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path) -> EvalResult:
    rng = random.Random(args.seed + 101)
    history_csv = out_dir / "ga_history.csv"
    write_csv_header_if_needed(history_csv, ["generation", "best_channels", "best_fitness", "best_val_acc", "best_params_ratio", "best_flops_ratio"])
    population, seen = [], set()
    while len(population) < args.ga_population:
        cand = random_candidate(space, rng)
        if cand not in seen:
            population.append(cand); seen.add(cand)
    global_best = None
    for gen in range(args.ga_generations):
        print(f"\n[GA] Generation {gen + 1}/{args.ga_generations}")
        results = {cand: evaluator.evaluate_candidate(cand) for cand in population}
        scores = {cand: r.fitness for cand, r in results.items()}
        gen_best = max(results.values(), key=lambda r: r.fitness)
        if global_best is None or gen_best.fitness > global_best.fitness:
            global_best = gen_best
        append_csv(history_csv, {"generation": gen + 1, "best_channels": candidate_to_key(gen_best.channels), "best_fitness": f"{gen_best.fitness:.6f}", "best_val_acc": f"{gen_best.val_acc:.4f}", "best_params_ratio": f"{gen_best.params_ratio:.6f}", "best_flops_ratio": f"{gen_best.flops_ratio:.6f}"})
        print(f"[GA] best={candidate_to_key(gen_best.channels)} fit={gen_best.fitness:.3f} val={gen_best.val_acc:.2f}% params↓={gen_best.params_compression_rate:.1%} flops↓={gen_best.flops_reduction_rate:.1%}")
        elites = sorted(population, key=lambda c: scores[c], reverse=True)[:max(1, args.ga_elites)]
        next_pop = list(elites)
        while len(next_pop) < args.ga_population:
            p1, p2 = tournament_select(population, scores, rng), tournament_select(population, scores, rng)
            child = tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(len(space))) if rng.random() < args.ga_crossover_rate else p1
            child = list(child)
            for i, group in enumerate(space):
                if rng.random() < args.ga_mutation_rate:
                    child[i] = rng.choice(group)
            next_pop.append(repair_candidate(child, space))
        population = next_pop
    assert global_best is not None
    (out_dir / "ga_best.json").write_text(json.dumps(asdict(global_best), ensure_ascii=False, indent=2), encoding="utf-8")
    return global_best


def run_pso(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path) -> EvalResult:
    rng = random.Random(args.seed + 202)
    history_csv = out_dir / "pso_history.csv"
    write_csv_header_if_needed(history_csv, ["iteration", "best_channels", "best_fitness", "best_val_acc", "best_params_ratio", "best_flops_ratio"])
    dims, upper = len(space), [len(g) - 1 for g in space]
    positions = [[rng.uniform(0, upper[d]) for d in range(dims)] for _ in range(args.pso_particles)]
    velocities = [[0.0] * dims for _ in range(args.pso_particles)]
    pbest_pos = [p[:] for p in positions]
    pbest_scores = [-float("inf")] * args.pso_particles
    gbest_pos, gbest = None, None
    for it in range(args.pso_iterations):
        print(f"\n[PSO] Iteration {it + 1}/{args.pso_iterations}")
        for i in range(args.pso_particles):
            res = evaluator.evaluate_candidate(decode_position(positions[i], space))
            if res.fitness > pbest_scores[i]:
                pbest_scores[i] = res.fitness; pbest_pos[i] = positions[i][:]
            if gbest is None or res.fitness > gbest.fitness:
                gbest, gbest_pos = res, positions[i][:]
        assert gbest is not None and gbest_pos is not None
        append_csv(history_csv, {"iteration": it + 1, "best_channels": candidate_to_key(gbest.channels), "best_fitness": f"{gbest.fitness:.6f}", "best_val_acc": f"{gbest.val_acc:.4f}", "best_params_ratio": f"{gbest.params_ratio:.6f}", "best_flops_ratio": f"{gbest.flops_ratio:.6f}"})
        print(f"[PSO] best={candidate_to_key(gbest.channels)} fit={gbest.fitness:.3f} val={gbest.val_acc:.2f}% params↓={gbest.params_compression_rate:.1%} flops↓={gbest.flops_reduction_rate:.1%}")
        for i in range(args.pso_particles):
            for d in range(dims):
                velocities[i][d] = args.pso_w * velocities[i][d] + args.pso_c1 * rng.random() * (pbest_pos[i][d] - positions[i][d]) + args.pso_c2 * rng.random() * (gbest_pos[d] - positions[i][d])
                positions[i][d] = max(0.0, min(float(upper[d]), positions[i][d] + velocities[i][d]))
    (out_dir / "pso_best.json").write_text(json.dumps(asdict(gbest), ensure_ascii=False, indent=2), encoding="utf-8")
    return gbest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast GA/PSO 15-D block-channel search for CIFAR-10 ResNet32 compression")
    parser.add_argument("--algorithm", choices=["ga", "pso", "both"], default="both")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--space", default=None)
    parser.add_argument("--search-epochs", type=int, default=1, help="with weight inheritance, 1 epoch is usually enough for proxy ranking; use 0 for pure inherited-weight evaluation")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--baseline-ckpt", default="runs/resnet32_baseline/best.pt")
    parser.add_argument("--lambda-params", type=float, default=0.05)
    parser.add_argument("--lambda-flops", type=float, default=0.10)
    parser.add_argument("--allowed-short-acc-drop", type=float, default=3.0)
    parser.add_argument("--penalty-mu", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="may be slower during search because every candidate compiles separately")
    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=5)
    parser.add_argument("--ga-elites", type=int, default=2)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.10)
    parser.add_argument("--pso-particles", type=int, default=8)
    parser.add_argument("--pso-iterations", type=int, default=5)
    parser.add_argument("--pso-w", type=float, default=0.6)
    parser.add_argument("--pso-c1", type=float, default=1.4)
    parser.add_argument("--pso-c2", type=float, default=1.4)
    args = parser.parse_args()

    setup_torch_fast(tf32=True, benchmark=True)
    set_seed(args.seed, deterministic=False)
    space = parse_space(args.space)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_name = args.run_name or f"block_channel_search_fast_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "search_config.json").write_text(json.dumps({**vars(args), "search_space": space, "baseline_block_channels": list(BASELINE_CHANNELS), "vector_dim": len(space)}, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluator = FitnessEvaluator(args, space, out_dir, device)
    print("Baseline block channels:", candidate_to_key(BASELINE_CHANNELS))
    print(f"Baseline params: {evaluator.baseline_params} ({human_number(evaluator.baseline_params)})")
    print(f"Baseline FLOPs : {evaluator.baseline_flops} ({human_number(evaluator.baseline_flops)})")
    print(f"Baseline-short val acc: {evaluator.baseline_short_acc:.2f}%")
    print("Search space dimension:", len(space))
    print("Search space:", space)

    total_start = time.time(); best_results = {}
    if args.algorithm in {"ga", "both"}:
        best_results["ga"] = asdict(run_ga(args, evaluator, space, out_dir))
    if args.algorithm in {"pso", "both"}:
        best_results["pso"] = asdict(run_pso(args, evaluator, space, out_dir))
    best_algorithm, best = max(best_results.items(), key=lambda item: item[1]["fitness"])
    summary = {"best_algorithm": best_algorithm, "best": best, "all_best": best_results, "baseline_short": asdict(evaluator.baseline_short), "baseline": {"channels": list(BASELINE_CHANNELS), "params": evaluator.baseline_params, "flops": evaluator.baseline_flops}, "num_unique_evaluations": len(evaluator.cache), "total_search_time_sec": time.time() - total_start}
    (out_dir / "best_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    best_ch = best["channels"]
    cmd = (
        "python train_block_width_resnet32.py "
        "--block-channels {} "
        "--run-name final_block_{}_{}_kd "
        "--epochs 80 --milestones 40,60 "
        "--batch-size 1024 --num-workers 8 "
        "--amp --amp-dtype bf16 --channels-last "
        "--baseline-ckpt {} "
        "--teacher-ckpt {} "
        "--kd-mode logits --kd-alpha 0.7 --kd-temperature 4.0"
    ).format(
        ",".join(map(str, best_ch)),
        best_algorithm,
        candidate_to_key(best_ch),
        args.baseline_ckpt,
        args.baseline_ckpt,
    )
    (out_dir / "final_train_command.txt").write_text(cmd + "\n", encoding="utf-8")
    print("\nSearch done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFinal training command:")
    print(cmd)


if __name__ == "__main__":
    main()
