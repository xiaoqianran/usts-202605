#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import get_cifar10_loaders
from src.models import width_resnet32
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed

Candidate = Tuple[int, int, int]


SEARCH_SPACE: list[list[int]] = [
    [8, 12, 16],
    [16, 20, 24, 28, 32],
    [32, 40, 48, 56, 64],
]
BASELINE_CHANNELS: Candidate = (16, 32, 64)


@dataclass
class EvalResult:
    channels: list[int]
    fitness: float
    test_acc: float
    train_acc: float
    test_loss: float
    train_loss: float
    params: int
    flops: int
    params_ratio: float
    flops_ratio: float
    params_compression_rate: float
    flops_reduction_rate: float
    eval_time_sec: float
    epochs: int


def candidate_to_key(channels: Sequence[int]) -> str:
    return "-".join(map(str, channels))


def parse_space(s: str | None) -> list[list[int]]:
    if not s:
        return SEARCH_SPACE
    # Format: "8,12,16;16,20,24,28,32;32,40,48,56,64"
    groups = []
    for group in s.split(";"):
        values = [int(x.strip()) for x in group.split(",") if x.strip()]
        if not values:
            raise argparse.ArgumentTypeError("empty group in search space")
        groups.append(values)
    if len(groups) != 3:
        raise argparse.ArgumentTypeError("search space must contain 3 groups separated by ';'")
    return groups


def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def random_candidate(space: list[list[int]], rng: random.Random) -> Candidate:
    return tuple(rng.choice(group) for group in space)  # type: ignore[return-value]


def repair_candidate(channels: Sequence[int], space: list[list[int]]) -> Candidate:
    fixed = []
    for value, group in zip(channels, space):
        # Snap to the closest allowed value in each group.
        fixed.append(min(group, key=lambda x: abs(x - int(round(value)))))
    return tuple(fixed)  # type: ignore[return-value]


def decode_position(position: Sequence[float], space: list[list[int]]) -> Candidate:
    decoded = []
    for pos, group in zip(position, space):
        idx = int(round(pos))
        idx = max(0, min(len(group) - 1, idx))
        decoded.append(group[idx])
    return tuple(decoded)  # type: ignore[return-value]


def conv_model_info(channels: Sequence[int], device: torch.device) -> tuple[int, int]:
    model = width_resnet32(stage_channels=channels, num_classes=10).to(device)
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return params, flops


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool) -> tuple[float, float]:
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in tqdm(loader, desc="candidate train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
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
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    for images, targets in tqdm(loader, desc="candidate eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
        self.rng = random.Random(args.seed)
        self.cache: Dict[str, EvalResult] = {}
        self.eval_csv = out_dir / "evaluations.csv"

        self.train_loader, self.test_loader = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )

        self.baseline_params, self.baseline_flops = conv_model_info(BASELINE_CHANNELS, device)
        fields = [
            "channels", "fitness", "test_acc", "train_acc", "test_loss", "train_loss",
            "params", "flops", "params_ratio", "flops_ratio",
            "params_compression_rate", "flops_reduction_rate", "eval_time_sec", "epochs",
        ]
        write_csv_header_if_needed(self.eval_csv, fields)

    def fitness_from_metrics(self, test_acc: float, params_ratio: float, flops_ratio: float) -> float:
        # Accuracy is in percent. Ratios are in [0, 1].
        # A lower Params/FLOPs ratio gets a smaller penalty.
        return test_acc - 100.0 * (self.args.lambda_params * params_ratio + self.args.lambda_flops * flops_ratio)

    def evaluate_candidate(self, channels: Sequence[int]) -> EvalResult:
        channels = repair_candidate(channels, self.space)
        key = candidate_to_key(channels)
        if key in self.cache:
            return self.cache[key]

        start = time.time()
        # Make candidate training deterministic enough for fair GA/PSO comparison.
        local_seed = self.args.seed + sum((i + 1) * c for i, c in enumerate(channels))
        set_seed(local_seed, deterministic=False)

        model = width_resnet32(stage_channels=channels, num_classes=10).to(self.device)
        params = count_parameters(model)
        flops = measure_flops(model, input_size=(3, 32, 32), device=self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.args.search_epochs, 1))

        train_loss = math.nan
        train_acc = math.nan
        for _ in range(self.args.search_epochs):
            train_loss, train_acc = train_one_epoch(model, self.train_loader, criterion, optimizer, self.device, self.use_amp)
            scheduler.step()
        test_loss, test_acc = evaluate(model, self.test_loader, criterion, self.device)

        params_ratio = params / self.baseline_params
        flops_ratio = flops / self.baseline_flops
        fitness = self.fitness_from_metrics(test_acc, params_ratio, flops_ratio)
        result = EvalResult(
            channels=list(channels),
            fitness=float(fitness),
            test_acc=float(test_acc),
            train_acc=float(train_acc),
            test_loss=float(test_loss),
            train_loss=float(train_loss),
            params=int(params),
            flops=int(flops),
            params_ratio=float(params_ratio),
            flops_ratio=float(flops_ratio),
            params_compression_rate=float(1.0 - params_ratio),
            flops_reduction_rate=float(1.0 - flops_ratio),
            eval_time_sec=float(time.time() - start),
            epochs=int(self.args.search_epochs),
        )
        self.cache[key] = result
        append_csv(self.eval_csv, {**asdict(result), "channels": key})

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def tournament_select(population: list[Candidate], scores: dict[Candidate, float], rng: random.Random, k: int = 3) -> Candidate:
    contenders = [rng.choice(population) for _ in range(k)]
    return max(contenders, key=lambda c: scores[c])


def run_ga(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path) -> EvalResult:
    rng = random.Random(args.seed + 101)
    history_csv = out_dir / "ga_history.csv"
    write_csv_header_if_needed(history_csv, ["generation", "best_channels", "best_fitness", "best_acc", "best_params_ratio", "best_flops_ratio"])

    population: list[Candidate] = []
    seen = set()
    while len(population) < args.ga_population:
        cand = random_candidate(space, rng)
        if cand not in seen:
            population.append(cand)
            seen.add(cand)

    global_best: EvalResult | None = None

    for gen in range(args.ga_generations):
        print(f"\n[GA] Generation {gen + 1}/{args.ga_generations}")
        results = {cand: evaluator.evaluate_candidate(cand) for cand in population}
        scores = {cand: res.fitness for cand, res in results.items()}
        gen_best = max(results.values(), key=lambda r: r.fitness)
        if global_best is None or gen_best.fitness > global_best.fitness:
            global_best = gen_best

        append_csv(history_csv, {
            "generation": gen + 1,
            "best_channels": candidate_to_key(gen_best.channels),
            "best_fitness": f"{gen_best.fitness:.6f}",
            "best_acc": f"{gen_best.test_acc:.4f}",
            "best_params_ratio": f"{gen_best.params_ratio:.6f}",
            "best_flops_ratio": f"{gen_best.flops_ratio:.6f}",
        })
        print(
            f"[GA] best={candidate_to_key(gen_best.channels)} "
            f"fitness={gen_best.fitness:.4f} acc={gen_best.test_acc:.2f}% "
            f"params↓={gen_best.params_compression_rate:.2%} flops↓={gen_best.flops_reduction_rate:.2%}"
        )

        # Elitism + tournament + uniform crossover + mutation.
        elite_count = max(1, args.ga_elites)
        elites = sorted(population, key=lambda c: scores[c], reverse=True)[:elite_count]
        next_population: list[Candidate] = list(elites)
        while len(next_population) < args.ga_population:
            p1 = tournament_select(population, scores, rng)
            p2 = tournament_select(population, scores, rng)
            if rng.random() < args.ga_crossover_rate:
                child = tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(3))
            else:
                child = p1
            child_list = list(child)
            for i, group in enumerate(space):
                if rng.random() < args.ga_mutation_rate:
                    child_list[i] = rng.choice(group)
            child = repair_candidate(child_list, space)
            next_population.append(child)
        population = next_population

    assert global_best is not None
    with (out_dir / "ga_best.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(global_best), f, ensure_ascii=False, indent=2)
    return global_best


def run_pso(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path) -> EvalResult:
    rng = random.Random(args.seed + 202)
    history_csv = out_dir / "pso_history.csv"
    write_csv_header_if_needed(history_csv, ["iteration", "best_channels", "best_fitness", "best_acc", "best_params_ratio", "best_flops_ratio"])

    dims = 3
    upper = [len(group) - 1 for group in space]
    positions = [[rng.uniform(0, upper[d]) for d in range(dims)] for _ in range(args.pso_particles)]
    velocities = [[0.0 for _ in range(dims)] for _ in range(args.pso_particles)]
    pbest_positions = [pos[:] for pos in positions]
    pbest_scores = [-float("inf") for _ in range(args.pso_particles)]
    pbest_results: list[EvalResult | None] = [None for _ in range(args.pso_particles)]
    gbest_position: list[float] | None = None
    gbest_result: EvalResult | None = None

    for it in range(args.pso_iterations):
        print(f"\n[PSO] Iteration {it + 1}/{args.pso_iterations}")
        for i in range(args.pso_particles):
            cand = decode_position(positions[i], space)
            res = evaluator.evaluate_candidate(cand)
            if res.fitness > pbest_scores[i]:
                pbest_scores[i] = res.fitness
                pbest_positions[i] = positions[i][:]
                pbest_results[i] = res
            if gbest_result is None or res.fitness > gbest_result.fitness:
                gbest_result = res
                gbest_position = positions[i][:]

        assert gbest_result is not None and gbest_position is not None
        append_csv(history_csv, {
            "iteration": it + 1,
            "best_channels": candidate_to_key(gbest_result.channels),
            "best_fitness": f"{gbest_result.fitness:.6f}",
            "best_acc": f"{gbest_result.test_acc:.4f}",
            "best_params_ratio": f"{gbest_result.params_ratio:.6f}",
            "best_flops_ratio": f"{gbest_result.flops_ratio:.6f}",
        })
        print(
            f"[PSO] best={candidate_to_key(gbest_result.channels)} "
            f"fitness={gbest_result.fitness:.4f} acc={gbest_result.test_acc:.2f}% "
            f"params↓={gbest_result.params_compression_rate:.2%} flops↓={gbest_result.flops_reduction_rate:.2%}"
        )

        for i in range(args.pso_particles):
            for d in range(dims):
                r1 = rng.random()
                r2 = rng.random()
                velocities[i][d] = (
                    args.pso_w * velocities[i][d]
                    + args.pso_c1 * r1 * (pbest_positions[i][d] - positions[i][d])
                    + args.pso_c2 * r2 * (gbest_position[d] - positions[i][d])
                )
                positions[i][d] += velocities[i][d]
                positions[i][d] = max(0.0, min(float(upper[d]), positions[i][d]))

    assert gbest_result is not None
    with (out_dir / "pso_best.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(gbest_result), f, ensure_ascii=False, indent=2)
    return gbest_result


def main() -> None:
    parser = argparse.ArgumentParser(description="GA/PSO stage-channel search for CIFAR-10 ResNet32 compression")
    parser.add_argument("--algorithm", choices=["ga", "pso", "both"], default="both")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default="channel_search_ga_pso")
    parser.add_argument("--space", default=None, help="e.g. '8,12,16;16,20,24,28,32;32,40,48,56,64'")
    parser.add_argument("--search-epochs", type=int, default=3, help="epochs used to score each candidate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=5000, help="0 means full train set")
    parser.add_argument("--max-test-samples", type=int, default=2000, help="0 means full test set")
    parser.add_argument("--lambda-params", type=float, default=0.15)
    parser.add_argument("--lambda-flops", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=5)
    parser.add_argument("--ga-elites", type=int, default=2)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.25)

    parser.add_argument("--pso-particles", type=int, default=8)
    parser.add_argument("--pso-iterations", type=int, default=5)
    parser.add_argument("--pso-w", type=float, default=0.6)
    parser.add_argument("--pso-c1", type=float, default=1.4)
    parser.add_argument("--pso-c2", type=float, default=1.4)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    space = parse_space(args.space)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    out_dir = Path(args.save_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["search_space"] = space
    config["baseline_channels"] = list(BASELINE_CHANNELS)
    config["model_family"] = "CIFAR-10 ResNet32, stage-level width search"
    with (out_dir / "search_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    evaluator = FitnessEvaluator(args, space, out_dir, device)
    print("Baseline channels:", candidate_to_key(BASELINE_CHANNELS))
    print(f"Baseline params: {evaluator.baseline_params} ({human_number(evaluator.baseline_params)})")
    print(f"Baseline FLOPs : {evaluator.baseline_flops} ({human_number(evaluator.baseline_flops)})")
    print("Search space:", space)
    print("Fitness = Acc - 100 * (lambda_params * ParamsRatio + lambda_flops * FLOPsRatio)")

    total_start = time.time()
    best_results = {}
    if args.algorithm in {"ga", "both"}:
        ga_best = run_ga(args, evaluator, space, out_dir)
        best_results["ga"] = asdict(ga_best)
    if args.algorithm in {"pso", "both"}:
        pso_best = run_pso(args, evaluator, space, out_dir)
        best_results["pso"] = asdict(pso_best)

    final_best_name, final_best = max(best_results.items(), key=lambda item: item[1]["fitness"])
    summary = {
        "algorithm": args.algorithm,
        "best_algorithm": final_best_name,
        "best": final_best,
        "baseline": {
            "channels": list(BASELINE_CHANNELS),
            "params": evaluator.baseline_params,
            "flops": evaluator.baseline_flops,
        },
        "num_unique_evaluations": len(evaluator.cache),
        "total_search_time_sec": time.time() - total_start,
    }
    with (out_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSearch done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nNext step: train the best channel configuration fully, for example:")
    best_channels = final_best["channels"]
    print(
        "python train_width_resnet32.py "
        f"--channels {','.join(map(str, best_channels))} "
        f"--run-name final_width_{candidate_to_key(best_channels)} "
        "--epochs 80 --milestones 40,60 --amp"
    )


if __name__ == "__main__":
    main()
