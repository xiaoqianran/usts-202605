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

from src.data import get_cifar10_loaders
from src.models import DEFAULT_BLOCK_CHANNELS, block_width_resnet32, resnet32
from src.search import candidate_key, default_block_search_space, load_block_importance_from_baseline
from src.utils.metrics import AverageMeter, accuracy, count_parameters, human_number, measure_flops
from src.utils.seed import set_seed

Candidate = Tuple[int, ...]


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


def write_csv_header_if_needed(path: Path, fields: Iterable[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(fields)).writeheader()


def append_csv(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)


def parse_space(s: str | None) -> list[list[int]]:
    if not s:
        return default_block_search_space()
    groups = []
    for group in s.split(";"):
        values = [int(x.strip()) for x in group.split(",") if x.strip()]
        groups.append(values)
    if len(groups) == 3:
        return [groups[0] for _ in range(5)] + [groups[1] for _ in range(5)] + [groups[2] for _ in range(5)]
    if len(groups) == 15:
        return groups
    raise argparse.ArgumentTypeError("space must have either 3 stage groups or 15 block groups")


def repair_candidate(channels: Sequence[int], space: list[list[int]]) -> Candidate:
    fixed = []
    for value, group in zip(channels, space):
        fixed.append(min(group, key=lambda x: abs(x - int(round(value)))))
    return tuple(fixed)


def random_candidate(space: list[list[int]], rng: random.Random, importance: list[float] | None = None) -> Candidate:
    values = []
    for i, group in enumerate(space):
        if importance is None:
            values.append(rng.choice(group))
        else:
            # Importance-guided sampling: important blocks are biased toward larger widths.
            # low importance -> more compression; high importance -> less compression.
            imp = importance[i]
            probs = []
            for rank, _ in enumerate(group):
                normalized_rank = rank / max(len(group) - 1, 1)
                # rank high means larger channel count. Bias by importance.
                weight = 0.25 + (1 - abs(normalized_rank - imp))
                probs.append(weight)
            total = sum(probs)
            r = rng.random() * total
            acc = 0.0
            chosen = group[-1]
            for g, p in zip(group, probs):
                acc += p
                if r <= acc:
                    chosen = g
                    break
            values.append(chosen)
    return tuple(values)


def decode_position(position: Sequence[float], space: list[list[int]]) -> Candidate:
    values = []
    for pos, group in zip(position, space):
        idx = int(round(pos))
        idx = max(0, min(len(group) - 1, idx))
        values.append(group[idx])
    return tuple(values)


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool) -> tuple[float, float]:
    model.train()
    losses, top1 = AverageMeter(), AverageMeter()
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
    losses, top1 = AverageMeter(), AverageMeter()
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
        base = resnet32(num_classes=10).to(device)
        self.baseline_params = count_parameters(base)
        self.baseline_flops = measure_flops(base, input_size=(3, 32, 32), device=device)
        del base
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fields = [
            "channels", "fitness", "test_acc", "train_acc", "test_loss", "train_loss",
            "params", "flops", "params_ratio", "flops_ratio", "params_compression_rate",
            "flops_reduction_rate", "eval_time_sec", "epochs",
        ]
        write_csv_header_if_needed(self.eval_csv, fields)

    def fitness_from_metrics(self, test_acc: float, params_ratio: float, flops_ratio: float, eval_time_sec: float) -> float:
        time_penalty = self.args.lambda_time * min(eval_time_sec / max(self.args.time_ref, 1e-6), 3.0)
        return test_acc - 100.0 * (self.args.lambda_params * params_ratio + self.args.lambda_flops * flops_ratio) - time_penalty

    def evaluate_candidate(self, channels: Sequence[int]) -> EvalResult:
        candidate = repair_candidate(channels, self.space)
        key = candidate_key(candidate)
        if key in self.cache:
            return self.cache[key]
        start = time.time()
        local_seed = self.args.seed + sum((i + 1) * c for i, c in enumerate(candidate))
        set_seed(local_seed, deterministic=False)
        model = block_width_resnet32(block_channels=candidate, num_classes=10).to(self.device)
        params = count_parameters(model)
        flops = measure_flops(model, input_size=(3, 32, 32), device=self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.args.search_epochs, 1))
        train_loss = train_acc = math.nan
        for _ in range(self.args.search_epochs):
            train_loss, train_acc = train_one_epoch(model, self.train_loader, criterion, optimizer, self.device, self.use_amp)
            scheduler.step()
        test_loss, test_acc = evaluate(model, self.test_loader, criterion, self.device)
        eval_time = time.time() - start
        params_ratio = params / self.baseline_params
        flops_ratio = flops / self.baseline_flops
        fitness = self.fitness_from_metrics(test_acc, params_ratio, flops_ratio, eval_time)
        result = EvalResult(
            channels=list(candidate), fitness=float(fitness), test_acc=float(test_acc), train_acc=float(train_acc),
            test_loss=float(test_loss), train_loss=float(train_loss), params=int(params), flops=int(flops),
            params_ratio=float(params_ratio), flops_ratio=float(flops_ratio),
            params_compression_rate=float(1 - params_ratio), flops_reduction_rate=float(1 - flops_ratio),
            eval_time_sec=float(eval_time), epochs=int(self.args.search_epochs),
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


def run_ga(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path, importance: list[float] | None) -> EvalResult:
    rng = random.Random(args.seed + 101)
    history_csv = out_dir / "ga_history.csv"
    write_csv_header_if_needed(history_csv, ["generation", "best_channels", "best_fitness", "best_acc", "params_compression_rate", "flops_reduction_rate"])
    population, seen = [], set()
    # Ensure baseline-like and a few compressed seeds are present.
    seeds = [tuple(DEFAULT_BLOCK_CHANNELS)]
    while len(population) < args.ga_population:
        cand = seeds.pop(0) if seeds else random_candidate(space, rng, importance)
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
            "generation": gen + 1, "best_channels": candidate_key(gen_best.channels),
            "best_fitness": f"{gen_best.fitness:.6f}", "best_acc": f"{gen_best.test_acc:.4f}",
            "params_compression_rate": f"{gen_best.params_compression_rate:.6f}",
            "flops_reduction_rate": f"{gen_best.flops_reduction_rate:.6f}",
        })
        print(f"[GA] best={candidate_key(gen_best.channels)} fitness={gen_best.fitness:.3f} acc={gen_best.test_acc:.2f}% params↓={gen_best.params_compression_rate:.2%} flops↓={gen_best.flops_reduction_rate:.2%}")
        elites = sorted(population, key=lambda c: scores[c], reverse=True)[:max(1, args.ga_elites)]
        next_pop = list(elites)
        while len(next_pop) < args.ga_population:
            p1 = tournament_select(population, scores, rng)
            p2 = tournament_select(population, scores, rng)
            if rng.random() < args.ga_crossover_rate:
                child = tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(15))
            else:
                child = p1
            child = list(child)
            for i, group in enumerate(space):
                if rng.random() < args.ga_mutation_rate:
                    if importance is not None and rng.random() < 0.65:
                        # Importance-guided mutation near the importance rank.
                        imp = importance[i]
                        target_idx = round(imp * (len(group) - 1))
                        choices = [max(0, min(len(group) - 1, target_idx + delta)) for delta in (-1, 0, 1)]
                        child[i] = group[rng.choice(choices)]
                    else:
                        child[i] = rng.choice(group)
            next_pop.append(repair_candidate(child, space))
        population = next_pop
    assert global_best is not None
    with (out_dir / "ga_best.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(global_best), f, ensure_ascii=False, indent=2)
    return global_best


def run_pso(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path, importance: list[float] | None) -> EvalResult:
    rng = random.Random(args.seed + 202)
    history_csv = out_dir / "pso_history.csv"
    write_csv_header_if_needed(history_csv, ["iteration", "best_channels", "best_fitness", "best_acc", "params_compression_rate", "flops_reduction_rate"])
    dims = 15
    upper = [len(group) - 1 for group in space]
    positions = []
    for _ in range(args.pso_particles):
        pos = []
        for d in range(dims):
            if importance is None:
                pos.append(rng.uniform(0, upper[d]))
            else:
                pos.append(max(0.0, min(float(upper[d]), importance[d] * upper[d] + rng.uniform(-0.8, 0.8))))
        positions.append(pos)
    velocities = [[0.0] * dims for _ in range(args.pso_particles)]
    pbest_positions = [p[:] for p in positions]
    pbest_scores = [-float("inf")] * args.pso_particles
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
            if gbest_result is None or res.fitness > gbest_result.fitness:
                gbest_result = res
                gbest_position = positions[i][:]
        assert gbest_result is not None and gbest_position is not None
        append_csv(history_csv, {
            "iteration": it + 1, "best_channels": candidate_key(gbest_result.channels),
            "best_fitness": f"{gbest_result.fitness:.6f}", "best_acc": f"{gbest_result.test_acc:.4f}",
            "params_compression_rate": f"{gbest_result.params_compression_rate:.6f}",
            "flops_reduction_rate": f"{gbest_result.flops_reduction_rate:.6f}",
        })
        print(f"[PSO] best={candidate_key(gbest_result.channels)} fitness={gbest_result.fitness:.3f} acc={gbest_result.test_acc:.2f}% params↓={gbest_result.params_compression_rate:.2%} flops↓={gbest_result.flops_reduction_rate:.2%}")
        for i in range(args.pso_particles):
            for d in range(dims):
                r1, r2 = rng.random(), rng.random()
                velocities[i][d] = args.pso_w * velocities[i][d] + args.pso_c1 * r1 * (pbest_positions[i][d] - positions[i][d]) + args.pso_c2 * r2 * (gbest_position[d] - positions[i][d])
                positions[i][d] = max(0.0, min(float(upper[d]), positions[i][d] + velocities[i][d]))
    assert gbest_result is not None
    with (out_dir / "pso_best.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(gbest_result), f, ensure_ascii=False, indent=2)
    return gbest_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Importance-guided GA/PSO block-level channel search for ResNet32")
    parser.add_argument("--algorithm", choices=["ga", "pso", "both"], default="both")
    parser.add_argument("--baseline-checkpoint", default=None, help="optional best.pt for BN-gamma importance guidance")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default="block_channel_search_ga_pso")
    parser.add_argument("--space", default=None, help="3 groups or 15 groups; e.g. '8,12,16;16,20,24,28,32;32,40,48,56,64'")
    parser.add_argument("--search-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--lambda-params", type=float, default=0.10)
    parser.add_argument("--lambda-flops", type=float, default=0.15)
    parser.add_argument("--lambda-time", type=float, default=0.02)
    parser.add_argument("--time-ref", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=5)
    parser.add_argument("--ga-elites", type=int, default=2)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.15)
    parser.add_argument("--pso-particles", type=int, default=8)
    parser.add_argument("--pso-iterations", type=int, default=5)
    parser.add_argument("--pso-w", type=float, default=0.6)
    parser.add_argument("--pso-c1", type=float, default=1.4)
    parser.add_argument("--pso-c2", type=float, default=1.4)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    space = parse_space(args.space)
    importance = load_block_importance_from_baseline(args.baseline_checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.save_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "search_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "search_space": space, "importance": importance}, f, ensure_ascii=False, indent=2)
    evaluator = FitnessEvaluator(args, space, out_dir, device)
    print(f"Baseline params={human_number(evaluator.baseline_params)} flops={human_number(evaluator.baseline_flops)}")
    print("Importance guidance:", "enabled" if importance is not None else "disabled")

    total_start = time.time()
    results = {}
    if args.algorithm in {"ga", "both"}:
        results["ga"] = asdict(run_ga(args, evaluator, space, out_dir, importance))
    if args.algorithm in {"pso", "both"}:
        results["pso"] = asdict(run_pso(args, evaluator, space, out_dir, importance))
    best_algo, best = max(results.items(), key=lambda item: item[1]["fitness"])
    summary = {
        "best_algorithm": best_algo,
        "best": best,
        "baseline": {"block_channels": DEFAULT_BLOCK_CHANNELS, "params": evaluator.baseline_params, "flops": evaluator.baseline_flops},
        "num_unique_evaluations": len(evaluator.cache),
        "total_search_time_sec": time.time() - total_start,
    }
    with (out_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nSearch done:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ch = ",".join(map(str, best["channels"]))
    print("\nNext command:")
    print(f"python train_block_resnet32_kd.py --block-channels {ch} --teacher-checkpoint runs/resnet32_baseline/best.pt --run-name final_block_{candidate_key(best['channels'])} --epochs 80 --milestones 40,60 --amp")


if __name__ == "__main__":
    main()
