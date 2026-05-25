#!/usr/bin/env python3
"""
基于遗传算法 (GA) 和粒子群优化 (PSO) 的 ResNet32 通道宽度搜索脚本。
在 CIFAR-10 上自动搜索最优的三阶段通道配置，平衡模型精度与参数量/FLOPs 的压缩。
"""
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

# 候选解的类型别名：三阶段通道宽度，如 (16, 32, 64)
Candidate = Tuple[int, int, int]


# ========================
# 搜索空间定义
# ========================
# 每个子列表对应 ResNet32 一个阶段（stage）可选的通道宽度。
# 三阶段分别对应浅层→中层→深层的特征通道数。
SEARCH_SPACE: list[list[int]] = [
    [8, 12, 16],                   # 第一阶段可选通道宽度
    [16, 20, 24, 28, 32],          # 第二阶段可选通道宽度
    [32, 40, 48, 56, 64],          # 第三阶段可选通道宽度
]
# 基线模型的标准通道配置，用于计算压缩率
BASELINE_CHANNELS: Candidate = (16, 32, 64)


@dataclass
class EvalResult:
    """单个候选模型的评估结果数据类，记录所有相关指标。"""
    channels: list[int]            # 三阶段通道宽度配置
    fitness: float                 # 适应度分数（综合考虑精度与模型开销）
    test_acc: float                # 测试集准确率 (%)
    train_acc: float               # 训练集准确率 (%)
    test_loss: float               # 测试集损失
    train_loss: float              # 训练集损失
    params: int                    # 模型参数量
    flops: int                     # 浮点运算量 (FLOPs)
    params_ratio: float            # 参数量相对于基线的比例
    flops_ratio: float             # FLOPs 相对于基线的比例
    params_compression_rate: float # 参数压缩率 (1 - params_ratio)
    flops_reduction_rate: float    # FLOPs 缩减率 (1 - flops_ratio)
    eval_time_sec: float           # 本次评估耗时（秒）
    epochs: int                    # 搜索时的训练轮数


def candidate_to_key(channels: Sequence[int]) -> str:
    """将通道配置转为字符串键，用作缓存的 key 和 CSV 记录的标识。"""
    return "-".join(map(str, channels))


def parse_space(s: str | None) -> list[list[int]]:
    """
    解析搜索空间字符串。
    格式: "8,12,16;16,20,24,28,32;32,40,48,56,64"
    分号分隔三个阶段，逗号分隔每个阶段内的可选值。
    若输入为 None 则返回默认搜索空间。
    """
    if not s:
        return SEARCH_SPACE
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
    """若 CSV 文件不存在，则创建文件并写入表头。"""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()


def append_csv(path: Path, row: dict) -> None:
    """向 CSV 文件追加一行数据。"""
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def random_candidate(space: list[list[int]], rng: random.Random) -> Candidate:
    """从搜索空间中随机采样一个候选解（每个阶段随机选一个通道值）。"""
    return tuple(rng.choice(group) for group in space)  # type: ignore[return-value]


def repair_candidate(channels: Sequence[int], space: list[list[int]]) -> Candidate:
    """
    修复候选解：将每个阶段的通道值映射到搜索空间中最近的合法值。
    用于修正交叉/变异操作后产生的非法通道配置。
    """
    fixed = []
    for value, group in zip(channels, space):
        # 选择与当前值距离最小的合法值（"吸附"操作）
        fixed.append(min(group, key=lambda x: abs(x - int(round(value)))))
    return tuple(fixed)  # type: ignore[return-value]


def decode_position(position: Sequence[float], space: list[list[int]]) -> Candidate:
    """
    将 PSO 中粒子的连续位置向量解码为离散的候选通道配置。
    浮点位置四舍五入为索引，然后查找对应阶段的通道值。
    """
    decoded = []
    for pos, group in zip(position, space):
        idx = int(round(pos))
        # 将索引限制在合法范围内 [0, len(group)-1]
        idx = max(0, min(len(group) - 1, idx))
        decoded.append(group[idx])
    return tuple(decoded)  # type: ignore[return-value]


def conv_model_info(channels: Sequence[int], device: torch.device) -> tuple[int, int]:
    """
    快速计算给定通道配置下的模型参数量和 FLOPs（不进行训练）。
    用于获取基线模型信息。
    """
    model = width_resnet32(stage_channels=channels, num_classes=10).to(device)
    params = count_parameters(model)
    flops = measure_flops(model, input_size=(3, 32, 32), device=device)
    del model
    # 释放 GPU 缓存，避免后续评估时显存不足
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return params, flops


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool) -> tuple[float, float]:
    """
    训练模型一个 epoch。
    支持混合精度训练 (AMP) 以加速 GPU 上的计算。
    返回: (平均训练损失, 训练 Top-1 准确率)
    """
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    # AMP 梯度缩放器：仅在启用 AMP 且使用 CUDA 时生效
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in tqdm(loader, desc="candidate train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 清零梯度（set_to_none=True 将梯度设为 None 以节省内存）
        optimizer.zero_grad(set_to_none=True)

        # 混合精度前向传播：在 autocast 上下文中自动选择 FP16/FP32
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # 反向传播（scaler 处理梯度缩放以避免 FP16 下溢）
        scaler.scale(loss).backward()
        # 更新参数（scaler 自动 unscale 梯度后调用 optimizer.step）
        scaler.step(optimizer)
        # 更新缩放因子
        scaler.update()

        # 计算当前批次的准确率（detach 避免不必要的计算图保留）
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        losses.update(float(loss.item()), images.size(0))
        top1.update(float(acc1.item()), images.size(0))

    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """
    在测试集上评估模型（关闭梯度计算以节省内存和加速）。
    返回: (平均测试损失, 测试 Top-1 准确率)
    """
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
    """
    适应度评估器：负责对候选通道配置进行完整的训练 + 评估流程。
    内部维护评估缓存，避免重复评估相同配置；同时将每次评估结果写入 CSV。
    """

    def __init__(self, args, space: list[list[int]], out_dir: Path, device: torch.device) -> None:
        self.args = args
        self.space = space
        self.out_dir = out_dir
        self.device = device
        # 仅在 CUDA 设备且用户启用 --amp 时使用混合精度
        self.use_amp = bool(args.amp and device.type == "cuda")
        self.rng = random.Random(args.seed)
        # 评估缓存：通道配置字符串 → EvalResult，避免重复评估
        self.cache: Dict[str, EvalResult] = {}
        # 所有候选评估的记录文件
        self.eval_csv = out_dir / "evaluations.csv"

        # 加载 CIFAR-10 训练集和测试集（可限制样本数以加速搜索）
        self.train_loader, self.test_loader = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )

        # 计算基线模型的参数量和 FLOPs，作为压缩率的参考基准
        self.baseline_params, self.baseline_flops = conv_model_info(BASELINE_CHANNELS, device)

        # 初始化评估记录 CSV 文件的表头
        fields = [
            "channels", "fitness", "test_acc", "train_acc", "test_loss", "train_loss",
            "params", "flops", "params_ratio", "flops_ratio",
            "params_compression_rate", "flops_reduction_rate", "eval_time_sec", "epochs",
        ]
        write_csv_header_if_needed(self.eval_csv, fields)

    def fitness_from_metrics(self, test_acc: float, params_ratio: float, flops_ratio: float) -> float:
        """
        根据测试准确率和模型开销比率计算适应度分数。
        适应度 = 准确率 - 100 × (λ_params × 参数比率 + λ_flops × FLOPs 比率)
        准确率越高、参数/FLOPs 越小，适应度越高。
        """
        # test_acc 单位为百分比；params_ratio 和 flops_ratio 为 [0, 1] 范围
        return test_acc - 100.0 * (self.args.lambda_params * params_ratio + self.args.lambda_flops * flops_ratio)

    def evaluate_candidate(self, channels: Sequence[int]) -> EvalResult:
        """
        评估单个候选通道配置。
        流程：修复通道值 → 检查缓存 → 构建模型 → 训练 → 测试 → 计算适应度 → 缓存并记录。
        """
        # 将通道值修正为搜索空间中最近的合法值
        channels = repair_candidate(channels, self.space)
        key = candidate_to_key(channels)

        # 命中缓存则直接返回（GA/PSO 中可能多次生成相同候选）
        if key in self.cache:
            return self.cache[key]

        start = time.time()

        # 根据通道配置确定性地设置种子，保证相同配置获得相同训练结果
        # 种子 = 基础种子 + 通道值的加权求和，确保不同配置产生不同随机状态
        local_seed = self.args.seed + sum((i + 1) * c for i, c in enumerate(channels))
        set_seed(local_seed, deterministic=False)

        # ========================
        # 模型构建
        # ========================
        model = width_resnet32(stage_channels=channels, num_classes=10).to(self.device)
        params = count_parameters(model)
        flops = measure_flops(model, input_size=(3, 32, 32), device=self.device)

        # ========================
        # 优化器与学习率调度器
        # ========================
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        # 余弦退火学习率调度：学习率从 lr 平滑降至接近 0
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.args.search_epochs, 1))

        # ========================
        # 快速训练（少量 epoch 评估候选质量）
        # ========================
        train_loss = math.nan
        train_acc = math.nan
        for _ in range(self.args.search_epochs):
            train_loss, train_acc = train_one_epoch(
                model, self.train_loader, criterion, optimizer, self.device, self.use_amp
            )
            scheduler.step()

        # ========================
        # 测试评估
        # ========================
        test_loss, test_acc = evaluate(model, self.test_loader, criterion, self.device)

        # ========================
        # 计算各项指标
        # ========================
        params_ratio = params / self.baseline_params       # 参数量比率（相对于基线）
        flops_ratio = flops / self.baseline_flops          # FLOPs 比率（相对于基线）
        fitness = self.fitness_from_metrics(test_acc, params_ratio, flops_ratio)

        # 封装评估结果
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
            params_compression_rate=float(1.0 - params_ratio),   # 参数压缩率
            flops_reduction_rate=float(1.0 - flops_ratio),       # FLOPs 缩减率
            eval_time_sec=float(time.time() - start),
            epochs=int(self.args.search_epochs),
        )

        # 存入缓存并追加写入 CSV
        self.cache[key] = result
        append_csv(self.eval_csv, {**asdict(result), "channels": key})

        # 释放模型内存
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result


def tournament_select(
    population: list[Candidate], scores: dict[Candidate, float], rng: random.Random, k: int = 3
) -> Candidate:
    """
    锦标赛选择算子：从种群中随机抽取 k 个个体，返回其中适应度最高的。
    锦标赛选择能在选择压力和种群多样性之间取得良好平衡。
    """
    contenders = [rng.choice(population) for _ in range(k)]
    return max(contenders, key=lambda c: scores[c])


def run_ga(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path) -> EvalResult:
    """
    运行遗传算法 (Genetic Algorithm) 搜索最优通道配置。
    流程：
      1. 随机初始化种群
      2. 评估每个个体的适应度
      3. 保留精英 → 锦标赛选择 → 均匀交叉 → 变异 → 生成下一代
      4. 重复直到达到最大代数
    返回: 全局最优评估结果
    """
    rng = random.Random(args.seed + 101)

    # 搜索历史记录 CSV
    history_csv = out_dir / "ga_history.csv"
    write_csv_header_if_needed(history_csv, [
        "generation", "best_channels", "best_fitness", "best_acc", "best_params_ratio", "best_flops_ratio"
    ])

    # ========================
    # 1. 随机初始化种群（确保无重复个体）
    # ========================
    population: list[Candidate] = []
    seen = set()
    while len(population) < args.ga_population:
        cand = random_candidate(space, rng)
        if cand not in seen:
            population.append(cand)
            seen.add(cand)

    # 记录全局最优解
    global_best: EvalResult | None = None

    # ========================
    # 2. 迭代进化
    # ========================
    for gen in range(args.ga_generations):
        print(f"\n[GA] Generation {gen + 1}/{args.ga_generations}")

        # 评估当代种群中所有个体的适应度
        results = {cand: evaluator.evaluate_candidate(cand) for cand in population}
        scores = {cand: res.fitness for cand, res in results.items()}

        # 记录当代最优和全局最优
        gen_best = max(results.values(), key=lambda r: r.fitness)
        if global_best is None or gen_best.fitness > global_best.fitness:
            global_best = gen_best

        # 写入历史记录
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

        # ========================
        # 3. 精英保留 + 选择 + 交叉 + 变异 → 生成下一代
        # ========================
        # 精英保留：直接将适应度最高的若干个体复制到下一代
        elite_count = max(1, args.ga_elites)
        elites = sorted(population, key=lambda c: scores[c], reverse=True)[:elite_count]
        next_population: list[Candidate] = list(elites)

        # 通过选择和交叉填充剩余个体
        while len(next_population) < args.ga_population:
            # 锦标赛选择两个父代
            p1 = tournament_select(population, scores, rng)
            p2 = tournament_select(population, scores, rng)

            # 均匀交叉：以交叉概率决定是否交叉，每个维度以 50% 概率来自任一父代
            if rng.random() < args.ga_crossover_rate:
                child = tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(3))
            else:
                child = p1

            # 变异：每个维度以变异概率随机替换为该阶段的任意合法值
            child_list = list(child)
            for i, group in enumerate(space):
                if rng.random() < args.ga_mutation_rate:
                    child_list[i] = rng.choice(group)

            # 修复操作：确保变异后的值仍为合法通道宽度
            child = repair_candidate(child_list, space)
            next_population.append(child)

        population = next_population

    # ========================
    # 4. 保存 GA 搜索结果
    # ========================
    assert global_best is not None
    with (out_dir / "ga_best.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(global_best), f, ensure_ascii=False, indent=2)
    return global_best


def run_pso(args, evaluator: FitnessEvaluator, space: list[list[int]], out_dir: Path) -> EvalResult:
    """
    运行粒子群优化 (Particle Swarm Optimization) 搜索最优通道配置。
    流程：
      1. 随机初始化粒子位置和速度
      2. 评估每个粒子对应配置的适应度
      3. 更新个体历史最优 (pbest) 和全局最优 (gbest)
      4. 根据 PSO 速度更新公式调整粒子速度和位置
      5. 重复直到达到最大迭代次数
    返回: 全局最优评估结果
    """
    rng = random.Random(args.seed + 202)

    # 搜索历史记录 CSV
    history_csv = out_dir / "pso_history.csv"
    write_csv_header_if_needed(history_csv, [
        "iteration", "best_channels", "best_fitness", "best_acc", "best_params_ratio", "best_flops_ratio"
    ])

    # ========================
    # 1. 初始化粒子群
    # ========================
    dims = 3  # 搜索维度 = 3（三个阶段的通道宽度）
    # 每个维度的上界（索引范围）
    upper = [len(group) - 1 for group in space]

    # 粒子位置：连续值，后续通过 decode_position 映射为离散通道配置
    positions = [[rng.uniform(0, upper[d]) for d in range(dims)] for _ in range(args.pso_particles)]
    # 粒子速度：初始为 0
    velocities = [[0.0 for _ in range(dims)] for _ in range(args.pso_particles)]

    # 个体历史最优位置和分数
    pbest_positions = [pos[:] for pos in positions]
    pbest_scores = [-float("inf") for _ in range(args.pso_particles)]
    pbest_results: list[EvalResult | None] = [None for _ in range(args.pso_particles)]

    # 全局最优位置和结果
    gbest_position: list[float] | None = None
    gbest_result: EvalResult | None = None

    # ========================
    # 2. 迭代优化
    # ========================
    for it in range(args.pso_iterations):
        print(f"\n[PSO] Iteration {it + 1}/{args.pso_iterations}")

        # 评估所有粒子的当前位置
        for i in range(args.pso_particles):
            # 将连续位置解码为离散的通道配置
            cand = decode_position(positions[i], space)
            res = evaluator.evaluate_candidate(cand)

            # 更新个体历史最优 (pbest)
            if res.fitness > pbest_scores[i]:
                pbest_scores[i] = res.fitness
                pbest_positions[i] = positions[i][:]
                pbest_results[i] = res

            # 更新全局最优 (gbest)
            if gbest_result is None or res.fitness > gbest_result.fitness:
                gbest_result = res
                gbest_position = positions[i][:]

        # 记录本轮迭代的全局最优
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

        # ========================
        # 3. 更新粒子速度和位置
        # ========================
        # PSO 核心公式:
        #   v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        #   x = x + v
        #   w:  惯性权重（控制历史速度的影响）
        #   c1: 认知系数（粒子向自身历史最优靠拢的倾向）
        #   c2: 社会系数（粒子向全局最优靠拢的倾向）
        #   r1, r2: [0,1) 的随机数，引入随机性
        for i in range(args.pso_particles):
            for d in range(dims):
                r1 = rng.random()
                r2 = rng.random()
                # 速度更新
                velocities[i][d] = (
                    args.pso_w * velocities[i][d]                                    # 惯性项
                    + args.pso_c1 * r1 * (pbest_positions[i][d] - positions[i][d])  # 认知项（个体最优吸引）
                    + args.pso_c2 * r2 * (gbest_position[d] - positions[i][d])      # 社会项（全局最优吸引）
                )
                # 位置更新
                positions[i][d] += velocities[i][d]
                # 将位置限制在合法索引范围内（边界约束）
                positions[i][d] = max(0.0, min(float(upper[d]), positions[i][d]))

    # ========================
    # 4. 保存 PSO 搜索结果
    # ========================
    assert gbest_result is not None
    with (out_dir / "pso_best.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(gbest_result), f, ensure_ascii=False, indent=2)
    return gbest_result


def main() -> None:
    """主函数：解析参数 → 初始化评估器 → 运行 GA/PSO 搜索 → 输出最终结果。"""

    # ========================
    # 1. 命令行参数解析
    # ========================
    parser = argparse.ArgumentParser(
        description="GA/PSO stage-channel search for CIFAR-10 ResNet32 compression"
    )

    # --- 通用参数 ---
    parser.add_argument("--algorithm", choices=["ga", "pso", "both"], default="both",
                        help="选择搜索算法：ga / pso / both（同时运行两者并比较）")
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 数据集目录")
    parser.add_argument("--save-dir", default="runs", help="结果保存根目录")
    parser.add_argument("--run-name", default="channel_search_ga_pso", help="本次实验名称")
    parser.add_argument("--space", default=None,
                        help="自定义搜索空间，格式: '8,12,16;16,20,24,28,32;32,40,48,56,64'")
    parser.add_argument("--search-epochs", type=int, default=3,
                        help="评估每个候选模型时的训练轮数（少量 epoch 快速评估）")
    parser.add_argument("--batch-size", type=int, default=128, help="训练/评估的批大小")
    parser.add_argument("--lr", type=float, default=0.05, help="初始学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD 动量系数")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减（L2 正则化）")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--max-train-samples", type=int, default=5000,
                        help="训练集最大样本数（0 表示使用全部，减少可加速搜索）")
    parser.add_argument("--max-test-samples", type=int, default=2000,
                        help="测试集最大样本数（0 表示使用全部）")
    parser.add_argument("--lambda-params", type=float, default=0.15,
                        help="适应度函数中参数量比率的惩罚系数")
    parser.add_argument("--lambda-flops", type=float, default=0.15,
                        help="适应度函数中 FLOPs 比率的惩罚系数")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备 (cuda/cpu)")
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练以加速")

    # --- GA 专属参数 ---
    parser.add_argument("--ga-population", type=int, default=8, help="GA 种群大小")
    parser.add_argument("--ga-generations", type=int, default=5, help="GA 进化代数")
    parser.add_argument("--ga-elites", type=int, default=2, help="GA 每代精英保留数量")
    parser.add_argument("--ga-crossover-rate", type=float, default=0.8, help="GA 交叉概率")
    parser.add_argument("--ga-mutation-rate", type=float, default=0.25, help="GA 单维度变异概率")

    # --- PSO 专属参数 ---
    parser.add_argument("--pso-particles", type=int, default=8, help="PSO 粒子数量")
    parser.add_argument("--pso-iterations", type=int, default=5, help="PSO 迭代轮数")
    parser.add_argument("--pso-w", type=float, default=0.6, help="PSO 惯性权重")
    parser.add_argument("--pso-c1", type=float, default=1.4, help="PSO 认知系数（个体最优吸引）")
    parser.add_argument("--pso-c2", type=float, default=1.4, help="PSO 社会系数（全局最优吸引）")

    args = parser.parse_args()

    # ========================
    # 2. 初始化设置
    # ========================
    # 设置全局随机种子（保证搜索过程可复现）
    set_seed(args.seed, deterministic=False)
    # 解析搜索空间
    space = parse_space(args.space)
    # 确定计算设备
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # 创建输出目录
    out_dir = Path(args.save_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次搜索的完整配置（方便复现和对比）
    config = vars(args).copy()
    config["search_space"] = space
    config["baseline_channels"] = list(BASELINE_CHANNELS)
    config["model_family"] = "CIFAR-10 ResNet32, stage-level width search"
    with (out_dir / "search_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # ========================
    # 3. 初始化评估器并打印基线信息
    # ========================
    evaluator = FitnessEvaluator(args, space, out_dir, device)
    print("Baseline channels:", candidate_to_key(BASELINE_CHANNELS))
    print(f"Baseline params: {evaluator.baseline_params} ({human_number(evaluator.baseline_params)})")
    print(f"Baseline FLOPs : {evaluator.baseline_flops} ({human_number(evaluator.baseline_flops)})")
    print("Search space:", space)
    print("Fitness = Acc - 100 * (lambda_params * ParamsRatio + lambda_flops * FLOPsRatio)")

    # ========================
    # 4. 运行搜索
    # ========================
    total_start = time.time()
    best_results = {}

    # 运行遗传算法搜索
    if args.algorithm in {"ga", "both"}:
        ga_best = run_ga(args, evaluator, space, out_dir)
        best_results["ga"] = asdict(ga_best)

    # 运行粒子群优化搜索
    if args.algorithm in {"pso", "both"}:
        pso_best = run_pso(args, evaluator, space, out_dir)
        best_results["pso"] = asdict(pso_best)

    # ========================
    # 5. 汇总最终结果
    # ========================
    # 从所有算法的结果中选出适应度最高的
    final_best_name, final_best = max(best_results.items(), key=lambda item: item[1]["fitness"])

    summary = {
        "algorithm": args.algorithm,                 # 使用的搜索算法
        "best_algorithm": final_best_name,           # 最优算法名称（ga 或 pso）
        "best": final_best,                          # 最优候选的完整评估结果
        "baseline": {                                # 基线模型信息
            "channels": list(BASELINE_CHANNELS),
            "params": evaluator.baseline_params,
            "flops": evaluator.baseline_flops,
        },
        "num_unique_evaluations": len(evaluator.cache),  # 总共评估了多少个不同的候选配置
        "total_search_time_sec": time.time() - total_start,  # 总搜索耗时
    }

    # 保存最终结果到 JSON 文件
    with (out_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ========================
    # 6. 打印结果和后续指引
    # ========================
    print("\nSearch done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 提示用户下一步：使用最优配置进行完整训练
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
