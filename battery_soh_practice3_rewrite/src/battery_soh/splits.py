from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import BATTERIES


@dataclass(frozen=True)
class Scenario:
    """实验场景数据类，封装单次 train/val/test 划分的完整上下文。

    Attributes:
        name: 场景唯一标识（如 "A_random_B0005_60_20_20"）。
        case: 实验类别（"A" 随机 / "B" 时序 / "C" 迁移）。
        source_battery: 训练数据来源电池（C 类场景中为源电池）。
        target_battery: 测试目标电池（A/B 类与 source_battery 相同）。
        train / val / test: 对应的 DataFrame 子集（已重置索引）。
    """
    name: str
    case: str
    source_battery: str
    target_battery: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_scenarios(features: pd.DataFrame, seed: int) -> list[Scenario]:
    """构建实践三要求的三类全部实验场景。

    A 类（随机）：每个电池独立做 60%/20%/20% 随机划分，共 4 个场景。
    B 类（时序）：每个电池前 60% cycle 做开发集（再内部分 80/20 训练/验证），后 40% 做测试，共 4 个场景。
    C 类（迁移）：遍历所有 source≠target 组合，source 全量 + target 前 10% 做训练/验证，target 后 90% 测试，共 4×3=12 个场景。

    总计返回 4 + 4 + 12 = 20 个 Scenario 对象。

    Args:
        features: load_nasa_features 返回的完整特征表。
        seed: A 类随机划分使用的随机种子。

    Returns:
        包含全部 20 个实验场景的列表。
    """
    scenarios: list[Scenario] = []

    for battery_id in BATTERIES:
        train, val, test = _random_single_battery(features, battery_id, seed)
        scenarios.append(Scenario(f"A_random_{battery_id}_60_20_20", "A", battery_id, battery_id, train, val, test))

    for battery_id in BATTERIES:
        train, val, test = _chronological_single_battery(features, battery_id)
        scenarios.append(Scenario(f"B_chrono_{battery_id}_first60_last40", "B", battery_id, battery_id, train, val, test))

    for source in BATTERIES:
        for target in BATTERIES:
            if source == target:
                continue
            train, val, test = _transfer_battery(features, source, target)
            scenarios.append(Scenario(f"C_transfer_{source}_to_{target}_target10", "C", source, target, train, val, test))

    return scenarios


def summarize_splits(scenarios: list[Scenario]) -> pd.DataFrame:
    """汇总所有场景各子集的样本量、涉及电池、cycle 范围、SOH 范围等统计信息。

    用于生成数据划分质量检查表。
    """
    rows = []
    for scenario in scenarios:
        for split_name, frame in [("train", scenario.train), ("validation", scenario.val), ("test", scenario.test)]:
            rows.append(
                {
                    "scenario": scenario.name,
                    "case": scenario.case,
                    "source_battery": scenario.source_battery,
                    "target_battery": scenario.target_battery,
                    "split": split_name,
                    "n": len(frame),
                    "batteries_in_split": ",".join(sorted(frame["battery_id"].unique())) if len(frame) else "",
                    "cycle_min": int(frame["cycle_index"].min()) if len(frame) else None,
                    "cycle_max": int(frame["cycle_index"].max()) if len(frame) else None,
                    "soh_min": float(frame["soh_percent"].min()) if len(frame) else None,
                    "soh_max": float(frame["soh_percent"].max()) if len(frame) else None,
                }
            )
    return pd.DataFrame(rows)


def _random_single_battery(df: pd.DataFrame, battery_id: str, seed: int):
    """对单个电池数据执行随机 60%/20%/20% 划分（先 40% 切分出开发+测试，再对开发集 50% 切分验证）。"""
    data = df[df["battery_id"] == battery_id].reset_index(drop=True)
    train, temp = train_test_split(data, test_size=0.40, shuffle=True, random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, shuffle=True, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _chronological_single_battery(df: pd.DataFrame, battery_id: str):
    """对单个电池按 cycle_index 顺序划分：前 60% 做开发集（内部再切 20% 作为 val），后 40% 做测试。"""
    data = df[df["battery_id"] == battery_id].sort_values("cycle_index").reset_index(drop=True)
    dev_count = int(len(data) * 0.60)
    dev = data.iloc[:dev_count].reset_index(drop=True)
    test = data.iloc[dev_count:].reset_index(drop=True)
    val_count = max(1, int(len(dev) * 0.20))
    train = dev.iloc[:-val_count].reset_index(drop=True)
    val = dev.iloc[-val_count:].reset_index(drop=True)
    return train, val, test


def _transfer_battery(df: pd.DataFrame, source_battery: str, target_battery: str):
    """构建迁移学习场景划分：
    - 源电池前 80% 做训练，源电池后 20% 做验证；
    - 目标电池前 10% 加入训练（用于适应），目标电池后 90% 做测试。
    """
    source = df[df["battery_id"] == source_battery].sort_values("cycle_index").reset_index(drop=True)
    target = df[df["battery_id"] == target_battery].sort_values("cycle_index").reset_index(drop=True)

    source_val_count = max(1, int(len(source) * 0.20))
    source_train = source.iloc[:-source_val_count].reset_index(drop=True)
    source_val = source.iloc[-source_val_count:].reset_index(drop=True)

    target_adapt_count = max(1, int(len(target) * 0.10))
    target_adapt = target.iloc[:target_adapt_count].reset_index(drop=True)
    target_test = target.iloc[target_adapt_count:].reset_index(drop=True)

    train = pd.concat([source_train, target_adapt], ignore_index=True)
    return train, source_val, target_test
