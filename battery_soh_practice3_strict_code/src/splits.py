from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# 每个目标电池只对应一个源电池，不混合多个源电池。
C_SOURCE_FOR_TARGET: Dict[str, str] = {
    "B0005": "B0007",
    "B0006": "B0007",
    "B0007": "B0005",
    "B0018": "B0005",
}


def split_a_random_single(df: pd.DataFrame, battery_id: str, seed: int = 42):
    """A: 单个电池随机 60% 训练、20% 验证、20% 测试。"""
    data = df[df["battery_id"] == battery_id].reset_index(drop=True)
    train, tmp = train_test_split(data, test_size=0.40, shuffle=True, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.50, shuffle=True, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def split_b_chrono_single(df: pd.DataFrame, battery_id: str):
    """
    B: 单个电池按 cycle 排序。
    前 60% cycle 作为训练/验证开发区间，后 40% cycle 作为测试集。
    为满足报告中的验证集要求，从前 60% 区间的末尾取 20% 作为验证集，其余作为训练集。
    """
    data = df[df["battery_id"] == battery_id].sort_values("cycle_index").reset_index(drop=True)
    n_dev = int(len(data) * 0.60)
    dev = data.iloc[:n_dev].reset_index(drop=True)
    test = data.iloc[n_dev:].reset_index(drop=True)
    n_val = max(1, int(len(dev) * 0.20))
    train = dev.iloc[:-n_val].reset_index(drop=True)
    val = dev.iloc[-n_val:].reset_index(drop=True)
    return train, val, test


def split_c_transfer(df: pd.DataFrame, source_battery: str, target_battery: str):
    """
    C: 单个源电池 + 目标电池前 10% cycle 用作训练，目标电池后 90% cycle 用作测试。
    为避免把目标前 10% 再切走，验证集从源电池尾部 20% 取得；目标前 10% 全部进入训练集。
    """
    source = df[df["battery_id"] == source_battery].sort_values("cycle_index").reset_index(drop=True)
    target = df[df["battery_id"] == target_battery].sort_values("cycle_index").reset_index(drop=True)

    n_source_val = max(1, int(len(source) * 0.20))
    source_train = source.iloc[:-n_source_val].reset_index(drop=True)
    source_val = source.iloc[-n_source_val:].reset_index(drop=True)

    n_target_adapt = max(1, int(len(target) * 0.10))
    target_adapt = target.iloc[:n_target_adapt].reset_index(drop=True)
    target_test = target.iloc[n_target_adapt:].reset_index(drop=True)

    train = pd.concat([source_train, target_adapt], ignore_index=True)
    val = source_val
    test = target_test
    return train, val, test


def make_all_scenarios(df: pd.DataFrame, seed: int = 42):
    """Return dict: scenario -> (case, source_battery, target_battery, train, val, test)."""
    scenarios = {}

    for battery_id in BATTERIES:
        train, val, test = split_a_random_single(df, battery_id, seed=seed)
        name = f"A_random_{battery_id}_60_20_20"
        scenarios[name] = ("A", battery_id, battery_id, train, val, test)

    for battery_id in BATTERIES:
        train, val, test = split_b_chrono_single(df, battery_id)
        name = f"B_chrono_{battery_id}_first60_last40"
        scenarios[name] = ("B", battery_id, battery_id, train, val, test)

    for target_battery in BATTERIES:
        source_battery = C_SOURCE_FOR_TARGET[target_battery]
        train, val, test = split_c_transfer(df, source_battery, target_battery)
        name = f"C_transfer_{source_battery}_to_{target_battery}_target10"
        scenarios[name] = ("C", source_battery, target_battery, train, val, test)

    return scenarios
