from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import BATTERIES

TRANSFER_SOURCE = {
    "B0005": "B0007",
    "B0006": "B0007",
    "B0007": "B0005",
    "B0018": "B0005",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    case: str
    source_battery: str
    target_battery: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_scenarios(features: pd.DataFrame, seed: int) -> list[Scenario]:
    scenarios: list[Scenario] = []

    for battery_id in BATTERIES:
        train, val, test = _random_single_battery(features, battery_id, seed)
        scenarios.append(Scenario(f"A_random_{battery_id}_60_20_20", "A", battery_id, battery_id, train, val, test))

    for battery_id in BATTERIES:
        train, val, test = _chronological_single_battery(features, battery_id)
        scenarios.append(Scenario(f"B_chrono_{battery_id}_first60_last40", "B", battery_id, battery_id, train, val, test))

    for target in BATTERIES:
        source = TRANSFER_SOURCE[target]
        train, val, test = _transfer_battery(features, source, target)
        scenarios.append(Scenario(f"C_transfer_{source}_to_{target}_target10", "C", source, target, train, val, test))

    return scenarios


def summarize_splits(scenarios: list[Scenario]) -> pd.DataFrame:
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
    data = df[df["battery_id"] == battery_id].reset_index(drop=True)
    train, temp = train_test_split(data, test_size=0.40, shuffle=True, random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, shuffle=True, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _chronological_single_battery(df: pd.DataFrame, battery_id: str):
    data = df[df["battery_id"] == battery_id].sort_values("cycle_index").reset_index(drop=True)
    dev_count = int(len(data) * 0.60)
    dev = data.iloc[:dev_count].reset_index(drop=True)
    test = data.iloc[dev_count:].reset_index(drop=True)
    val_count = max(1, int(len(dev) * 0.20))
    train = dev.iloc[:-val_count].reset_index(drop=True)
    val = dev.iloc[-val_count:].reset_index(drop=True)
    return train, val, test


def _transfer_battery(df: pd.DataFrame, source_battery: str, target_battery: str):
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

