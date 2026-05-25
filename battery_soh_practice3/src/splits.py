from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_a_random_single(df: pd.DataFrame, battery_id: str = "B0005", seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Case A: single battery random 60% train, 20% validation, 20% test."""
    data = df[df["battery_id"] == battery_id].reset_index(drop=True)
    train, tmp = train_test_split(data, test_size=0.40, random_state=seed, shuffle=True)
    val, test = train_test_split(tmp, test_size=0.50, random_state=seed, shuffle=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def split_b_chrono_single(df: pd.DataFrame, battery_id: str = "B0005") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Case B: first 60% cycles are used for model development; last 40% cycles are used for test.
    A small validation set is taken from the tail of the first-60% block for early stopping.
    """
    data = df[df["battery_id"] == battery_id].sort_values("cycle_index").reset_index(drop=True)
    n_pool = int(len(data) * 0.60)
    train_val = data.iloc[:n_pool].reset_index(drop=True)
    test = data.iloc[n_pool:].reset_index(drop=True)
    n_val = max(1, int(len(train_val) * 0.20))
    train = train_val.iloc[:-n_val].reset_index(drop=True)
    val = train_val.iloc[-n_val:].reset_index(drop=True)
    return train, val, test


def split_c_transfer(
    df: pd.DataFrame,
    source_battery: str = "B0005",
    target_battery: str = "B0007",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Case C: one battery is source training data; first 10% cycles of target battery are used for adaptation;
    later 90% cycles of target battery are used for test.
    """
    source = df[df["battery_id"] == source_battery].sort_values("cycle_index").reset_index(drop=True)
    target = df[df["battery_id"] == target_battery].sort_values("cycle_index").reset_index(drop=True)
    n_adapt = max(1, int(len(target) * 0.10))
    adapt = target.iloc[:n_adapt].reset_index(drop=True)
    test = target.iloc[n_adapt:].reset_index(drop=True)

    # Use the last 20% of the target adaptation block as validation; source data stays in train.
    n_val = max(1, int(len(adapt) * 0.20))
    train = pd.concat([source, adapt.iloc[:-n_val]], ignore_index=True)
    val = adapt.iloc[-n_val:].reset_index(drop=True)
    return train, val, test
