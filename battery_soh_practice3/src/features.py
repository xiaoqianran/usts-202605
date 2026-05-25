import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
numpy_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def extract_zip_if_needed(data_zip: str, work_dir: str) -> Path:
    """Extract BatteryAgingARC.zip and return directory containing .mat files."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    mat_files = list(work.rglob("B0005.mat"))
    if mat_files:
        return mat_files[0].parent

    with zipfile.ZipFile(data_zip, "r") as zf:
        zf.extractall(work)

    mat_files = list(work.rglob("B0005.mat"))
    if not mat_files:
        raise FileNotFoundError("Could not find B0005.mat after extracting the zip file.")
    return mat_files[0].parent


def _load_battery_cycles(mat_path: str):
    battery_id = Path(mat_path).stem
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if battery_id not in mat:
        raise KeyError(f"{battery_id} not found in {mat_path}")
    return battery_id, mat[battery_id].cycle


def _first_time_voltage_below(time_s: np.ndarray, voltage: np.ndarray, threshold: float) -> float:
    """Return the first interpolated time when discharge voltage falls below threshold."""
    idx = np.where(voltage <= threshold)[0]
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(time_s[0])
    v0, v1 = float(voltage[i - 1]), float(voltage[i])
    t0, t1 = float(time_s[i - 1]), float(time_s[i])
    if abs(v1 - v0) < 1e-12:
        return t1
    ratio = (threshold - v0) / (v1 - v0)
    return float(t0 + ratio * (t1 - t0))


def _value_at_fraction(time_s: np.ndarray, values: np.ndarray, fraction: float) -> float:
    target_t = float(time_s[0] + fraction * (time_s[-1] - time_s[0]))
    return float(np.interp(target_t, time_s, values))


def extract_discharge_features(mat_path: str) -> pd.DataFrame:
    """
    Extract one row per discharge cycle.

    Label:
        SOH (%) = discharge Capacity / rated capacity 2 Ah * 100

    Important anti-leakage note:
        We do not use Capacity as an input feature. We also avoid using direct
        ampere-hour integration as an input because it nearly duplicates the label.
    """
    battery_id, cycles = _load_battery_cycles(mat_path)
    rows: List[Dict[str, float]] = []
    discharge_idx = 0

    for global_idx, cycle in enumerate(cycles):
        if str(cycle.type) != "discharge":
            continue
        data = cycle.data
        try:
            time_s = np.asarray(data.Time, dtype=float).reshape(-1)
            voltage = np.asarray(data.Voltage_measured, dtype=float).reshape(-1)
            current = np.asarray(data.Current_measured, dtype=float).reshape(-1)
            temp = np.asarray(data.Temperature_measured, dtype=float).reshape(-1)
            capacity = float(np.asarray(data.Capacity).squeeze())
        except Exception:
            continue

        # Basic cleaning: finite values, consistent length, valid capacity, monotonic time.
        if len(time_s) < 10 or not (len(time_s) == len(voltage) == len(current) == len(temp)):
            continue
        finite = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temp)
        time_s, voltage, current, temp = time_s[finite], voltage[finite], current[finite], temp[finite]
        if len(time_s) < 10 or not np.isfinite(capacity):
            continue
        if capacity <= 0.5 or capacity > 2.2:
            continue

        order = np.argsort(time_s)
        time_s, voltage, current, temp = time_s[order], voltage[order], current[order], temp[order]
        _, unique_idx = np.unique(time_s, return_index=True)
        time_s, voltage, current, temp = time_s[unique_idx], voltage[unique_idx], current[unique_idx], temp[unique_idx]
        if len(time_s) < 10:
            continue
        duration = float(time_s[-1] - time_s[0])
        if duration <= 0:
            continue

        abs_current = np.abs(current)
        # Wh is not exactly the label but is kept as a physically meaningful discharge-energy feature.
        # Ah integral is deliberately not used in model feature list.
        energy_wh = float(numpy_trapezoid(voltage * abs_current, time_s) / 3600.0)
        charge_ah_integral = float(numpy_trapezoid(abs_current, time_s) / 3600.0)

        row: Dict[str, float] = {
            "battery_id": battery_id,
            "cycle_index": discharge_idx,
            "global_cycle_index": global_idx,
            "ambient_temperature": float(cycle.ambient_temperature),
            "capacity_ah": capacity,
            "soh_percent": capacity / 2.0 * 100.0,
            "duration_s": duration,
            "voltage_start": float(voltage[0]),
            "voltage_end": float(voltage[-1]),
            "voltage_mean": float(np.mean(voltage)),
            "voltage_std": float(np.std(voltage)),
            "voltage_min": float(np.min(voltage)),
            "voltage_max": float(np.max(voltage)),
            "current_mean": float(np.mean(current)),
            "current_abs_mean": float(np.mean(abs_current)),
            "current_std": float(np.std(current)),
            "temp_start": float(temp[0]),
            "temp_end": float(temp[-1]),
            "temp_mean": float(np.mean(temp)),
            "temp_max": float(np.max(temp)),
            "temp_min": float(np.min(temp)),
            "temp_rise": float(temp[-1] - temp[0]),
            "temp_range": float(np.max(temp) - np.min(temp)),
            "voltage_slope": float((voltage[-1] - voltage[0]) / duration),
            "energy_wh": energy_wh,
            "charge_ah_integral": charge_ah_integral,
            "avg_power_w": float(energy_wh * 3600.0 / duration),
        }

        for threshold in [4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5]:
            key = f"time_to_{str(threshold).replace('.', 'p')}v_s"
            row[key] = _first_time_voltage_below(time_s, voltage, threshold)

        for fraction in [0.25, 0.50, 0.75]:
            suffix = str(int(fraction * 100))
            row[f"voltage_at_{suffix}pct_time"] = _value_at_fraction(time_s, voltage, fraction)
            row[f"temp_at_{suffix}pct_time"] = _value_at_fraction(time_s, temp, fraction)

        early_cut = time_s[0] + 0.25 * duration
        early_mask = time_s <= early_cut
        row["early_voltage_mean"] = float(np.mean(voltage[early_mask]))
        row["early_temp_mean"] = float(np.mean(temp[early_mask]))
        if early_mask.sum() > 1 and time_s[early_mask][-1] > time_s[early_mask][0]:
            row["early_voltage_slope"] = float(
                (voltage[early_mask][-1] - voltage[early_mask][0])
                / (time_s[early_mask][-1] - time_s[early_mask][0])
            )
        else:
            row["early_voltage_slope"] = np.nan

        rows.append(row)
        discharge_idx += 1

    return pd.DataFrame(rows)


def build_feature_table(data_dir: str) -> pd.DataFrame:
    mat_paths = sorted(Path(data_dir).glob("B*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No B*.mat files found in {data_dir}")
    tables = [extract_discharge_features(str(path)) for path in mat_paths]
    df = pd.concat(tables, ignore_index=True)
    return df


def model_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "battery_id",
        "capacity_ah",
        "soh_percent",
        # capacity_ah is label; charge_ah_integral is almost the same physical quantity.
        "charge_ah_integral",
        "energy_wh",
    }
    return [c for c in df.columns if c not in excluded]
