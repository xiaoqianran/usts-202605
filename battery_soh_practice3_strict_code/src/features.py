from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

# NumPy compatibility: trapezoid (NumPy >=2.0) falls back to trapz (NumPy <2.0)
_trapezoid = getattr(np, "trapezoid", np.trapz)

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH = 2.0


def extract_zip_if_needed(data_zip: str, work_dir: str) -> Path:
    """Extract NASA BatteryAgingARC.zip and return the directory containing B0005.mat etc."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    existing = list(work.rglob("B0005.mat"))
    if existing:
        return existing[0].parent

    zip_path = Path(data_zip)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"找不到数据压缩包：{zip_path}\n"
            "请把 BatteryAgingARC.zip 放到 data/ 目录，或用 --data_zip 指定路径。"
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work)

    mats = list(work.rglob("B0005.mat"))
    if not mats:
        raise FileNotFoundError("解压后没有找到 B0005.mat，请确认压缩包是否为 NASA BatteryAgingARC 数据集。")
    return mats[0].parent


def _load_cycles(mat_path: Path):
    battery_id = mat_path.stem
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if battery_id not in mat:
        raise KeyError(f"{battery_id} not found in {mat_path}")
    return battery_id, mat[battery_id].cycle


def _first_time_voltage_below(time_s: np.ndarray, voltage: np.ndarray, threshold: float) -> float:
    idx = np.where(voltage <= threshold)[0]
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(time_s[0])
    t0, t1 = float(time_s[i - 1]), float(time_s[i])
    v0, v1 = float(voltage[i - 1]), float(voltage[i])
    if abs(v1 - v0) < 1e-12:
        return t1
    ratio = (threshold - v0) / (v1 - v0)
    return float(t0 + ratio * (t1 - t0))


def _value_at_fraction(time_s: np.ndarray, values: np.ndarray, fraction: float) -> float:
    target_t = float(time_s[0] + fraction * (time_s[-1] - time_s[0]))
    return float(np.interp(target_t, time_s, values))


def extract_discharge_features(mat_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Extract one row per discharge cycle.

    Label: SOH(%) = Capacity / 2.0Ah * 100.
    Anti-leakage: Capacity, charge_ah_integral and energy_wh are not used as model inputs.
    """
    battery_id, cycles = _load_cycles(mat_path)
    rows: List[Dict[str, float]] = []
    summary = {
        "battery_id": battery_id,
        "raw_total_cycles": int(len(cycles)),
        "raw_discharge_cycles": 0,
        "kept_discharge_cycles": 0,
        "drop_parse_error": 0,
        "drop_short_or_length_mismatch": 0,
        "drop_nonfinite_or_bad_capacity": 0,
        "drop_bad_time": 0,
    }

    discharge_index = 0
    for global_index, cycle in enumerate(cycles):
        if str(cycle.type) != "discharge":
            continue
        summary["raw_discharge_cycles"] += 1
        try:
            data = cycle.data
            time_s = np.asarray(data.Time, dtype=float).reshape(-1)
            voltage = np.asarray(data.Voltage_measured, dtype=float).reshape(-1)
            current = np.asarray(data.Current_measured, dtype=float).reshape(-1)
            temp = np.asarray(data.Temperature_measured, dtype=float).reshape(-1)
            capacity = float(np.asarray(data.Capacity).squeeze())
        except Exception:
            summary["drop_parse_error"] += 1
            continue

        if len(time_s) < 10 or not (len(time_s) == len(voltage) == len(current) == len(temp)):
            summary["drop_short_or_length_mismatch"] += 1
            continue
        if (not np.isfinite(capacity)) or capacity <= 0.5 or capacity > 2.2:
            summary["drop_nonfinite_or_bad_capacity"] += 1
            continue

        finite = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temp)
        time_s, voltage, current, temp = time_s[finite], voltage[finite], current[finite], temp[finite]
        if len(time_s) < 10:
            summary["drop_short_or_length_mismatch"] += 1
            continue

        order = np.argsort(time_s)
        time_s, voltage, current, temp = time_s[order], voltage[order], current[order], temp[order]
        time_s, unique_idx = np.unique(time_s, return_index=True)
        voltage, current, temp = voltage[unique_idx], current[unique_idx], temp[unique_idx]
        if len(time_s) < 10 or time_s[-1] <= time_s[0]:
            summary["drop_bad_time"] += 1
            continue

        duration_s = float(time_s[-1] - time_s[0])
        abs_current = np.abs(current)
        charge_ah_integral = float(_trapezoid(abs_current, time_s) / 3600.0)
        energy_wh = float(_trapezoid(voltage * abs_current, time_s) / 3600.0)

        row: Dict[str, float] = {
            "battery_id": battery_id,
            "cycle_index": discharge_index,
            "global_cycle_index": int(global_index),
            "ambient_temperature": float(cycle.ambient_temperature),
            "capacity_ah": capacity,
            "soh_percent": capacity / RATED_CAPACITY_AH * 100.0,
            "duration_s": duration_s,
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
            "voltage_slope": float((voltage[-1] - voltage[0]) / duration_s),
            "charge_ah_integral": charge_ah_integral,
            "energy_wh": energy_wh,
            "avg_power_w": float(energy_wh * 3600.0 / duration_s),
        }

        for threshold in [4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5]:
            key = f"time_to_{str(threshold).replace('.', 'p')}v_s"
            row[key] = _first_time_voltage_below(time_s, voltage, threshold)

        for fraction in [0.25, 0.50, 0.75]:
            suffix = str(int(fraction * 100))
            row[f"voltage_at_{suffix}pct_time"] = _value_at_fraction(time_s, voltage, fraction)
            row[f"temp_at_{suffix}pct_time"] = _value_at_fraction(time_s, temp, fraction)

        early_cut = time_s[0] + 0.25 * duration_s
        early_mask = time_s <= early_cut
        row["early_voltage_mean"] = float(np.mean(voltage[early_mask]))
        row["early_temp_mean"] = float(np.mean(temp[early_mask]))
        if early_mask.sum() >= 2 and time_s[early_mask][-1] > time_s[early_mask][0]:
            row["early_voltage_slope"] = float(
                (voltage[early_mask][-1] - voltage[early_mask][0]) /
                (time_s[early_mask][-1] - time_s[early_mask][0])
            )
        else:
            row["early_voltage_slope"] = np.nan

        rows.append(row)
        discharge_index += 1
        summary["kept_discharge_cycles"] += 1

    return pd.DataFrame(rows), summary


def build_feature_table(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, int]] = []
    for battery_id in BATTERIES:
        mat_path = data_path / f"{battery_id}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing {mat_path}")
        frame, summary = extract_discharge_features(mat_path)
        frames.append(frame)
        summaries.append(summary)
    features = pd.concat(frames, ignore_index=True)
    cleaning_summary = pd.DataFrame(summaries)
    return features, cleaning_summary


def make_data_description(features: pd.DataFrame, cleaning_summary: pd.DataFrame) -> pd.DataFrame:
    desc_rows = []
    for battery_id, group in features.groupby("battery_id"):
        clean = cleaning_summary[cleaning_summary["battery_id"] == battery_id].iloc[0]
        desc_rows.append({
            "battery_id": battery_id,
            "raw_total_cycles": int(clean["raw_total_cycles"]),
            "raw_discharge_cycles": int(clean["raw_discharge_cycles"]),
            "kept_discharge_cycles": int(clean["kept_discharge_cycles"]),
            "cycle_min": int(group["cycle_index"].min()),
            "cycle_max": int(group["cycle_index"].max()),
            "capacity_min_ah": float(group["capacity_ah"].min()),
            "capacity_max_ah": float(group["capacity_ah"].max()),
            "soh_min_percent": float(group["soh_percent"].min()),
            "soh_max_percent": float(group["soh_percent"].max()),
        })
    return pd.DataFrame(desc_rows)


def model_feature_columns(features: pd.DataFrame) -> List[str]:
    excluded = {
        "battery_id",
        "capacity_ah",
        "soh_percent",
        # These are removed to avoid near-label leakage.
        "charge_ah_integral",
        "energy_wh",
    }
    return [c for c in features.columns if c not in excluded]


def feature_reason(feature: str) -> str:
    if feature.startswith("time_to_") or feature == "duration_s":
        return "放电到指定电压阈值所需时间或总放电时长会随容量衰减缩短，能直接反映容量退化状态。"
    if feature.startswith("voltage") or "voltage" in feature:
        return "电压平台、起止电压和电压斜率会随电池老化改变，能够描述放电曲线形状变化。"
    if feature.startswith("temp") or "temp" in feature:
        return "温度变化反映放电过程中的热行为，电池老化会影响内阻和温升特征。"
    if feature in {"cycle_index", "global_cycle_index"}:
        return "循环序号反映电池使用进程，是退化趋势的重要辅助信息；报告中需说明其泛化局限。"
    if "current" in feature or "power" in feature:
        return "电流与功率统计量反映放电工况和能量释放特征，可辅助判断 SOH。"
    if feature == "ambient_temperature":
        return "环境温度影响放电过程和容量表现，是电池实验中的重要工况变量。"
    return "该特征与 SOH 的 Pearson 相关系数较高，因此被选作模型输入。"
