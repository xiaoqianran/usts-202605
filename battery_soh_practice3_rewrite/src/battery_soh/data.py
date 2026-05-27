from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import BATTERIES, RATED_CAPACITY_AH

_trapezoid = getattr(np, "trapezoid", np.trapz)


def locate_data_dir(data_zip: Path, data_dir: Path) -> Path:
    """定位并返回包含 NASA 电池 .mat 文件的目录。

    如果 data_dir 下已存在 B0005.mat，则直接返回该目录；
    否则自动解压 data_zip 到 data_dir，并返回实际存放 .mat 文件的父目录。

    Args:
        data_zip: BatteryAgingARC.zip 压缩包路径。
        data_dir: 目标解压/存放目录。

    Returns:
        包含 B0005.mat 等文件的目录路径。

    Raises:
        FileNotFoundError: 压缩包不存在或解压后仍未找到数据文件。
    """
    if (data_dir / "B0005.mat").exists():
        return data_dir

    if not data_zip.exists():
        raise FileNotFoundError(
            f"Missing NASA dataset. Put BatteryAgingARC.zip at {data_zip} "
            f"or put extracted .mat files under {data_dir}."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_zip, "r") as zf:
        zf.extractall(data_dir)

    matches = list(data_dir.rglob("B0005.mat"))
    if not matches:
        raise FileNotFoundError("B0005.mat was not found after extracting the archive.")
    return matches[0].parent


def load_nasa_features(data_dir: Path, remove_capacity_spikes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 NASA 四个电池的放电循环特征表与数据清洗摘要。

    依次处理 B0005、B0006、B0007、B0018 四个电池的 .mat 文件，
    提取每个放电循环的统计特征，并可选地剔除容量局部尖峰异常点。

    Args:
        data_dir: 存放 .mat 文件的目录。
        remove_capacity_spikes: 是否启用容量局部尖峰剔除逻辑（默认 True）。

    Returns:
        features: 所有有效放电循环的特征 DataFrame（含 battery_id、cycle_index、容量、电压/电流/温度统计量等）。
        summary: 每个电池的原始/保留样本数、各类丢弃原因统计。
    """
    frames = []
    summaries = []
    for battery_id in BATTERIES:
        frame, summary = _extract_discharge_features(
            data_dir / f"{battery_id}.mat",
            remove_capacity_spikes=remove_capacity_spikes,
        )
        frames.append(frame)
        summaries.append(summary)
    features = pd.concat(frames, ignore_index=True)
    return features, pd.DataFrame(summaries)


def capacity_spike_report(features: pd.DataFrame) -> pd.DataFrame:
    """生成容量局部尖峰异常点的详细报告表。

    针对每个电池，检测并记录被判定为局部尖峰的所有 cycle，
    包含原始容量、局部中位数、偏离量、判定阈值等信息，用于后续分析与可视化。

    Args:
        features: load_nasa_features 返回的特征表（需包含 battery_id、cycle_index、capacity_ah 等列）。

    Returns:
        尖峰详情 DataFrame，每行对应一个被剔除的异常容量点。
    """
    rows = []
    for battery_id, group in features.groupby("battery_id", sort=True):
        group = group.sort_values("cycle_index").reset_index(drop=True)
        details = _capacity_spike_details(group["capacity_ah"].to_numpy(dtype=float))
        for detail in details:
            idx = int(detail["cycle_index"])
            row = group.iloc[idx]
            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle_index": int(row["cycle_index"]),
                    "global_cycle_index": int(row["global_cycle_index"]),
                    "capacity_ah": float(row["capacity_ah"]),
                    "soh_percent": float(row["soh_percent"]),
                    "local_median_capacity_ah": float(detail["local_median_capacity_ah"]),
                    "delta_from_local_median_ah": float(detail["delta_from_local_median_ah"]),
                    "threshold_ah": float(detail["threshold_ah"]),
                    "window_radius": int(detail["window_radius"]),
                }
            )
    return pd.DataFrame(rows)


def describe_dataset(features: pd.DataFrame, cleaning: pd.DataFrame) -> pd.DataFrame:
    """汇总每个电池清洗前后的样本统计与 SOH 范围描述。

    Args:
        features: 清洗后的特征表。
        cleaning: load_nasa_features 返回的第二个 DataFrame（每个电池的原始/保留计数）。

    Returns:
        每个电池的统计摘要 DataFrame，包含原始/保留 cycle 数、cycle 范围、容量/SOH 极值等。
    """
    rows = []
    for battery_id, group in features.groupby("battery_id", sort=True):
        clean = cleaning.loc[cleaning["battery_id"] == battery_id].iloc[0]
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows)


def model_input_columns(features: pd.DataFrame) -> list[str]:
    """返回可作为模型输入的特征列名列表（自动排除泄露与标签列）。

    排除以下列以防止标签泄露：
    - battery_id（电池标识）
    - capacity_ah、soh_percent（标签相关）
    - charge_ah_integral、energy_wh（与 capacity 高度线性相关）

    Args:
        features: 完整特征表。

    Returns:
        适合作为 MLP 输入的特征列名列表。
    """
    leakage_or_label = {
        "battery_id",
        "capacity_ah",
        "soh_percent",
        "charge_ah_integral",
        "energy_wh",
    }
    return [col for col in features.columns if col not in leakage_or_label]


def feature_reason(feature: str) -> str:
    """为给定的特征名称返回其业务含义与建模理由（中文）。

    用于生成报告中的特征解释文字。不同特征类别给出不同的专业解释。

    Args:
        feature: 特征列名。

    Returns:
        对应的中文业务解释字符串。
    """
    if feature.startswith("time_to_") or feature == "duration_s":
        return "放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。"
    if "voltage" in feature:
        return "电压平台、端电压和电压斜率描述放电曲线形状，老化会改变这些曲线特征。"
    if feature.startswith("temp") or "temp" in feature:
        return "温度特征反映放电热行为，电池老化导致内阻变化，进而影响温升。"
    if feature in {"cycle_index", "global_cycle_index"}:
        return "循环序号刻画退化进程，适合单电池趋势建模，但跨电池泛化时需谨慎解释。"
    if "current" in feature or "power" in feature:
        return "电流和功率统计量反映放电工况，可辅助区分不同退化状态下的输出特征。"
    if feature == "ambient_temperature":
        return "环境温度是电池实验的重要工况变量，会影响容量测量和放电过程。"
    return "该特征与 SOH 的 Pearson 相关性较高，因此进入候选输入集合。"


def _extract_discharge_features(mat_path: Path, remove_capacity_spikes: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从单个电池 .mat 文件中提取所有有效放电循环的特征（内部实现函数）。

    完整流程包括：加载 cycle → 筛选 discharge 类型 → 解析数据 → 长度/容量/时间合法性检查 →
    清洗时间序列 → 计算统计特征 → 可选的容量局部尖峰剔除。

    Args:
        mat_path: 单个电池的 .mat 文件完整路径。
        remove_capacity_spikes: 是否在最后阶段剔除容量局部尖峰。

    Returns:
        frame: 该电池所有保留放电循环的特征 DataFrame。
        summary: 各类丢弃原因的计数统计字典。
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing battery file: {mat_path}")

    battery_id = mat_path.stem
    cycles = _load_cycles(mat_path, battery_id)
    rows = []
    summary: dict[str, Any] = {
        "battery_id": battery_id,
        "raw_total_cycles": int(len(cycles)),
        "raw_discharge_cycles": 0,
        "kept_discharge_cycles": 0,
        "drop_parse_error": 0,
        "drop_short_or_length_mismatch": 0,
        "drop_nonfinite_or_bad_capacity": 0,
        "drop_bad_time": 0,
        "drop_capacity_local_spike": 0,
    }

    discharge_index = 0
    for global_index, cycle in enumerate(cycles):
        if str(cycle.type) != "discharge":
            continue
        summary["raw_discharge_cycles"] += 1

        parsed = _parse_cycle(cycle)
        if parsed is None:
            summary["drop_parse_error"] += 1
            continue
        time_s, voltage, current, temp, capacity = parsed

        if len(time_s) < 10 or not (len(time_s) == len(voltage) == len(current) == len(temp)):
            summary["drop_short_or_length_mismatch"] += 1
            continue
        if not np.isfinite(capacity) or capacity <= 0.5 or capacity > 2.2:
            summary["drop_nonfinite_or_bad_capacity"] += 1
            continue

        clean = _clean_series(time_s, voltage, current, temp)
        if clean is None:
            summary["drop_bad_time"] += 1
            continue
        time_s, voltage, current, temp = clean

        row = _make_feature_row(
            battery_id=battery_id,
            cycle_index=discharge_index,
            global_index=global_index,
            ambient_temperature=float(cycle.ambient_temperature),
            time_s=time_s,
            voltage=voltage,
            current=current,
            temp=temp,
            capacity=capacity,
        )
        rows.append(row)
        discharge_index += 1
        summary["kept_discharge_cycles"] += 1

    frame = pd.DataFrame(rows)
    if len(frame) and remove_capacity_spikes:
        spike_mask = _capacity_spike_mask(frame["capacity_ah"].to_numpy(dtype=float))
        summary["drop_capacity_local_spike"] = int(spike_mask.sum())
        if spike_mask.any():
            frame = frame.loc[~spike_mask].reset_index(drop=True)
            frame["cycle_index"] = np.arange(len(frame), dtype=int)
            summary["kept_discharge_cycles"] = int(len(frame))

    return frame, summary


def _load_cycles(mat_path: Path, battery_id: str):
    """使用 scipy.io.loadmat 加载单个电池 mat 文件中的 cycle 结构体数组。"""
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if battery_id not in mat:
        raise KeyError(f"{battery_id} not found in {mat_path}")
    return mat[battery_id].cycle


def _parse_cycle(cycle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """安全解析单个 cycle 结构体的测量数据字段。

    提取 Time、Voltage_measured、Current_measured、Temperature_measured、Capacity。
    任一字段缺失或类型转换失败均返回 None（由调用方统计丢弃原因）。

    Returns:
        (time_s, voltage, current, temp, capacity) 元组或 None。
    """
    try:
        data = cycle.data
        time_s = np.asarray(data.Time, dtype=float).reshape(-1)
        voltage = np.asarray(data.Voltage_measured, dtype=float).reshape(-1)
        current = np.asarray(data.Current_measured, dtype=float).reshape(-1)
        temp = np.asarray(data.Temperature_measured, dtype=float).reshape(-1)
        capacity = float(np.asarray(data.Capacity).squeeze())
    except Exception:
        return None
    return time_s, voltage, current, temp, capacity


def _clean_series(
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    temp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """对单 cycle 的时间序列进行完整性与单调性清洗。

    处理步骤：
    1. 剔除任一通道含 NaN/Inf 的采样点
    2. 按时间排序
    3. 去除时间重复点（保留首次出现）
    4. 最终长度与时间跨度合法性检查

    Returns:
        清洗后的四通道数组，或 None（样本太短/时间异常）。
    """
    finite = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temp)
    time_s, voltage, current, temp = time_s[finite], voltage[finite], current[finite], temp[finite]
    if len(time_s) < 10:
        return None

    order = np.argsort(time_s)
    time_s, voltage, current, temp = time_s[order], voltage[order], current[order], temp[order]
    time_s, unique_idx = np.unique(time_s, return_index=True)
    voltage, current, temp = voltage[unique_idx], current[unique_idx], temp[unique_idx]
    if len(time_s) < 10 or time_s[-1] <= time_s[0]:
        return None
    return time_s, voltage, current, temp


def _make_feature_row(
    battery_id: str,
    cycle_index: int,
    global_index: int,
    ambient_temperature: float,
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    temp: np.ndarray,
    capacity: float,
) -> dict[str, float | int | str]:
    """基于单 cycle 清洗后的原始波形计算全部结构化统计特征。

    包含：
    - 基础统计：均值、标准差、最值、范围、斜率
    - 能量积分：电荷量(Ah)、能量(Wh)、平均功率
    - 关键电压阈值到达时间（4.1V~3.5V）
    - 放电过程 25%/50%/75% 时刻的电压与温度
    - 放电早期（前25%时间）电压/温度均值与斜率

    Returns:
        适合直接转为 DataFrame 一行的字典。
    """
    duration_s = float(time_s[-1] - time_s[0])
    abs_current = np.abs(current)
    charge_ah_integral = float(_trapezoid(abs_current, time_s) / 3600.0)
    energy_wh = float(_trapezoid(voltage * abs_current, time_s) / 3600.0)

    row: dict[str, float | int | str] = {
        "battery_id": battery_id,
        "cycle_index": cycle_index,
        "global_cycle_index": int(global_index),
        "ambient_temperature": ambient_temperature,
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

    for threshold in (4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5):
        key = f"time_to_{str(threshold).replace('.', 'p')}v_s"
        row[key] = _first_time_voltage_below(time_s, voltage, threshold)

    for fraction in (0.25, 0.50, 0.75):
        suffix = str(int(fraction * 100))
        row[f"voltage_at_{suffix}pct_time"] = _value_at_fraction(time_s, voltage, fraction)
        row[f"temp_at_{suffix}pct_time"] = _value_at_fraction(time_s, temp, fraction)

    early_mask = time_s <= time_s[0] + 0.25 * duration_s
    row["early_voltage_mean"] = float(np.mean(voltage[early_mask]))
    row["early_temp_mean"] = float(np.mean(temp[early_mask]))
    row["early_voltage_slope"] = _segment_slope(time_s[early_mask], voltage[early_mask])
    return row


def _first_time_voltage_below(time_s: np.ndarray, voltage: np.ndarray, threshold: float) -> float:
    """线性插值计算电压首次降至指定阈值以下的时间（秒）。"""
    idx = np.where(voltage <= threshold)[0]
    if len(idx) == 0:
        return float("nan")
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
    """使用线性插值返回放电过程指定时间分位点（0~1）的对应物理量值。"""
    target_t = float(time_s[0] + fraction * (time_s[-1] - time_s[0]))
    return float(np.interp(target_t, time_s, values))


def _segment_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    """计算指定时间段内物理量的平均斜率（变化率）。时间非法时返回 NaN。"""
    if len(time_s) < 2 or time_s[-1] <= time_s[0]:
        return float("nan")
    return float((values[-1] - values[0]) / (time_s[-1] - time_s[0]))


def _capacity_spike_mask(capacity: np.ndarray, window_radius: int = 4) -> np.ndarray:
    """检测容量序列中的局部尖峰位置，返回布尔掩码。

    内部委托 _capacity_spike_details 实现检测逻辑。
    """
    details = _capacity_spike_details(capacity, window_radius=window_radius)
    mask = np.zeros(len(capacity), dtype=bool)
    for detail in details:
        mask[int(detail["cycle_index"])] = True
    return mask


def _capacity_spike_details(capacity: np.ndarray, window_radius: int = 4) -> list[dict[str, float | int]]:
    """使用稳健统计（中位数 + MAD）检测容量序列中的局部异常尖峰。

    对每个点，取前后 window_radius 个邻居计算中位数与 MAD，
    若当前值显著高于局部中位数（超过 3*robust_sigma 且至少 0.04Ah），则判定为尖峰。

    Args:
        capacity: 按 cycle 顺序排列的容量序列（Ah）。
        window_radius: 局部窗口半径（默认 4，即前后各 4 个 cycle）。

    Returns:
        尖峰详情列表，每个元素为包含 cycle_index、local_median 等信息的字典。
    """
    details: list[dict[str, float | int]] = []
    if len(capacity) < window_radius * 2 + 1:
        return details

    for idx, value in enumerate(capacity):
        start = max(0, idx - window_radius)
        stop = min(len(capacity), idx + window_radius + 1)
        neighbours = np.delete(capacity[start:stop], idx - start)
        if len(neighbours) < 4:
            continue

        median = float(np.median(neighbours))
        mad = float(np.median(np.abs(neighbours - median)))
        robust_sigma = 1.4826 * mad
        threshold = max(0.04, 3.0 * robust_sigma)
        if value > median + threshold:
            details.append(
                {
                    "cycle_index": int(idx),
                    "local_median_capacity_ah": median,
                    "delta_from_local_median_ah": float(value - median),
                    "threshold_ah": threshold,
                    "window_radius": int(window_radius),
                }
            )
    return details
