from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .model import SOHRegressor
from .splits import Scenario


@dataclass(frozen=True)
class ExperimentResult:
    metrics: dict[str, float | int | str]
    selected_features: list[str]
    pcc_table: pd.DataFrame
    predictions: pd.DataFrame
    history: pd.DataFrame


def run_experiment(
    scenario: Scenario,
    feature_cols: list[str],
    top_k: int,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> ExperimentResult:
    """执行单个场景（划分方式）的完整训练与评估流程。

    流程：设置随机种子 → Pearson 特征排序选 Top-K → 标准化+中位数填补 →
    构建并训练 MLP（AdamW + SmoothL1Loss + 早停）→ 在测试集上反标准化预测 →
    计算 MAE/RMSE/R2 → 组装 ExperimentResult。

    Args:
        scenario: 由 splits.py 构建的 Scenario 对象（含 train/val/test 划分）。
        feature_cols: 全部候选特征列名。
        top_k: 选择相关性最高的 top_k 个特征。
        seed: 随机种子。
        epochs: 最大训练轮数。
        patience: 早停耐心值。
        learning_rate: 学习率。

    Returns:
        ExperimentResult 包含指标、选中特征、PCC 表、预测结果、训练历史。
    """
    set_seed(seed)
    pcc_table = pearson_ranking(scenario.train, feature_cols, target_col="soh_percent")
    selected = pcc_table.head(top_k)["feature"].tolist()

    x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test, y_scaler = _prepare_arrays(
        scenario.train, scenario.val, scenario.test, selected
    )

    model = SOHRegressor(input_dim=len(selected))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    Xtr = torch.tensor(x_train, dtype=torch.float32)
    Xv = torch.tensor(x_val, dtype=torch.float32)
    ytr = torch.tensor(y_train_scaled, dtype=torch.float32)
    yv = torch.tensor(y_val_scaled, dtype=torch.float32)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(Xtr), ytr)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        history_rows.append(
            {
                "scenario": scenario.name,
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss),
            }
        )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_test, dtype=torch.float32)).numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    metrics = {
        "case": scenario.case,
        "source_battery": scenario.source_battery,
        "target_battery": scenario.target_battery,
        "scenario": scenario.name,
        "n_train": int(len(scenario.train)),
        "n_val": int(len(scenario.val)),
        "n_test": int(len(scenario.test)),
        "top_k_features": int(top_k),
        "input_dim": int(len(selected)),
        "hidden_layers": "64,32",
        "output_dim": 1,
        "best_epoch": int(best_epoch),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "selected_features": "; ".join(selected),
    }

    predictions = scenario.test[["battery_id", "cycle_index", "capacity_ah", "soh_percent"]].copy()
    predictions["scenario"] = scenario.name
    predictions["case"] = scenario.case
    predictions["source_battery"] = scenario.source_battery
    predictions["target_battery"] = scenario.target_battery
    predictions["pred_soh_percent"] = pred
    predictions["abs_error"] = np.abs(predictions["soh_percent"] - predictions["pred_soh_percent"])

    ranked = pcc_table.copy()
    ranked["selected"] = ranked["rank"] <= top_k
    ranked["scenario"] = scenario.name
    ranked["case"] = scenario.case
    ranked["source_battery"] = scenario.source_battery
    ranked["target_battery"] = scenario.target_battery

    history = pd.DataFrame(history_rows)
    history["case"] = scenario.case
    history["source_battery"] = scenario.source_battery
    history["target_battery"] = scenario.target_battery

    return ExperimentResult(metrics, selected, ranked, predictions, history)


def pearson_ranking(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """在训练集上计算各特征与目标的 Pearson 相关系数，并按绝对值降序排序。

    常数列或全 NaN 列的 PCC 记为 0。结果自动添加 rank 列（1 为最高相关）。

    Args:
        train_df: 训练集 DataFrame。
        feature_cols: 待评估的特征列名列表。
        target_col: 目标列名（通常为 "soh_percent"）。

    Returns:
        含 feature、pcc、abs_pcc、rank 列的 DataFrame。
    """
    y = train_df[target_col].astype(float)
    rows = []
    for col in feature_cols:
        x = train_df[col].astype(float)
        if x.nunique(dropna=True) <= 1 or x.isna().all():
            pcc = 0.0
        else:
            pcc = float(np.corrcoef(x.fillna(x.median()), y)[0, 1])
            if not np.isfinite(pcc):
                pcc = 0.0
        rows.append({"feature": col, "pcc": pcc, "abs_pcc": abs(pcc)})

    table = pd.DataFrame(rows).sort_values("abs_pcc", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def aggregate_case_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """按实验案例（A/B/C）聚合多个电池/迁移组合的 MAE/RMSE/R2 均值与标准差。"""
    return metrics_df.groupby("case", as_index=False).agg(
        MAE_mean=("MAE", "mean"),
        RMSE_mean=("RMSE", "mean"),
        R2_mean=("R2", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_std=("RMSE", "std"),
        R2_std=("R2", "std"),
    )


def set_seed(seed: int) -> None:
    """设置 Python、NumPy、PyTorch 的全局随机种子，并限制 PyTorch 单线程以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _prepare_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected: list[str],
):
    """对选定特征做中位数填补 + 标准化，对标签仅做标准化。

    填补器与缩放器均仅在训练集上 fit，验证/测试集仅 transform，
    严格避免数据泄露。

    Returns:
        x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test, y_scaler
    """
    imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(imputer.fit_transform(train_df[selected]))
    x_val = x_scaler.transform(imputer.transform(val_df[selected]))
    x_test = x_scaler.transform(imputer.transform(test_df[selected]))

    y_train = train_df["soh_percent"].to_numpy(dtype=float).reshape(-1, 1)
    y_val = val_df["soh_percent"].to_numpy(dtype=float).reshape(-1, 1)
    y_test = test_df["soh_percent"].to_numpy(dtype=float)

    y_train_scaled = y_scaler.fit_transform(y_train).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val).reshape(-1)
    return x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test, y_scaler

