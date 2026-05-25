from dataclasses import dataclass
from typing import Dict, List
import random

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .model import SOHMLP


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    selected_features: List[str]
    pcc_table: pd.DataFrame
    predictions: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def pearson_select(train_df: pd.DataFrame, feature_cols: List[str], target_col: str, top_k: int) -> pd.DataFrame:
    rows = []
    y = train_df[target_col].astype(float)
    for col in feature_cols:
        x = train_df[col].astype(float)
        if x.nunique(dropna=True) <= 1 or x.isna().all():
            r = 0.0
        else:
            x = x.fillna(x.median())
            r = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(r):
                r = 0.0
        rows.append({"feature": col, "pcc": r, "abs_pcc": abs(r)})
    table = pd.DataFrame(rows).sort_values("abs_pcc", ascending=False).reset_index(drop=True)
    table["selected"] = False
    table.loc[: max(0, top_k - 1), "selected"] = True
    return table


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scenario: str,
    top_k: int = 8,
    seed: int = 42,
    epochs: int = 1000,
    patience: int = 80,
    learning_rate: float = 2e-3,
) -> TrainResult:
    set_seed(seed)

    pcc_table = pearson_select(train_df, feature_cols, "soh_percent", top_k=top_k)
    selected = pcc_table.loc[pcc_table["selected"], "feature"].tolist()

    imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(imputer.fit_transform(train_df[selected]))
    X_val = x_scaler.transform(imputer.transform(val_df[selected]))
    X_test = x_scaler.transform(imputer.transform(test_df[selected]))

    y_train = train_df["soh_percent"].to_numpy(dtype=float).reshape(-1, 1)
    y_val = val_df["soh_percent"].to_numpy(dtype=float).reshape(-1, 1)
    y_test = test_df["soh_percent"].to_numpy(dtype=float)
    y_train_scaled = y_scaler.fit_transform(y_train).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val).reshape(-1)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train_scaled, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val_scaled, dtype=torch.float32)

    model = SOHMLP(input_dim=Xtr.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    best_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    metrics = {
        "scenario": scenario,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "top_k_features": int(top_k),
        "best_epoch": int(best_epoch),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }

    predictions = test_df[["battery_id", "cycle_index", "capacity_ah", "soh_percent"]].copy()
    predictions["scenario"] = scenario
    predictions["pred_soh_percent"] = pred
    predictions["abs_error"] = np.abs(predictions["soh_percent"] - predictions["pred_soh_percent"])

    return TrainResult(metrics=metrics, selected_features=selected, pcc_table=pcc_table, predictions=predictions)
