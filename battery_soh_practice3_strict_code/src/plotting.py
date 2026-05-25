from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd

plt.rcParams["axes.unicode_minus"] = False


def plot_capacity_degradation(features: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for battery_id, group in features.groupby("battery_id"):
        group = group.sort_values("cycle_index")
        plt.plot(group["cycle_index"], group["soh_percent"], linewidth=1.2, label=battery_id)
    plt.axhline(70, linestyle="--", linewidth=1, label="EOL 70% SOH")
    plt.xlabel("Discharge cycle index")
    plt.ylabel("SOH (%)")
    plt.title("Original capacity degradation curves after cleaning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_split_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axis("off")
    rows = [
        ("A. Random 60/20/20 split", "For each single battery independently", [("Train 60%", 0.6), ("Val 20%", 0.2), ("Test 20%", 0.2)], "B0005/B0006/B0007/B0018: 4 runs"),
        ("B. Chronological first 60% / last 40%", "Sort cycles by cycle index", [("First 60% train/val", 0.6), ("Last 40% test", 0.4)], "B0005/B0006/B0007/B0018: 4 runs"),
        ("C. Transfer setting", "One source battery + target first 10%", [("Source train/val", 0.50), ("Target first 10% train", 0.15), ("Target last 90% test", 0.35)], "Four target batteries: 4 runs; no mixed test"),
    ]
    y_positions = [0.78, 0.48, 0.18]
    for (title, subtitle, segs, note), y in zip(rows, y_positions):
        ax.text(0.02, y + 0.09, title, fontsize=13, fontweight="bold", va="center")
        ax.text(0.02, y + 0.035, subtitle, fontsize=10, va="center")
        start = 0.35
        total_w = 0.44
        x = start
        total = sum(v for _, v in segs)
        for label, frac in segs:
            w = total_w * frac / total
            ax.add_patch(Rectangle((x, y), w, 0.12, fill=False, linewidth=1.5))
            ax.text(x + w / 2, y + 0.06, label, ha="center", va="center", fontsize=9)
            x += w
        ax.add_patch(FancyArrowPatch((0.31, y + 0.06), (0.345, y + 0.06), arrowstyle="->", mutation_scale=12, linewidth=1.0))
        ax.text(0.82, y + 0.06, note, fontsize=9, va="center", wrap=True)
    ax.text(0.5, 0.95, "Three data split protocols used in Practice 3", ha="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pcc_heatmap(train_df: pd.DataFrame, pcc_table: pd.DataFrame, top_k: int, out_path: Path) -> List[str]:
    top_features = pcc_table.head(top_k)["feature"].tolist()
    cols = top_features + ["soh_percent"]
    corr = train_df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"PCC heatmap: selected top K={top_k} features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top_features


def plot_mlp_structure(top_k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    layers = [
        (f"Input\nK={top_k} features", 0.08),
        (f"Linear\n{top_k} -> 64", 0.27),
        ("ReLU + Dropout\np=0.05", 0.46),
        ("Linear + ReLU\n64 -> 32", 0.65),
        ("Output\n32 -> 1 SOH", 0.84),
    ]
    for label, x in layers:
        ax.add_patch(Rectangle((x - 0.075, 0.35), 0.15, 0.32, fill=False, linewidth=1.5))
        ax.text(x, 0.51, label, ha="center", va="center", fontsize=10)
    for i in range(len(layers) - 1):
        ax.add_patch(FancyArrowPatch((layers[i][1] + 0.075, 0.51), (layers[i+1][1] - 0.075, 0.51), arrowstyle="->", mutation_scale=14, linewidth=1.2))
    ax.text(0.5, 0.83, "MLP regression network", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.20, "Loss: SmoothL1Loss; Optimizer: AdamW; Output: SOH percentage", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(metrics_df: pd.DataFrame, out_path: Path) -> None:
    data = metrics_df.copy().reset_index(drop=True)
    labels = data["case"] + "-" + data["target_battery"].str.replace("B00", "", regex=False)
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - 0.2, data["MAE"], width=0.4, label="MAE")
    ax1.bar(x + 0.2, data["RMSE"], width=0.4, label="RMSE")
    ax1.set_ylabel("Error / SOH percentage points")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, data["R2"], marker="o", linewidth=1.2, label="R²")
    ax2.set_ylabel("R²")
    ax2.set_ylim(min(0.0, data["R2"].min() - 0.1), 1.05)
    ax2.legend(loc="upper right")
    plt.title("MAE/RMSE/R² comparison for 12 independent runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_prediction_curve(predictions: pd.DataFrame, scenario: str, out_path: Path) -> None:
    data = predictions[predictions["scenario"] == scenario].sort_values("cycle_index")
    plt.figure(figsize=(8, 5))
    plt.plot(data["cycle_index"], data["soh_percent"], marker="o", markersize=3, linewidth=1, label="True SOH")
    plt.plot(data["cycle_index"], data["pred_soh_percent"], marker="x", markersize=3, linewidth=1, label="Predicted SOH")
    plt.xlabel("Discharge cycle index")
    plt.ylabel("SOH (%)")
    plt.title(f"Predicted vs true SOH: {scenario}")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_loss_curve(history: pd.DataFrame, scenario: str, out_path: Path) -> None:
    data = history[history["scenario"] == scenario].sort_values("epoch")
    plt.figure(figsize=(8, 4.2))
    plt.plot(data["epoch"], data["train_loss"], label="Train loss")
    plt.plot(data["epoch"], data["val_loss"], label="Validation loss")
    best_idx = data["val_loss"].idxmin()
    best_epoch = int(data.loc[best_idx, "epoch"])
    plt.axvline(best_epoch, linestyle="--", linewidth=1, label=f"Best epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 loss (scaled SOH)")
    plt.title(f"Loss curves: {scenario}")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_group_predictions(pred_all: pd.DataFrame, metrics_df: pd.DataFrame, case_name: str, out_path: Path) -> None:
    sub_metrics = metrics_df[metrics_df["case"] == case_name].copy()
    if sub_metrics.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    for ax, (_, row) in zip(axes, sub_metrics.iterrows()):
        scenario = row["scenario"]
        data = pred_all[pred_all["scenario"] == scenario].sort_values("cycle_index")
        ax.plot(data["cycle_index"], data["soh_percent"], marker="o", markersize=2, linewidth=1, label="True")
        ax.plot(data["cycle_index"], data["pred_soh_percent"], marker="x", markersize=2, linewidth=1, label="Pred")
        ax.set_title(f"{row['target_battery']} | MAE={row['MAE']:.3f}, R²={row['R2']:.3f}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, linewidth=0.3)
    for ax in axes[len(sub_metrics):]:
        ax.axis("off")
    axes[0].legend()
    fig.suptitle(f"Case {case_name}: true vs predicted SOH")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
