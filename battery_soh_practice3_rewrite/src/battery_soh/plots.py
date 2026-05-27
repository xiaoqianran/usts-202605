from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

plt.rcParams["axes.unicode_minus"] = False


def plot_capacity_degradation(features: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for battery_id, group in features.groupby("battery_id", sort=True):
        group = group.sort_values("cycle_index")
        ax.plot(group["cycle_index"], group["soh_percent"], linewidth=1.3, label=battery_id)
    ax.axhline(70, linestyle="--", linewidth=1, label="EOL 70% SOH")
    ax.set_xlabel("Discharge cycle index")
    ax.set_ylabel("SOH (%)")
    ax.set_title("Cleaned NASA battery capacity degradation")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_capacity_spikes(raw_features: pd.DataFrame, spike_report: pd.DataFrame, out_path: Path) -> None:
    batteries = sorted(raw_features["battery_id"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    axes = axes.reshape(-1)
    for ax, battery_id in zip(axes, batteries):
        group = raw_features[raw_features["battery_id"] == battery_id].sort_values("cycle_index")
        spikes = spike_report[spike_report["battery_id"] == battery_id]
        ax.plot(group["cycle_index"], group["capacity_ah"], marker="o", markersize=2.5, linewidth=1, label="Raw capacity")
        if len(spikes):
            ax.scatter(
                spikes["cycle_index"],
                spikes["capacity_ah"],
                s=48,
                marker="x",
                linewidths=1.8,
                color="#d62728",
                label="Removed local spike",
            )
        ax.axhline(1.4, linestyle="--", linewidth=1, color="#666666", label="EOL 1.4Ah")
        ax.set_title(f"{battery_id}: local capacity spike check")
        ax.set_xlabel("Discharge cycle index")
        ax.set_ylabel("Capacity (Ah)")
        ax.grid(True, linewidth=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_split_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axis("off")
    rows = [
        ("A. Random split", "Single battery: 60% train, 20% validation, 20% test", [("Train 60%", 0.6), ("Val 20%", 0.2), ("Test 20%", 0.2)]),
        ("B. Chronological split", "Single battery: first 60% cycles for development, last 40% for test", [("First 60%", 0.6), ("Last 40%", 0.4)]),
        ("C. Transfer split", "Source battery + target first 10% for adaptation; target last 90% for test", [("Source", 0.50), ("Target 10%", 0.15), ("Target 90%", 0.35)]),
    ]
    colors = ["#6baed6", "#fdae6b", "#74c476"]
    for idx, (title, note, segs) in enumerate(rows):
        y = 0.75 - idx * 0.28
        ax.text(0.03, y + 0.07, title, fontsize=13, fontweight="bold", va="center")
        ax.text(0.03, y + 0.02, note, fontsize=10, va="center")
        x = 0.43
        total_width = 0.48
        total = sum(frac for _, frac in segs)
        for label, frac in segs:
            w = total_width * frac / total
            ax.add_patch(Rectangle((x, y - 0.03), w, 0.12, facecolor=colors[len(label) % 3], edgecolor="black", alpha=0.50))
            ax.text(x + w / 2, y + 0.03, label, ha="center", va="center", fontsize=9)
            x += w
        ax.add_patch(FancyArrowPatch((0.38, y + 0.03), (0.425, y + 0.03), arrowstyle="->", mutation_scale=14, linewidth=1.1))
    ax.text(0.5, 0.95, "Three required train/validation/test protocols", ha="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pcc_heatmap(train_df: pd.DataFrame, pcc_table: pd.DataFrame, top_k: int, out_path: Path) -> list[str]:
    top_features = pcc_table.sort_values("rank").head(top_k)["feature"].tolist()
    cols = top_features + ["soh_percent"]
    corr = train_df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"PCC heatmap of selected top {top_k} features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top_features


def plot_mlp_structure(top_k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    ax.axis("off")
    layers = [
        (f"Input\n{top_k} features", 0.08),
        (f"Linear\n{top_k} -> 64", 0.27),
        ("ReLU\nDropout 0.05", 0.46),
        ("Linear + ReLU\n64 -> 32", 0.65),
        ("Output\n32 -> 1", 0.84),
    ]
    for label, x in layers:
        ax.add_patch(Rectangle((x - 0.075, 0.34), 0.15, 0.34, fill=False, linewidth=1.5))
        ax.text(x, 0.51, label, ha="center", va="center", fontsize=10)
    for idx in range(len(layers) - 1):
        ax.add_patch(FancyArrowPatch((layers[idx][1] + 0.075, 0.51), (layers[idx + 1][1] - 0.075, 0.51), arrowstyle="->", mutation_scale=14, linewidth=1.2))
    ax.text(0.5, 0.84, "MLP regression model for SOH (%)", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.20, "Optimizer: AdamW | Loss: SmoothL1Loss | Early stopping on validation loss", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    data = metrics_df.reset_index(drop=True)
    labels = data.apply(
        lambda row: f"C-{row['source_battery']}->{row['target_battery']}"
        if row["case"] == "C"
        else f"{row['case']}-{row['target_battery']}",
        axis=1,
    )
    x = np.arange(len(data))
    fig_width = max(12, len(data) * 0.75)
    fig, ax1 = plt.subplots(figsize=(fig_width, 5.5))
    ax1.bar(x - 0.2, data["MAE"], width=0.4, label="MAE")
    ax1.bar(x + 0.2, data["RMSE"], width=0.4, label="RMSE")
    ax1.set_ylabel("Error / SOH percentage points")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", linewidth=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, data["R2"], marker="o", linewidth=1.2, label="R2")
    ax2.set_ylabel("R2")
    ax2.set_ylim(min(0.0, float(data["R2"].min()) - 0.1), 1.05)
    ax2.legend(loc="upper right")
    ax1.set_title(f"MAE/RMSE/R2 comparison across {len(data)} runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_group_predictions(predictions: pd.DataFrame, metrics_df: pd.DataFrame, case_name: str, out_path: Path) -> None:
    sub_metrics = metrics_df[metrics_df["case"] == case_name]
    n_plots = len(sub_metrics)
    n_cols = min(4, max(1, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, (_, row) in zip(axes, sub_metrics.iterrows()):
        scenario = row["scenario"]
        data = predictions[predictions["scenario"] == scenario].sort_values("cycle_index")
        ax.plot(data["cycle_index"], data["soh_percent"], marker="o", markersize=2, linewidth=1, label="True")
        ax.plot(data["cycle_index"], data["pred_soh_percent"], marker="x", markersize=2, linewidth=1, label="Pred")
        title = (
            f"{row['source_battery']}->{row['target_battery']}"
            if case_name == "C"
            else str(row["target_battery"])
        )
        ax.set_title(f"{title} | MAE={row['MAE']:.3f}, R2={row['R2']:.3f}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, linewidth=0.3)
    for ax in axes[n_plots:]:
        ax.axis("off")
    axes[0].legend()
    fig.suptitle(f"Case {case_name}: true vs predicted SOH")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_prediction_curve(predictions: pd.DataFrame, scenario: str, out_path: Path) -> None:
    data = predictions[predictions["scenario"] == scenario].sort_values("cycle_index")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["cycle_index"], data["soh_percent"], marker="o", markersize=3, linewidth=1, label="True SOH")
    ax.plot(data["cycle_index"], data["pred_soh_percent"], marker="x", markersize=3, linewidth=1, label="Predicted SOH")
    ax.set_xlabel("Discharge cycle index")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"Predicted vs true SOH: {scenario}")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_loss_curve(history: pd.DataFrame, scenario: str, out_path: Path) -> None:
    data = history[history["scenario"] == scenario].sort_values("epoch")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(data["epoch"], data["train_loss"], label="Train loss")
    ax.plot(data["epoch"], data["val_loss"], label="Validation loss")
    if len(data):
        best_idx = data["val_loss"].idxmin()
        best_epoch = int(data.loc[best_idx, "epoch"])
        ax.axvline(best_epoch, linestyle="--", linewidth=1, label=f"Best epoch={best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 loss on scaled SOH")
    ax.set_title(f"Loss curve: {scenario}")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
