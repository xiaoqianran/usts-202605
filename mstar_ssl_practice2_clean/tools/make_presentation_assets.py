#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


EXPERIMENTS = {
    "full": ("01_supervised_full", "Full supervised"),
    "limited": ("02_supervised_10percent", "10% supervised"),
    "fixmatch": ("03_fixmatch_10percent", "FixMatch"),
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_runs(runs_dir: Path) -> dict:
    runs = {}
    for key, (folder, label) in EXPERIMENTS.items():
        run_dir = runs_dir / folder
        runs[key] = {
            "label": label,
            "dir": run_dir,
            "metrics": load_json(run_dir / "metrics.json"),
            "history": pd.read_csv(run_dir / "history.csv"),
            "confusion": pd.read_csv(run_dir / "confusion_matrix.csv", index_col=0),
        }
    return runs


def accuracy_figure(runs: dict, out_dir: Path) -> None:
    labels = [runs[k]["label"] for k in EXPERIMENTS]
    acc = [runs[k]["metrics"]["best_test_acc"] * 100 for k in EXPERIMENTS]
    full, limited, fixmatch = acc
    drop = full - limited
    gain = fixmatch - limited
    recovery = gain / drop * 100 if drop > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
    axes[0].bar(labels, acc, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Accuracy comparison")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for i, value in enumerate(acc):
        axes[0].text(i, value + 1, f"{value:.2f}%", ha="center", fontsize=8)

    axes[1].plot([0, 1, 2], acc, marker="o", color="#333333")
    axes[1].set_xticks([0, 1, 2], ["Full", "10% Sup.", "FixMatch"])
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Accuracy drop and recovery")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].annotate(f"Drop {drop:.2f} pp", xy=(0.5, (full + limited) / 2), xytext=(0.1, limited + 7),
                     arrowprops={"arrowstyle": "->", "lw": 1}, fontsize=8)
    axes[1].annotate(f"Recover {gain:.2f} pp\n{recovery:.1f}% of loss", xy=(1.5, (limited + fixmatch) / 2),
                     xytext=(1.25, fixmatch + 8), arrowprops={"arrowstyle": "->", "lw": 1}, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_recovery.png", bbox_inches="tight")
    plt.close(fig)


def curves_figure(runs: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=200)
    for key, run in runs.items():
        hist = run["history"]
        label = run["label"]
        train_acc_col = "train_acc" if "train_acc" in hist.columns else "labeled_train_acc"
        axes[0, 0].plot(hist["epoch"], hist["train_loss"], label=label)
        axes[0, 1].plot(hist["epoch"], hist["test_loss"], label=label)
        axes[1, 0].plot(hist["epoch"], hist[train_acc_col] * 100, label=label)
        axes[1, 1].plot(hist["epoch"], hist["test_acc"] * 100, label=label)
    titles = ["Train loss", "Test loss", "Train accuracy", "Test accuracy"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_ylabel("%")
    axes[1, 1].set_ylabel("%")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def confusion_figure(runs: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=200)
    for ax, key in zip(axes, EXPERIMENTS):
        run = runs[key]
        matrix = run["confusion"].to_numpy(dtype=float)
        row_sum = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(run["label"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(out_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def flowchart(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=200)
    ax.axis("off")
    nodes = [
        (0.04, 0.55, "Labeled image\nweak aug."),
        (0.30, 0.55, "Supervised loss\nCross Entropy"),
        (0.04, 0.15, "Unlabeled image\nweak/strong aug."),
        (0.30, 0.15, "Weak prediction\npseudo label"),
        (0.56, 0.15, "Confidence filter\np >= threshold"),
        (0.76, 0.35, "Strong prediction\nconsistency loss"),
    ]
    for x, y, text in nodes:
        patch = FancyBboxPatch((x, y), 0.18, 0.22, boxstyle="round,pad=0.02", fc="#F7F7F7", ec="#333333")
        ax.add_patch(patch)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=9)
    arrows = [((0.22, 0.66), (0.30, 0.66)), ((0.22, 0.26), (0.30, 0.26)), ((0.48, 0.26), (0.56, 0.26)), ((0.74, 0.26), (0.76, 0.42)), ((0.48, 0.66), (0.76, 0.52))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, lw=1.2))
    ax.text(0.89, 0.52, "Total loss = Lx + lambda_u Lu", ha="center", va="center", fontsize=10)
    fig.savefig(out_dir / "fixmatch_flowchart.png", bbox_inches="tight")
    plt.close(fig)


def summary_markdown(runs: dict, out_dir: Path) -> None:
    rows = []
    for key, run in runs.items():
        metrics = run["metrics"]
        rows.append(
            f"| {run['label']} | {metrics['num_labeled']} | "
            f"{metrics.get('num_unlabeled', metrics.get('num_unlabeled_unused', 0))} | "
            f"{metrics['best_test_acc'] * 100:.2f}% | {metrics['best_epoch']} |"
        )
    content = "\n".join(
        [
            "# 实践二答辩素材摘要",
            "",
            "| 实验 | 有标签样本 | 无标签/未使用样本 | 最优测试精度 | 最优 epoch |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "建议答辩顺序：先说明全标签基线，再说明标签减少到 10% 后的精度下降，最后展示 FixMatch 如何利用剩余无标签样本恢复精度。",
            "",
            "可插入图片：`accuracy_recovery.png`、`training_curves.png`、`confusion_matrices.png`、`fixmatch_flowchart.png`。",
        ]
    )
    (out_dir / "presentation_summary.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="presentation_assets")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.runs))
    accuracy_figure(runs, out_dir)
    curves_figure(runs, out_dir)
    confusion_figure(runs, out_dir)
    flowchart(out_dir)
    summary_markdown(runs, out_dir)
    print(f"Presentation assets written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
