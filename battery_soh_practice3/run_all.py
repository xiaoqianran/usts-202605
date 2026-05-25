#!/usr/bin/env python3
"""
Practice 3: Deep learning application for lithium-ion battery SOH prediction.

This version follows the revised requirement strictly:
- Case A is run independently on B0005, B0006, B0007, B0018.
- Case B is run independently on B0005, B0006, B0007, B0018.
- Case C is also run four times, once for each target battery.
  Every C run uses only one source battery plus the first 10% cycles of that target battery;
  multiple source batteries are never merged.
"""
import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import extract_zip_if_needed, build_feature_table, model_feature_columns
from src.splits import split_a_random_single, split_b_chrono_single, split_c_transfer
from src.train_eval import train_and_evaluate


BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# For case C, each target battery is tested once.  Each run uses one source battery only.
# This avoids the earlier problem of using only a single C scenario or mixing several batteries together.
C_SOURCE_FOR_TARGET: Dict[str, str] = {
    "B0005": "B0007",  # source -> target
    "B0006": "B0007",
    "B0007": "B0005",
    "B0018": "B0005",
}


def plot_capacity(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for battery_id, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_index")
        plt.plot(g["cycle_index"], g["soh_percent"], label=battery_id)
    plt.axhline(70, linestyle="--", linewidth=1, label="EOL 70%")
    plt.xlabel("Discharge cycle index")
    plt.ylabel("SOH (%)")
    plt.title("Capacity degradation curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pcc(pcc_table: pd.DataFrame, scenario: str, out_path: Path, top_n: int = 12) -> None:
    top = pcc_table.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(top["feature"], top["pcc"])
    plt.xlabel("Pearson correlation coefficient with SOH")
    plt.title(f"Top PCC features - {scenario}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_predictions(pred_df: pd.DataFrame, scenario: str, out_path: Path) -> None:
    data = pred_df.sort_values(["battery_id", "cycle_index"])
    plt.figure(figsize=(8, 5))
    plt.plot(data["cycle_index"], data["soh_percent"], marker="o", markersize=3, linewidth=1, label="True SOH")
    plt.plot(data["cycle_index"], data["pred_soh_percent"], marker="x", markersize=3, linewidth=1, label="Predicted SOH")
    plt.xlabel("Discharge cycle index")
    plt.ylabel("SOH (%)")
    plt.title(f"Prediction result - {scenario}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_group_predictions(pred_all: pd.DataFrame, metrics_df: pd.DataFrame, case_name: str, out_path: Path) -> None:
    subset_metrics = metrics_df[metrics_df["case"] == case_name].copy()
    if subset_metrics.empty:
        return
    scenarios = subset_metrics["scenario"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.reshape(-1)
    for ax, scenario in zip(axes, scenarios):
        data = pred_all[pred_all["scenario"] == scenario].sort_values("cycle_index")
        row = subset_metrics[subset_metrics["scenario"] == scenario].iloc[0]
        ax.plot(data["cycle_index"], data["soh_percent"], marker="o", markersize=2, linewidth=1, label="True")
        ax.plot(data["cycle_index"], data["pred_soh_percent"], marker="x", markersize=2, linewidth=1, label="Pred")
        ax.set_title(f"{row['target_battery']} | MAE={row['MAE']:.3f}, R2={row['R2']:.3f}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, linewidth=0.3)
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    axes[0].legend(loc="best")
    fig.suptitle(f"Case {case_name}: true vs predicted SOH for four independent runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    data = metrics_df.copy()
    x = np.arange(len(data))
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - 0.2, data["MAE"], width=0.4, label="MAE")
    ax1.bar(x + 0.2, data["RMSE"], width=0.4, label="RMSE")
    ax1.set_ylabel("Error / SOH percentage points")
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["case"] + "-" + data["target_battery"].str.replace("B00", ""), rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, data["R2"], marker="o", linewidth=1.2, label="R2")
    ax2.set_ylabel("R2")
    ax2.set_ylim(min(0.0, data["R2"].min() - 0.1), 1.05)
    ax2.legend(loc="upper right")
    plt.title("Metrics of 12 independent SOH prediction experiments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_scenarios(df: pd.DataFrame, seed: int) -> Dict[str, Tuple[str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Return scenario_name -> (case, source, target, train_df, val_df, test_df)."""
    scenarios = {}
    for battery_id in BATTERIES:
        name = f"A_random_{battery_id}_60_20_20"
        train_df, val_df, test_df = split_a_random_single(df, battery_id=battery_id, seed=seed)
        scenarios[name] = ("A", battery_id, battery_id, train_df, val_df, test_df)

    for battery_id in BATTERIES:
        name = f"B_chrono_{battery_id}_first60_last40"
        train_df, val_df, test_df = split_b_chrono_single(df, battery_id=battery_id)
        scenarios[name] = ("B", battery_id, battery_id, train_df, val_df, test_df)

    for target_battery in BATTERIES:
        source_battery = C_SOURCE_FOR_TARGET[target_battery]
        name = f"C_transfer_{source_battery}_to_{target_battery}_target10"
        train_df, val_df, test_df = split_c_transfer(df, source_battery=source_battery, target_battery=target_battery)
        scenarios[name] = ("C", source_battery, target_battery, train_df, val_df, test_df)

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice 3: NASA battery SOH prediction with PyTorch")
    parser.add_argument("--data_zip", type=str, default="./data/BatteryAgingARC.zip", help="Path to BatteryAgingARC.zip")
    parser.add_argument("--work_dir", type=str, default="./data/extracted", help="Directory for extracted .mat files")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--top_k", type=int, default=8, help="Number of PCC-selected features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_individual_plots", action="store_true", help="Also save one PCC and one prediction plot for every scenario")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = extract_zip_if_needed(args.data_zip, args.work_dir)
    df = build_feature_table(str(data_dir))
    df.to_csv(output_dir / "features_all.csv", index=False, encoding="utf-8-sig")
    plot_capacity(df, output_dir / "capacity_degradation.png")

    feature_cols = model_feature_columns(df)
    scenarios = make_scenarios(df, args.seed)

    metrics_rows = []
    split_rows = []
    pcc_tables = []
    pred_tables = []

    for scenario, (case_name, source_battery, target_battery, train_df, val_df, test_df) in scenarios.items():
        print(f"Running {scenario} ...", flush=True)
        for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            split_rows.append({
                "scenario": scenario,
                "case": case_name,
                "source_battery": source_battery,
                "target_battery": target_battery,
                "split": split_name,
                "n": len(split_df),
                "batteries": ",".join(sorted(split_df["battery_id"].unique())),
                "cycle_min": int(split_df["cycle_index"].min()) if len(split_df) else None,
                "cycle_max": int(split_df["cycle_index"].max()) if len(split_df) else None,
            })

        result = train_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            scenario=scenario,
            top_k=args.top_k,
            seed=args.seed,
            epochs=600,
            patience=60,
        )
        metric = result.metrics.copy()
        metric["case"] = case_name
        metric["source_battery"] = source_battery
        metric["target_battery"] = target_battery
        metric["selected_features"] = "; ".join(result.selected_features)
        metrics_rows.append(metric)
        print(f"Done {scenario}: MAE={metric['MAE']:.4f}, RMSE={metric['RMSE']:.4f}, R2={metric['R2']:.4f}", flush=True)

        pcc = result.pcc_table.copy()
        pcc["scenario"] = scenario
        pcc["case"] = case_name
        pcc["source_battery"] = source_battery
        pcc["target_battery"] = target_battery
        pcc_tables.append(pcc)

        pred = result.predictions.copy()
        pred["case"] = case_name
        pred["source_battery"] = source_battery
        pred["target_battery"] = target_battery
        pred_tables.append(pred)

        if args.save_individual_plots:
            safe = scenario.replace("/", "_")
            plot_pcc(result.pcc_table, scenario, output_dir / f"pcc_{safe}.png")
            plot_predictions(result.predictions, scenario, output_dir / f"predictions_{safe}.png")

    metrics_df = pd.DataFrame(metrics_rows)
    # Put identification columns first.
    front = ["case", "source_battery", "target_battery", "scenario"]
    metrics_df = metrics_df[front + [c for c in metrics_df.columns if c not in front]]
    splits_df = pd.DataFrame(split_rows)
    pcc_all = pd.concat(pcc_tables, ignore_index=True)
    pred_all = pd.concat(pred_tables, ignore_index=True)

    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    splits_df.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    pcc_all.to_csv(output_dir / "pcc_all_scenarios.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(output_dir / "predictions_all_scenarios.csv", index=False, encoding="utf-8-sig")

    plot_group_predictions(pred_all, metrics_df, "A", output_dir / "predictions_A_four_batteries.png")
    plot_group_predictions(pred_all, metrics_df, "B", output_dir / "predictions_B_four_batteries.png")
    plot_group_predictions(pred_all, metrics_df, "C", output_dir / "predictions_C_four_batteries.png")
    plot_metrics(metrics_df, output_dir / "metrics_12_runs.png")

    print("\n=== Metrics: 12 independent runs ===")
    display_cols = ["case", "source_battery", "target_battery", "n_train", "n_val", "n_test", "MAE", "RMSE", "R2"]
    print(metrics_df[display_cols].to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
