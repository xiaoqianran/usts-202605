#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.battery_soh.config import PipelineConfig
from src.battery_soh.data import (
    describe_dataset,
    feature_reason,
    load_nasa_features,
    locate_data_dir,
    model_input_columns,
)
from src.battery_soh.experiment import aggregate_case_metrics, run_experiment
from src.battery_soh.plots import (
    plot_capacity_degradation,
    plot_group_predictions,
    plot_loss_curve,
    plot_metrics,
    plot_mlp_structure,
    plot_pcc_heatmap,
    plot_prediction_curve,
    plot_split_schematic,
)
from src.battery_soh.presentation import write_presentation_summary
from src.battery_soh.splits import build_scenarios, summarize_splits


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Practice 3: NASA lithium-ion battery SOH prediction")
    parser.add_argument("--data_zip", type=Path, default=Path("data/BatteryAgingARC.zip"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--save_individual_plots", action="store_true")
    args = parser.parse_args()
    return PipelineConfig(
        data_zip=args.data_zip,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        save_individual_plots=args.save_individual_plots,
    )


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.asset_dir.mkdir(parents=True, exist_ok=True)

    data_dir = locate_data_dir(config.data_zip, config.data_dir)
    features, cleaning = load_nasa_features(data_dir)
    data_desc = describe_dataset(features, cleaning)
    feature_cols = model_input_columns(features)

    features.to_csv(config.output_dir / "01_features_all.csv", index=False, encoding="utf-8-sig")
    data_desc.to_csv(config.output_dir / "02_data_description.csv", index=False, encoding="utf-8-sig")
    cleaning.to_csv(config.output_dir / "03_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    plot_capacity_degradation(features, config.asset_dir / "01_capacity_degradation.png")

    scenarios = build_scenarios(features, seed=config.seed)
    split_summary = summarize_splits(scenarios)
    split_summary.to_csv(config.output_dir / "04_split_summary.csv", index=False, encoding="utf-8-sig")
    plot_split_schematic(config.asset_dir / "02_split_schematic.png")

    metrics_rows = []
    pcc_tables = []
    prediction_tables = []
    history_tables = []
    for scenario in scenarios:
        print(f"Running {scenario.name} ...", flush=True)
        result = run_experiment(
            scenario=scenario,
            feature_cols=feature_cols,
            top_k=config.top_k,
            seed=config.seed,
            epochs=config.epochs,
            patience=config.patience,
            learning_rate=config.learning_rate,
        )
        metrics_rows.append(result.metrics)
        pcc_tables.append(result.pcc_table)
        prediction_tables.append(result.predictions)
        history_tables.append(result.history)
        print(
            f"Done {scenario.name}: MAE={result.metrics['MAE']:.4f}, "
            f"RMSE={result.metrics['RMSE']:.4f}, R2={result.metrics['R2']:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(metrics_rows)
    pcc_all = pd.concat(pcc_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    history = pd.concat(history_tables, ignore_index=True)
    case_mean = aggregate_case_metrics(metrics)

    metrics.to_csv(config.output_dir / "05_metrics_summary_12runs.csv", index=False, encoding="utf-8-sig")
    case_mean.to_csv(config.output_dir / "06_metrics_by_case_mean.csv", index=False, encoding="utf-8-sig")
    pcc_all.to_csv(config.output_dir / "07_pcc_all_scenarios.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.output_dir / "09_predictions_all_scenarios.csv", index=False, encoding="utf-8-sig")
    history.to_csv(config.output_dir / "10_loss_history_all_scenarios.csv", index=False, encoding="utf-8-sig")

    representative = "A_random_B0005_60_20_20"
    representative_train = next(item.train for item in scenarios if item.name == representative)
    representative_pcc = pcc_all[pcc_all["scenario"] == representative].sort_values("rank")
    selected_features = plot_pcc_heatmap(
        representative_train,
        representative_pcc,
        config.top_k,
        config.asset_dir / "03_pcc_heatmap_topK.png",
    )
    top_feature_rows = []
    for feature in selected_features:
        row = representative_pcc[representative_pcc["feature"] == feature].iloc[0]
        top_feature_rows.append(
            {
                "rank": int(row["rank"]),
                "feature": feature,
                "pcc": float(row["pcc"]),
                "abs_pcc": float(row["abs_pcc"]),
                "reason": feature_reason(feature),
            }
        )
    top_features = pd.DataFrame(top_feature_rows)
    top_features.to_csv(config.output_dir / "08_topK_features_and_reasons.csv", index=False, encoding="utf-8-sig")

    plot_mlp_structure(config.top_k, config.asset_dir / "04_mlp_structure.png")
    plot_metrics(metrics, config.asset_dir / "05_metrics_comparison_12runs.png")
    plot_group_predictions(predictions, metrics, "A", config.asset_dir / "06_predictions_A_four_batteries.png")
    plot_group_predictions(predictions, metrics, "B", config.asset_dir / "07_predictions_B_four_batteries.png")
    plot_group_predictions(predictions, metrics, "C", config.asset_dir / "08_predictions_C_four_batteries.png")

    prediction_scenario = "B_chrono_B0005_first60_last40"
    plot_prediction_curve(predictions, prediction_scenario, config.asset_dir / "09_prediction_true_vs_pred_B0005_B.png")
    plot_loss_curve(history, prediction_scenario, config.asset_dir / "10_loss_curve_B0005_B.png")

    if config.save_individual_plots:
        individual_dir = config.asset_dir / "individual_runs"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for scenario in metrics["scenario"]:
            plot_prediction_curve(predictions, scenario, individual_dir / f"prediction_{scenario}.png")
            plot_loss_curve(history, scenario, individual_dir / f"loss_{scenario}.png")

    summary_path = write_presentation_summary(
        output_dir=config.output_dir,
        data_desc=data_desc,
        cleaning=cleaning,
        split_summary=split_summary,
        metrics=metrics,
        case_mean=case_mean,
        top_features=top_features,
    )

    print("\n=== 12 runs ===")
    print(metrics[["case", "source_battery", "target_battery", "n_train", "n_val", "n_test", "MAE", "RMSE", "R2"]].to_string(index=False))
    print("\n=== Mean by case ===")
    print(case_mean.to_string(index=False))
    print(f"\nOutputs: {config.output_dir.resolve()}")
    print(f"Presentation summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()

