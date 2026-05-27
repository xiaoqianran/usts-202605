#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.battery_soh.config import PipelineConfig
from src.battery_soh.data import (
    capacity_spike_report,
    describe_dataset,
    feature_reason,
    load_nasa_features,
    locate_data_dir,
    model_input_columns,
)
from src.battery_soh.experiment import aggregate_case_metrics, run_experiment
from src.battery_soh.plots import (
    plot_capacity_degradation,
    plot_capacity_spikes,
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
    raw_features, _ = load_nasa_features(data_dir, remove_capacity_spikes=False)
    spikes = capacity_spike_report(raw_features)
    spikes.to_csv(config.output_dir / "03b_capacity_spike_report.csv", index=False, encoding="utf-8-sig")
    plot_capacity_spikes(raw_features, spikes, config.asset_dir / "01b_capacity_spikes_removed.png")

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

    metrics, pcc_all, predictions, history = run_experiment_suite(
        scenarios=scenarios,
        feature_cols=feature_cols,
        config=config,
        label="main",
    )
    case_mean = aggregate_case_metrics(metrics)

    metrics_path = config.output_dir / f"05_metrics_summary_{len(metrics)}runs.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(config.output_dir / "05_metrics_summary_all_runs.csv", index=False, encoding="utf-8-sig")
    case_mean.to_csv(config.output_dir / "06_metrics_by_case_mean.csv", index=False, encoding="utf-8-sig")
    pcc_all.to_csv(config.output_dir / "07_pcc_all_scenarios.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.output_dir / "09_predictions_all_scenarios.csv", index=False, encoding="utf-8-sig")
    history.to_csv(config.output_dir / "10_loss_history_all_scenarios.csv", index=False, encoding="utf-8-sig")

    raw_scenarios = build_scenarios(raw_features, seed=config.seed)
    raw_feature_cols = model_input_columns(raw_features)
    raw_metrics, _, _, _ = run_experiment_suite(
        scenarios=raw_scenarios,
        feature_cols=raw_feature_cols,
        config=config,
        label="ablation_no_spike_removal",
    )
    cleaning_ablation = compare_metrics(
        baseline=metrics,
        ablation=raw_metrics,
        baseline_name="remove_spikes",
        ablation_name="keep_spikes",
    )
    cleaning_ablation.to_csv(config.output_dir / "11_ablation_capacity_spike_cleaning.csv", index=False, encoding="utf-8-sig")

    no_cycle_cols = [col for col in feature_cols if col not in {"cycle_index", "global_cycle_index"}]
    c_scenarios = [scenario for scenario in scenarios if scenario.case == "C"]
    no_cycle_metrics, _, _, _ = run_experiment_suite(
        scenarios=c_scenarios,
        feature_cols=no_cycle_cols,
        config=config,
        label="ablation_C_no_cycle_index",
    )
    cycle_ablation = compare_metrics(
        baseline=metrics[metrics["case"] == "C"],
        ablation=no_cycle_metrics,
        baseline_name="with_cycle_index",
        ablation_name="without_cycle_index",
    )
    cycle_ablation.to_csv(config.output_dir / "12_ablation_transfer_no_cycle_index.csv", index=False, encoding="utf-8-sig")

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
    plot_metrics(metrics, config.asset_dir / "05_metrics_comparison_all_runs.png")
    plot_group_predictions(predictions, metrics, "A", config.asset_dir / "06_predictions_A_four_batteries.png")
    plot_group_predictions(predictions, metrics, "B", config.asset_dir / "07_predictions_B_four_batteries.png")
    plot_group_predictions(predictions, metrics, "C", config.asset_dir / "08_predictions_C_all_transfers.png")

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

    print(f"\n=== {len(metrics)} runs ===")
    print(metrics[["case", "source_battery", "target_battery", "n_train", "n_val", "n_test", "MAE", "RMSE", "R2"]].to_string(index=False))
    print("\n=== Mean by case ===")
    print(case_mean.to_string(index=False))
    print("\n=== Ablation: capacity spike cleaning ===")
    print(cleaning_ablation[["scenario", "MAE_remove_spikes", "MAE_keep_spikes", "MAE_delta_keep_minus_remove"]].to_string(index=False))
    print("\n=== Ablation: C without cycle index ===")
    print(cycle_ablation[["scenario", "MAE_with_cycle_index", "MAE_without_cycle_index", "MAE_delta_without_minus_with"]].to_string(index=False))
    print(f"\nOutputs: {config.output_dir.resolve()}")
    print(f"Presentation summary: {summary_path.resolve()}")


def run_experiment_suite(
    scenarios,
    feature_cols: list[str],
    config: PipelineConfig,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    pcc_tables = []
    prediction_tables = []
    history_tables = []
    for scenario in scenarios:
        print(f"Running [{label}] {scenario.name} ...", flush=True)
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
            f"Done [{label}] {scenario.name}: MAE={result.metrics['MAE']:.4f}, "
            f"RMSE={result.metrics['RMSE']:.4f}, R2={result.metrics['R2']:.4f}",
            flush=True,
        )

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(pcc_tables, ignore_index=True),
        pd.concat(prediction_tables, ignore_index=True),
        pd.concat(history_tables, ignore_index=True),
    )


def compare_metrics(
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    baseline_name: str,
    ablation_name: str,
) -> pd.DataFrame:
    cols = ["scenario", "case", "source_battery", "target_battery", "n_train", "n_val", "n_test", "MAE", "RMSE", "R2"]
    merged = baseline[cols].merge(
        ablation[cols],
        on=["scenario", "case", "source_battery", "target_battery"],
        suffixes=(f"_{baseline_name}", f"_{ablation_name}"),
    )
    merged[f"MAE_delta_{ablation_name.replace('keep_spikes', 'keep_minus_remove').replace('without_cycle_index', 'without_minus_with')}"] = (
        merged[f"MAE_{ablation_name}"] - merged[f"MAE_{baseline_name}"]
    )
    merged[f"RMSE_delta_{ablation_name.replace('keep_spikes', 'keep_minus_remove').replace('without_cycle_index', 'without_minus_with')}"] = (
        merged[f"RMSE_{ablation_name}"] - merged[f"RMSE_{baseline_name}"]
    )
    merged[f"R2_delta_{ablation_name.replace('keep_spikes', 'keep_minus_remove').replace('without_cycle_index', 'without_minus_with')}"] = (
        merged[f"R2_{ablation_name}"] - merged[f"R2_{baseline_name}"]
    )
    return merged


if __name__ == "__main__":
    main()
