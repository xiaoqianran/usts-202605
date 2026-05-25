#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features import (
    extract_zip_if_needed,
    build_feature_table,
    make_data_description,
    model_feature_columns,
    feature_reason,
)
from src.splits import make_all_scenarios
from src.train_eval import train_and_evaluate
from src.plotting import (
    plot_capacity_degradation,
    plot_split_schematic,
    plot_pcc_heatmap,
    plot_mlp_structure,
    plot_metrics_comparison,
    plot_prediction_curve,
    plot_loss_curve,
    plot_group_predictions,
)


def save_split_summary(scenarios: dict, out_path: Path) -> pd.DataFrame:
    rows = []
    for scenario, (case, source, target, train, val, test) in scenarios.items():
        for split_name, split_df in [("train", train), ("validation", val), ("test", test)]:
            rows.append({
                "scenario": scenario,
                "case": case,
                "source_battery": source,
                "target_battery": target,
                "split": split_name,
                "n": len(split_df),
                "batteries_in_split": ",".join(sorted(split_df["battery_id"].unique())) if len(split_df) else "",
                "cycle_min": int(split_df["cycle_index"].min()) if len(split_df) else None,
                "cycle_max": int(split_df["cycle_index"].max()) if len(split_df) else None,
                "soh_min": float(split_df["soh_percent"].min()) if len(split_df) else None,
                "soh_max": float(split_df["soh_percent"].max()) if len(split_df) else None,
            })
    split_summary = pd.DataFrame(rows)
    split_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    return split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice 3: PyTorch SOH prediction on NASA battery aging data")
    parser.add_argument("--data_zip", type=str, default="./data/BatteryAgingARC.zip", help="Path to BatteryAgingARC.zip")
    parser.add_argument("--work_dir", type=str, default="./data/extracted", help="Extraction directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--top_k", type=int, default=8, help="Number of PCC-selected features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--save_individual_plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    asset_dir = output_dir / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据读取、清洗、描述
    data_dir = extract_zip_if_needed(args.data_zip, args.work_dir)
    features, cleaning_summary = build_feature_table(str(data_dir))
    data_description = make_data_description(features, cleaning_summary)
    features.to_csv(output_dir / "01_features_all.csv", index=False, encoding="utf-8-sig")
    data_description.to_csv(output_dir / "02_data_description.csv", index=False, encoding="utf-8-sig")
    cleaning_summary.to_csv(output_dir / "03_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    plot_capacity_degradation(features, asset_dir / "01_capacity_degradation.png")

    # 2. A/B/C 三种划分；每种对四个电池单独运行
    scenarios = make_all_scenarios(features, seed=args.seed)
    split_summary = save_split_summary(scenarios, output_dir / "04_split_summary.csv")
    plot_split_schematic(asset_dir / "02_split_schematic.png")

    feature_cols = model_feature_columns(features)

    metric_rows = []
    pcc_tables = []
    prediction_tables = []
    history_tables = []

    for scenario, (case, source, target, train_df, val_df, test_df) in scenarios.items():
        print(f"Running {scenario} ...", flush=True)
        result = train_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            scenario=scenario,
            top_k=args.top_k,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
        )

        metrics = result.metrics.copy()
        metrics["case"] = case
        metrics["source_battery"] = source
        metrics["target_battery"] = target
        metrics["selected_features"] = "; ".join(result.selected_features)
        metric_rows.append(metrics)

        pcc = result.pcc_table.copy()
        pcc["scenario"] = scenario
        pcc["case"] = case
        pcc["source_battery"] = source
        pcc["target_battery"] = target
        pcc_tables.append(pcc)

        pred = result.predictions.copy()
        pred["case"] = case
        pred["source_battery"] = source
        pred["target_battery"] = target
        prediction_tables.append(pred)

        hist = result.history.copy()
        hist["case"] = case
        hist["source_battery"] = source
        hist["target_battery"] = target
        history_tables.append(hist)

        print(f"Done {scenario}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}", flush=True)

    metrics_df = pd.DataFrame(metric_rows)
    front_cols = ["case", "source_battery", "target_battery", "scenario"]
    metrics_df = metrics_df[front_cols + [c for c in metrics_df.columns if c not in front_cols]]
    metrics_df.to_csv(output_dir / "05_metrics_summary_12runs.csv", index=False, encoding="utf-8-sig")

    case_mean = metrics_df.groupby("case", as_index=False).agg(
        MAE_mean=("MAE", "mean"),
        RMSE_mean=("RMSE", "mean"),
        R2_mean=("R2", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_std=("RMSE", "std"),
        R2_std=("R2", "std"),
    )
    case_mean.to_csv(output_dir / "06_metrics_by_case_mean.csv", index=False, encoding="utf-8-sig")

    pcc_all = pd.concat(pcc_tables, ignore_index=True)
    pred_all = pd.concat(prediction_tables, ignore_index=True)
    hist_all = pd.concat(history_tables, ignore_index=True)
    pcc_all.to_csv(output_dir / "07_pcc_all_scenarios.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(output_dir / "09_predictions_all_scenarios.csv", index=False, encoding="utf-8-sig")
    hist_all.to_csv(output_dir / "10_loss_history_all_scenarios.csv", index=False, encoding="utf-8-sig")

    # 3. 报告图：PCC 热力图、前 K 特征及理由、模型结构图
    representative_scenario = "A_random_B0005_60_20_20"
    rep_case, rep_source, rep_target, rep_train, rep_val, rep_test = scenarios[representative_scenario]
    rep_pcc = pcc_all[pcc_all["scenario"] == representative_scenario].sort_values("rank")
    top_features = plot_pcc_heatmap(rep_train, rep_pcc, args.top_k, asset_dir / "03_pcc_heatmap_topK.png")
    top_feature_rows = []
    for feature in top_features:
        row = rep_pcc[rep_pcc["feature"] == feature].iloc[0]
        top_feature_rows.append({
            "rank": int(row["rank"]),
            "feature": feature,
            "pcc": float(row["pcc"]),
            "abs_pcc": float(row["abs_pcc"]),
            "reason": feature_reason(feature),
        })
    pd.DataFrame(top_feature_rows).to_csv(output_dir / "08_topK_features_and_reasons.csv", index=False, encoding="utf-8-sig")
    plot_mlp_structure(args.top_k, asset_dir / "04_mlp_structure.png")

    # 4. 报告图：指标对比、预测曲线、loss 曲线
    plot_metrics_comparison(metrics_df, asset_dir / "05_metrics_comparison_12runs.png")
    plot_group_predictions(pred_all, metrics_df, "A", asset_dir / "06_predictions_A_four_batteries.png")
    plot_group_predictions(pred_all, metrics_df, "B", asset_dir / "07_predictions_B_four_batteries.png")
    plot_group_predictions(pred_all, metrics_df, "C", asset_dir / "08_predictions_C_four_batteries.png")

    prediction_scenario = "B_chrono_B0005_first60_last40"
    plot_prediction_curve(pred_all, prediction_scenario, asset_dir / "09_prediction_true_vs_pred_B0005_B.png")
    plot_loss_curve(hist_all, prediction_scenario, asset_dir / "10_loss_curve_B0005_B.png")

    if args.save_individual_plots:
        individual_dir = asset_dir / "individual_runs"
        individual_dir.mkdir(exist_ok=True)
        for scenario in metrics_df["scenario"]:
            plot_prediction_curve(pred_all, scenario, individual_dir / f"prediction_{scenario}.png")
            plot_loss_curve(hist_all, scenario, individual_dir / f"loss_{scenario}.png")

    print("\n=== 12 independent runs: MAE/RMSE/R² ===")
    print(metrics_df[["case", "source_battery", "target_battery", "n_train", "n_val", "n_test", "MAE", "RMSE", "R2"]].to_string(index=False))
    print("\n=== Mean metrics by split case ===")
    print(case_mean.to_string(index=False))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")
    print(f"Report figures saved to: {asset_dir.resolve()}")


if __name__ == "__main__":
    main()
