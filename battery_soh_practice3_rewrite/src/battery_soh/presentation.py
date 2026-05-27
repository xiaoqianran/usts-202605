from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_presentation_summary(
    output_dir: Path,
    data_desc: pd.DataFrame,
    cleaning: pd.DataFrame,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    case_mean: pd.DataFrame,
    top_features: pd.DataFrame,
) -> Path:
    path = output_dir / "presentation_summary.md"
    best = metrics.sort_values("MAE").iloc[0]
    hardest = metrics.sort_values("MAE", ascending=False).iloc[0]

    lines = [
        "# 实践三答辩摘要：锂离子电池 SOH 预测",
        "",
        "## 1. 数据预处理",
        f"- 使用 NASA Battery Aging 数据集中 B0005、B0006、B0007、B0018 四个电池。",
        f"- 清洗后放电 cycle 总数：{int(data_desc['kept_discharge_cycles'].sum())}。",
        "- 异常处理包括解析失败、长度不一致、非有限值、容量范围异常、时间轴异常和局部容量尖峰过滤。",
        "- 标签定义：SOH(%) = 当前放电容量 / 2.0Ah * 100。",
        "- 为避免标签泄露，capacity_ah、charge_ah_integral、energy_wh 不作为模型输入。",
        "",
        "## 2. 三种划分方式",
        "- A：单电池随机 60%/20%/20% 划分。",
        "- B：单电池按 cycle 顺序，前 60% 用于训练/验证，后 40% 测试。",
        "- C：遍历全部源-目标电池有向组合；一个源电池 + 目标电池前 10% 训练，目标电池后 90% 测试。",
        "",
        "## 3. 特征选择",
        "- 使用训练集上的 Pearson 相关系数排序，选择绝对相关性最高的 Top K 特征。",
        "- 代表场景 Top K 特征：",
    ]
    for _, row in top_features.iterrows():
        lines.append(f"  - {int(row['rank'])}. {row['feature']}，PCC={row['pcc']:.4f}：{row['reason']}")

    lines.extend(
        [
            "",
            "## 4. 模型设计",
            "- 模型：MLP 回归网络，输入为 Top K 特征，隐藏层为 64 和 32，输出 1 个 SOH 百分比。",
            "- 选择理由：特征为每个 cycle 的结构化统计量，MLP 参数少、训练稳定、适合小样本回归。",
            "- 训练策略：AdamW + SmoothL1Loss + 验证集早停。",
            "",
            "## 5. 结果概览",
            _markdown_table(case_mean, ["case", "MAE_mean", "RMSE_mean", "R2_mean"]),
            "",
            f"- 最好场景：{best['scenario']}，MAE={best['MAE']:.4f}，RMSE={best['RMSE']:.4f}，R2={best['R2']:.4f}。",
            f"- 最难场景：{hardest['scenario']}，MAE={hardest['MAE']:.4f}，RMSE={hardest['RMSE']:.4f}，R2={hardest['R2']:.4f}。",
            "- 通常 A 随机划分结果最好，B 更接近未来 cycle 预测，C 体现跨电池迁移泛化难度。",
            "",
            "## 6. PPT 插图清单",
            "- report_assets/01_capacity_degradation.png：容量衰退曲线。",
            "- report_assets/01b_capacity_spikes_removed.png：被剔除容量尖峰位置。",
            "- report_assets/02_split_schematic.png：三种划分方式示意图。",
            "- report_assets/03_pcc_heatmap_topK.png：PCC 热力图。",
            "- report_assets/04_mlp_structure.png：模型结构图。",
            f"- report_assets/05_metrics_comparison_all_runs.png：{len(metrics)} 次实验指标对比。",
            "- report_assets/08_predictions_C_all_transfers.png：全部迁移组合预测曲线。",
            "- report_assets/09_prediction_true_vs_pred_B0005_B.png：代表预测曲线。",
            "- report_assets/10_loss_curve_B0005_B.png：代表 Loss 曲线。",
            "- 11_ablation_capacity_spike_cleaning.csv：容量尖峰剔除消融。",
            "- 12_ablation_transfer_no_cycle_index.csv：C 类去除 cycle 序号特征消融。",
            "",
            "## 7. 答辩时可强调的问题与解决",
            "- 问题：NASA 原始 mat 文件层级复杂。解决：只抽取 discharge cycle，并统一清洗时间、电压、电流和温度序列。",
            "- 问题：容量相关积分特征容易形成标签泄露。解决：保留到数据表用于分析，但从模型输入中排除。",
            "- 问题：随机划分指标很高但不代表真实未来预测。解决：同时设计 B 时序外推和 C 跨电池迁移实验。",
            "- 问题：容量尖峰剔除和 cycle 序号特征可能被质疑。解决：补充清洗消融和 C 类去序号特征消融，透明展示策略影响。",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    sub = df[columns].copy()
    for col in sub.columns:
        if col != "case":
            sub[col] = sub[col].map(lambda value: f"{float(value):.4f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in sub.to_numpy()]
    return "\n".join([header, sep, *rows])
