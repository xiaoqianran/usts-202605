# 实践三：锂离子电池 SOH 预测重写版

这是面向“实验报告 + 答辩展示”重新整理的实践三项目。代码按职责拆分，输出文件直接对应任务书和答辩说明中的要求。

## 对应任务书要求

- 数据预处理：读取 NASA 电池数据、抽取 discharge cycle、过滤异常样本，输出数据描述和清洗统计。
- 三种划分：A 单电池随机 60/20/20；B 单电池前 60% cycle 与后 40% cycle；C 遍历全部源-目标电池有向组合，源电池 + 目标电池前 10% 训练，目标后 90% 测试。
- 特征选择：在训练集计算 Pearson Correlation Coefficient，输出 Top K 特征、PCC 热力图和选择理由。
- 模型选择：使用小型 MLP 回归网络，给出网络结构图和训练策略。
- 训练测试：完成 20 次独立实验，输出 MAE、RMSE、R2、预测曲线和 Loss 曲线。
- 补充消融：输出容量尖峰剔除前后对比、C 类迁移去除 `cycle_index/global_cycle_index` 的对比，用于说明清洗和序号特征的影响。

## 运行

```bash
cd battery_soh_practice3_rewrite
pip install -r requirements.txt
python run_pipeline.py --output_dir outputs --save_individual_plots
```

如需指定数据位置：

```bash
python run_pipeline.py --data_zip data/BatteryAgingARC.zip --data_dir data/extracted
```

## 主要输出

- `outputs/01_features_all.csv`：清洗后的 cycle 特征表。
- `outputs/02_data_description.csv`：每个电池的原始/清洗后数据描述。
- `outputs/03_cleaning_summary.csv`：异常处理统计。
- `outputs/03b_capacity_spike_report.csv`：被剔除容量尖峰的 cycle 位置、局部中位数、偏离量和阈值。
- `outputs/04_split_summary.csv`：三种划分方式下训练、验证、测试集数量。
- `outputs/05_metrics_summary_all_runs.csv`：全部独立实验的 MAE/RMSE/R2。
- `outputs/05_metrics_summary_20runs.csv`：20 次独立实验的 MAE/RMSE/R2。
- `outputs/06_metrics_by_case_mean.csv`：A/B/C 三类实验平均指标。
- `outputs/07_pcc_all_scenarios.csv`：每个实验场景的 PCC 排序。
- `outputs/08_topK_features_and_reasons.csv`：答辩可用的 Top K 特征和解释。
- `outputs/09_predictions_all_scenarios.csv`：全部测试集真实值、预测值和误差。
- `outputs/10_loss_history_all_scenarios.csv`：全部训练过程 Loss。
- `outputs/11_ablation_capacity_spike_cleaning.csv`：容量尖峰剔除 vs 保留的指标对比。
- `outputs/12_ablation_transfer_no_cycle_index.csv`：C 类迁移中保留 vs 去除 `cycle_index/global_cycle_index` 的指标对比。
- `outputs/presentation_summary.md`：可直接整理进 PPT 的答辩摘要。
- `outputs/report_assets/`：答辩说明要求的图片素材。

## 补充消融结论

容量尖峰检测使用局部窗口半径 4 的邻域中位数和 MAD。判定规则为：当前容量高于邻域中位数，且偏离量超过 `max(0.04Ah, 3 * 1.4826 * MAD)`。本次共标记 7 个局部回升点，位置和阈值见 `outputs/03b_capacity_spike_report.csv`，曲线图见 `outputs/report_assets/01b_capacity_spikes_removed.png`。

剔除容量尖峰不是为了单向提升指标。消融结果显示，保留尖峰时 A 类平均 MAE 从 0.2217 变为 0.1832，B 类从 0.7871 变为 0.9922，C 类从 3.2211 变为 3.1667。也就是说，清洗影响具有场景依赖性：随机划分和跨电池平均结果对这些点不敏感，甚至保留时略低；但在时序外推 B 类中，保留尖峰会明显损害模型对未来衰退趋势的预测。因此该步骤应作为标签质量控制与敏感性分析呈现，而不是简单解释为提升测试指标的技巧。

C 类迁移不是零样本迁移，而是“源电池全量 + 目标电池前 10% 早期数据”的少样本校准设置。去除 `cycle_index/global_cycle_index` 后，C 类平均 MAE 从 3.2211 上升到 3.9387，平均 R2 从 0.7638 降到 0.6849，说明序号特征对部分迁移组合有帮助，但也可能编码源电池特定退化节奏。分组合看，`B0005 -> B0007` 的 MAE 从 2.0611 降到 1.3683，`B0007 -> B0005` 从 1.0493 降到 0.5399，说明序号特征在某些迁移对上也可能带来负迁移风险。因此报告中应同时展示该消融，并说明跨电池泛化仍然是主要难点。

## 项目结构

```text
battery_soh_practice3_rewrite/
  run_pipeline.py
  src/battery_soh/
    data.py          # 数据定位、读取、清洗、特征提取
    splits.py        # A/B/C 三种实验划分
    experiment.py    # PCC、训练、评估
    model.py         # MLP SOH 回归模型
    plots.py         # 报告和 PPT 图表
    presentation.py  # 自动生成答辩摘要
  docs/
    答辩展示提纲.md
    实验报告素材说明.md
```
