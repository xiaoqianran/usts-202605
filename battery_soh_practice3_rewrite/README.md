# 实践三：锂离子电池 SOH 预测重写版

这是面向“实验报告 + 答辩展示”重新整理的实践三项目。代码按职责拆分，输出文件直接对应任务书和答辩说明中的要求。

## 对应任务书要求

- 数据预处理：读取 NASA 电池数据、抽取 discharge cycle、过滤异常样本，输出数据描述和清洗统计。
- 三种划分：A 单电池随机 60/20/20；B 单电池前 60% cycle 与后 40% cycle；C 源电池 + 目标电池前 10% 训练，目标后 90% 测试。
- 特征选择：在训练集计算 Pearson Correlation Coefficient，输出 Top K 特征、PCC 热力图和选择理由。
- 模型选择：使用小型 MLP 回归网络，给出网络结构图和训练策略。
- 训练测试：完成 12 次独立实验，输出 MAE、RMSE、R2、预测曲线和 Loss 曲线。

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
- `outputs/04_split_summary.csv`：三种划分方式下训练、验证、测试集数量。
- `outputs/05_metrics_summary_12runs.csv`：12 次独立实验的 MAE/RMSE/R2。
- `outputs/06_metrics_by_case_mean.csv`：A/B/C 三类实验平均指标。
- `outputs/07_pcc_all_scenarios.csv`：每个实验场景的 PCC 排序。
- `outputs/08_topK_features_and_reasons.csv`：答辩可用的 Top K 特征和解释。
- `outputs/09_predictions_all_scenarios.csv`：全部测试集真实值、预测值和误差。
- `outputs/10_loss_history_all_scenarios.csv`：全部训练过程 Loss。
- `outputs/presentation_summary.md`：可直接整理进 PPT 的答辩摘要。
- `outputs/report_assets/`：答辩说明要求的图片素材。

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

