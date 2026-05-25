# 实践三：基于 PyTorch 的锂离子电池 SOH 预测

本代码严格对应实验报告要求：

1. 数据预处理：原始数据描述、异常处理、A/B/C 三种划分方式示意图。
2. 特征选择：PCC 热力图、前 K 个特征及理由。
3. 模型设计：MLP 网络结构图、层数、输入输出维度、选择理由。
4. 训练与测试结果：A/B/C 的 MAE、RMSE、R² 对比表；一种方式的预测值 vs 真实值曲线图；Loss 曲线。
5. 12 次独立实验：A/B/C 均分别对 B0005、B0006、B0007、B0018 单独训练和测试，不混合测试。

## 运行方法

```bash
pip install -r requirements.txt
python run_all.py --data_zip ./data/BatteryAgingARC.zip --output_dir ./outputs --save_individual_plots
```

## 主要输出

- `outputs/01_features_all.csv`：清洗后的放电 cycle 特征表。
- `outputs/02_data_description.csv`：原始/清洗后数据描述。
- `outputs/03_cleaning_summary.csv`：异常处理统计。
- `outputs/04_split_summary.csv`：A/B/C 三种划分的训练、验证、测试数量。
- `outputs/05_metrics_summary_12runs.csv`：12 次独立实验 MAE/RMSE/R²。
- `outputs/06_metrics_by_case_mean.csv`：A/B/C 三类划分平均指标。
- `outputs/07_pcc_all_scenarios.csv`：每个实验场景的 PCC 排序。
- `outputs/08_topK_features_and_reasons.csv`：报告可用的前 K 个特征及理由。
- `outputs/09_predictions_all_scenarios.csv`：全部测试集预测结果。
- `outputs/10_loss_history_all_scenarios.csv`：全部训练过程 loss。
- `outputs/report_assets/`：报告需要插入的图片。
