# 实践三：基于 PyTorch 的锂离子电池 SOH 预测

本项目完成“深度学习应用实践”任务，使用 NASA Battery Aging ARC-FY08Q4 数据集，通过放电曲线健康特征与 PyTorch MLP 回归模型预测锂离子电池 SOH。

## 1. 运行环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 数据准备与运行

项目压缩包中已包含解压后的 `B0005.mat`、`B0006.mat`、`B0007.mat`、`B0018.mat`。如果重新下载数据，也可以将 `BatteryAgingARC.zip` 放到 `data/` 目录下，或使用 `--data_zip` 指定路径。

```bash
python run_all.py --output_dir ./outputs
```

如果希望额外保存每个 scenario 的单独 PCC 图和预测图，可以运行：

```bash
python run_all.py --output_dir ./outputs --save_individual_plots
```

## 3. 实验内容

脚本自动完成：

1. 读取并解析 B0005、B0006、B0007、B0018 四个 `.mat` 文件；
2. 仅保留 discharge cycle，提取 SOH 标签：`SOH = Capacity / 2.0Ah * 100%`；
3. 构造放电时间、到达固定电压阈值时间、温度变化、曲线均值/斜率等特征；
4. 使用训练集 PCC 进行特征选择；
5. 使用 PyTorch MLP 回归模型训练；
6. 严格按照修正后的要求完成 12 次独立实验：
   - A：B0005、B0006、B0007、B0018 分别随机 60% / 20% / 20%，共 4 次；
   - B：B0005、B0006、B0007、B0018 分别按前 60% cycle 训练、后 40% cycle 测试，共 4 次；
   - C：四个目标电池分别独立测试，每次只使用一个源电池，不合并多个源电池，共 4 次：B0007→B0005、B0007→B0006、B0005→B0007、B0005→B0018；
7. 输出 MAE、RMSE、R2、PCC 结果和预测曲线。

## 4. 输出文件

运行后 `outputs/` 中包含：

- `features_all.csv`：预处理后的 cycle-level 特征表；
- `metrics_summary.csv`：12 次独立实验的 MAE、RMSE、R2；
- `split_summary.csv`：12 次独立实验的训练/验证/测试集样本数；
- `pcc_all_scenarios.csv`：各场景 PCC 排序；
- `predictions_all_scenarios.csv`：测试集真实值与预测值；
- `capacity_degradation.png`：四个电池 SOH 衰减曲线；
- `metrics_12_runs.png`：12 次实验指标对比图；
- `predictions_A_four_batteries.png`、`predictions_B_four_batteries.png`、`predictions_C_four_batteries.png`：A/B/C 三种设置下四个电池的预测曲线。

## 5. 说明

为了避免标签泄露，模型不会把 `capacity_ah` 作为输入。`charge_ah_integral` 与 Capacity 高度等价，`energy_wh` 也与完整放电容量强相关，因此二者默认均从模型输入中排除。模型主要使用到达电压阈值时间、温度变化、曲线均值/斜率等可解释健康指标进行 SOH 回归。
