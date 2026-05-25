# 实践三答辩摘要：锂离子电池 SOH 预测

## 1. 数据预处理
- 使用 NASA Battery Aging 数据集中 B0005、B0006、B0007、B0018 四个电池。
- 清洗后放电 cycle 总数：636。
- 异常处理包括解析失败、长度不一致、非有限值、容量异常、时间轴异常过滤。
- 标签定义：SOH(%) = 当前放电容量 / 2.0Ah * 100。
- 为避免标签泄露，capacity_ah、charge_ah_integral、energy_wh 不作为模型输入。

## 2. 三种划分方式
- A：单电池随机 60%/20%/20% 划分。
- B：单电池按 cycle 顺序，前 60% 用于训练/验证，后 40% 测试。
- C：一个源电池 + 目标电池前 10% 训练，目标电池后 90% 测试。

## 3. 特征选择
- 使用训练集上的 Pearson 相关系数排序，选择绝对相关性最高的 Top K 特征。
- 代表场景 Top K 特征：
  - 1. time_to_3p6v_s，PCC=0.9984：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 2. time_to_3p5v_s，PCC=0.9982：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 3. time_to_3p7v_s，PCC=0.9970：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 4. early_voltage_slope，PCC=0.9952：电压平台、端电压和电压斜率描述放电曲线形状，老化会改变这些曲线特征。
  - 5. global_cycle_index，PCC=-0.9878：循环序号刻画退化进程，适合单电池趋势建模，但跨电池泛化时需谨慎解释。
  - 6. cycle_index，PCC=-0.9860：循环序号刻画退化进程，适合单电池趋势建模，但跨电池泛化时需谨慎解释。
  - 7. time_to_3p8v_s，PCC=0.9830：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 8. voltage_mean，PCC=0.9795：电压平台、端电压和电压斜率描述放电曲线形状，老化会改变这些曲线特征。

## 4. 模型设计
- 模型：MLP 回归网络，输入为 Top K 特征，隐藏层为 64 和 32，输出 1 个 SOH 百分比。
- 选择理由：特征为每个 cycle 的结构化统计量，MLP 参数少、训练稳定、适合小样本回归。
- 训练策略：AdamW + SmoothL1Loss + 验证集早停。

## 5. 结果概览
| case | MAE_mean | RMSE_mean | R2_mean |
| --- | --- | --- | --- |
| A | 0.1832 | 0.2417 | 0.9991 |
| B | 0.9922 | 1.1776 | 0.8076 |
| C | 2.1567 | 2.5554 | 0.9025 |

- 最好场景：A_random_B0005_60_20_20，MAE=0.1552，RMSE=0.1909，R2=0.9995。
- 最难场景：C_transfer_B0007_to_B0006_target10，MAE=4.1562，RMSE=4.6267，R2=0.8244。
- 通常 A 随机划分结果最好，B 更接近未来 cycle 预测，C 体现跨电池迁移泛化难度。

## 6. PPT 插图清单
- report_assets/01_capacity_degradation.png：容量衰退曲线。
- report_assets/02_split_schematic.png：三种划分方式示意图。
- report_assets/03_pcc_heatmap_topK.png：PCC 热力图。
- report_assets/04_mlp_structure.png：模型结构图。
- report_assets/05_metrics_comparison_12runs.png：12 次实验指标对比。
- report_assets/09_prediction_true_vs_pred_B0005_B.png：代表预测曲线。
- report_assets/10_loss_curve_B0005_B.png：代表 Loss 曲线。

## 7. 答辩时可强调的问题与解决
- 问题：NASA 原始 mat 文件层级复杂。解决：只抽取 discharge cycle，并统一清洗时间、电压、电流和温度序列。
- 问题：容量相关积分特征容易形成标签泄露。解决：保留到数据表用于分析，但从模型输入中排除。
- 问题：随机划分指标很高但不代表真实未来预测。解决：同时设计 B 时序外推和 C 跨电池迁移实验。