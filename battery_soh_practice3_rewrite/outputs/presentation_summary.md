# 实践三答辩摘要：锂离子电池 SOH 预测

## 1. 数据预处理
- 使用 NASA Battery Aging 数据集中 B0005、B0006、B0007、B0018 四个电池。
- 清洗后放电 cycle 总数：629。
- 异常处理包括解析失败、长度不一致、非有限值、容量范围异常、时间轴异常和局部容量尖峰过滤。
- 标签定义：SOH(%) = 当前放电容量 / 2.0Ah * 100。
- 为避免标签泄露，capacity_ah、charge_ah_integral、energy_wh 不作为模型输入。

## 2. 三种划分方式
- A：单电池随机 60%/20%/20% 划分。
- B：单电池按 cycle 顺序，前 60% 用于训练/验证，后 40% 测试。
- C：遍历全部源-目标电池有向组合；一个源电池 + 目标电池前 10% 训练，目标电池后 90% 测试。

## 3. 特征选择
- 使用训练集上的 Pearson 相关系数排序，选择绝对相关性最高的 Top K 特征。
- cycle_index 的 PCC 排名较高，但跨电池迁移中可能编码源电池退化节奏，因此通过 C 类去序号消融单独讨论。
- 代表场景 Top K 特征：
  - 1. time_to_3p6v_s，PCC=0.9986：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 2. time_to_3p5v_s，PCC=0.9984：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 3. time_to_3p7v_s，PCC=0.9973：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 4. early_voltage_slope，PCC=0.9953：电压平台、端电压和电压斜率描述放电曲线形状，老化会改变这些曲线特征。
  - 5. global_cycle_index，PCC=-0.9886：循环序号刻画退化进程，适合单电池趋势建模，但跨电池泛化时需谨慎解释。
  - 6. cycle_index，PCC=-0.9869：循环序号刻画退化进程，适合单电池趋势建模，但跨电池泛化时需谨慎解释。
  - 7. time_to_3p8v_s，PCC=0.9846：放电时长和到达电压阈值的时间会随容量衰退缩短，能反映可用容量变化。
  - 8. voltage_mean，PCC=0.9801：电压平台、端电压和电压斜率描述放电曲线形状，老化会改变这些曲线特征。

## 4. 模型设计
- 模型：MLP 回归网络，输入为 Top K 特征，隐藏层为 64 和 32，输出 1 个 SOH 百分比。
- 选择理由：特征为每个 cycle 的结构化统计量，MLP 参数少、训练稳定、适合小样本回归。
- 训练策略：AdamW + SmoothL1Loss + 验证集早停。

## 5. 结果概览
| case | MAE_mean | RMSE_mean | R2_mean |
| --- | --- | --- | --- |
| A | 0.2217 | 0.2949 | 0.9987 |
| B | 0.7871 | 0.8880 | 0.8728 |
| C | 3.2211 | 3.4955 | 0.7638 |

- 最好场景：A_random_B0006_60_20_20，MAE=0.2060，RMSE=0.2334，R2=0.9997。
- 最难场景：C_transfer_B0018_to_B0006_target10，MAE=6.7627，RMSE=6.9490，R2=0.6011。
- 通常 A 随机划分结果最好，B 更接近未来 cycle 预测，C 体现跨电池迁移泛化难度。

## 6. PPT 插图清单
- report_assets/01_capacity_degradation.png：容量衰退曲线。
- report_assets/01b_capacity_spikes_removed.png：被剔除容量尖峰位置。
- report_assets/02_split_schematic.png：三种划分方式示意图。
- report_assets/03_pcc_heatmap_topK.png：PCC 热力图。
- report_assets/04_mlp_structure.png：模型结构图。
- report_assets/05_metrics_comparison_all_runs.png：20 次实验指标对比。
- report_assets/08_predictions_C_all_transfers.png：全部迁移组合预测曲线。
- report_assets/09_prediction_true_vs_pred_B0005_B.png：代表预测曲线。
- report_assets/10_loss_curve_B0005_B.png：代表 Loss 曲线。
- 11_ablation_capacity_spike_cleaning.csv：容量尖峰剔除消融。
- 12_ablation_transfer_no_cycle_index.csv：C 类去除 cycle 序号特征消融。

## 7. 答辩时可强调的问题与解决
- 问题：NASA 原始 mat 文件层级复杂。解决：只抽取 discharge cycle，并统一清洗时间、电压、电流和温度序列。
- 问题：容量相关积分特征容易形成标签泄露。解决：保留到数据表用于分析，但从模型输入中排除。
- 问题：随机划分指标很高但不代表真实未来预测。解决：同时设计 B 时序外推和 C 跨电池迁移实验。
- 问题：容量尖峰剔除和 cycle 序号特征可能被质疑。解决：补充清洗消融和 C 类去序号特征消融，透明展示策略影响。
- 消融结论：容量尖峰剔除主要改善 B 类时序外推；cycle 序号在 C 类平均有帮助，但部分迁移对去除后更好，存在场景依赖性。
