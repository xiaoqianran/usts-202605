# USTS AI Training Experiments - 2026.05
# 苏州科技大学人工智能实训实验仓库 - 2026.05

苏州科技大学（USTS）《人工智能开发实训 II》2026.05 课程实验与实践成果仓库。  
Repository for Artificial Intelligence Development Practical Training II at Suzhou University of Science and Technology (USTS), May 2026.

包含官方实践一、二、三的完整实现、工具链、大量实验结果与答辩素材。  
Contains complete implementations, toolchains, extensive experimental results, and defense presentation materials for the official Practices 1–3.

---

## 具体实验内容 / Specific Experiments

### 实践一：PyTorch 深度学习实践 / Practice 1: PyTorch Deep Learning Basics

**中文**  
熟悉 PyTorch 框架与训练流程，完成基础深度学习分类/回归实验及环境测试。提交带 loss 曲线、测试结果与完整代码的实验报告。

**English**  
Familiarization with the PyTorch framework and training pipeline. Complete basic deep learning classification/regression experiments and environment validation. Submit reports including loss curves, test results, and full source code.

---

### 实践二：深度半监督学习实践（MSTAR SAR 图像分类） / Practice 2: Deep Semi-Supervised Learning (MSTAR SAR Classification)

**中文**  
**对应任务书要求**：研究标签有限对性能的影响，并使用半监督方法恢复精度。

**本仓库完整实现**（v1.0.0 重点交付）：
- 三组核心对照实验：
  - 全标签监督学习（100% 标签，上限参考）
  - 10% 有限标签监督学习（性能下降分析）
  - FixMatch 半监督学习（10% 有标签 + 大量无标签数据）
- **大规模网格搜索**（64+ 组完整结果）：
  - 监督学习：batch size 32/64/96/128 × full vs 10% 标签 × fixed/scaled 学习率策略
  - FixMatch 半监督：batch × μ(2/4) × 两种 lr 策略
- 配套工具链：一键网格运行脚本（`run_supervised_grid.sh`、`run_fixmatch_grid.sh`）、结果自动汇总工具、实时监控脚本
- 报告与答辩素材：7 张核心图表 + `practice2_section_2to3_pages.*`（docx/html/md，可直接插入报告）
- 其他：中文字体支持、完整实验结果归档、`实验结果汇总.md`、`监控提示词.md`

**主要目录**：`mstar_ssl_practice2_clean/`（重写版，支撑完整实践二与答辩）

**English**  
**Official requirements**: Study the impact of limited labels on performance and use semi-supervised methods to recover accuracy.

**Full implementation in this repository** (key deliverable of v1.0.0):
- Three core controlled experiments:
  - Fully supervised (100% labels, upper-bound reference)
  - 10% limited-label supervised (performance degradation analysis)
  - FixMatch semi-supervised (10% labeled + large unlabeled data)
- **Large-scale hyperparameter grid search** (64+ complete runs):
  - Supervised: batch sizes 32/64/96/128 × full vs 10% labels × fixed/scaled LR strategies
  - FixMatch: batch × μ(2/4) × two LR strategies
- Toolchain: one-click grid scripts, automatic result summarization tools, real-time monitoring scripts
- Report & defense assets: 7 core figures + multi-format section documents ready for reports
- Additional: Chinese font support, full archived results, experiment summary and monitoring prompt docs

**Primary directory**: `mstar_ssl_practice2_clean/` (rewritten version supporting complete Practice 2 and defense)

---

### 实践三：深度学习应用实践（锂离子电池 SOH 预测） / Practice 3: Real-World DL Application (Li-ion Battery SOH Prediction)

**中文**  
**对应任务书要求**：使用深度学习解决锂离子电池健康状态（SOH）预测实际问题，完成数据预处理、三种划分、特征选择、模型训练与评估。

**本仓库完整实现**（v1.0.0 重点交付）：
- 数据预处理：NASA 电池数据读取、放电 cycle 提取、异常样本（容量尖峰）剔除
- 三种标准划分：
  - A. 单电池随机 60/20/20
  - B. 单电池时序前 60% / 后 40%
  - C. 跨电池迁移（源电池全量 + 目标电池前 10%）
- 特征选择：Pearson 相关系数（PCC）分析 + Top-K 特征及理由
- 模型：小型 MLP 回归网络，完成 **20 次独立随机实验**
- 完整评估：MAE、RMSE、R² + 预测曲线、Loss 曲线
- 补充消融实验：
  - 容量尖峰剔除前后对比
  - C 类迁移中是否使用 cycle_index 特征的对比
- 自动生成答辩素材：`presentation_summary.md` + 51 张报告图片

**主要目录**：`battery_soh_practice3_rewrite/`（按任务书要求模块化，文档完善）

**English**  
**Official requirements**: Use deep learning to solve the real-world problem of lithium-ion battery State-of-Health (SOH) prediction. Perform data preprocessing, three split strategies, feature selection, model training and evaluation.

**Full implementation in this repository** (key deliverable of v1.0.0):
- Data preprocessing: NASA battery data loading, discharge cycle extraction, anomaly removal (capacity spikes)
- Three standard splits:
  - A. Single battery random 60/20/20
  - B. Temporal (first 60% / last 40% cycles)
  - C. Cross-battery transfer (full source + first 10% of target)
- Feature selection: Pearson Correlation Coefficient (PCC) analysis + Top-K features with justifications
- Model: Small MLP regressor trained with **20 independent random runs**
- Full evaluation: MAE, RMSE, R² + prediction curves and loss histories
- Additional ablation studies:
  - Effect of capacity spike cleaning
  - Impact of `cycle_index` features in transfer setting
- Auto-generated defense materials: `presentation_summary.md` + 51 report figures

**Primary directory**: `battery_soh_practice3_rewrite/` (modularized according to task requirements with rich documentation)

---

### 扩展实验：模型结构搜索与压缩 / Extended Experiments: Architecture Search & Compression

**中文**  
用于课程论文的模型优化方向实验，采用遗传算法（GA）和粒子群优化（PSO）在保持 ResNet32 深度不变的前提下搜索通道配置。

- **Stage-level 搜索**（3 个 stage）：多个项目变体在 H100 / L40S 上运行
- **Block-level 成熟方案**：对 15 个 residual block 分别搜索通道数，使用 BN gamma 重要性先验引导 + 可选知识蒸馏（KD）
- 推荐结果示例：`[12, 24, 32]` 配置在精度损失仅 3.21% 的情况下实现 66.8% 参数压缩和约 3.2× 训练加速
- 相关分析：BF16 / FP16 混合精度实验差异

**主要目录**：
- `resnet32_ga_pso_search/`
- `resnet32_l40s_mature_ga_pso/`（block-level 成熟版）
- `resnet32_l40s_optimized_ga_pso_search/`
- `resnet32_mature_ga_pso/`

**English**  
Model optimization experiments for the course paper. Use Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) to search channel configurations while keeping ResNet32 depth fixed.

- **Stage-level search** (3 stages): Multiple project variants run on H100 / L40S GPUs
- **Block-level mature solution**: Search channel widths for all 15 residual blocks, guided by BN gamma importance prior, with optional knowledge distillation (KD)
- Example recommended result: `[12, 24, 32]` achieves only 3.21% accuracy drop with 66.8% parameter compression and ~3.2× training speedup
- Related analysis: BF16 vs FP16 mixed-precision experimental differences

**Main directories**:
- `resnet32_ga_pso_search/`
- `resnet32_l40s_mature_ga_pso/` (block-level mature version)
- `resnet32_l40s_optimized_ga_pso_search/`
- `resnet32_mature_ga_pso/`

---

## 主要版本 / Releases

- **[v1.0.0](https://github.com/xiaoqianran/usts-202605/releases/tag/v1.0.0)** (2026-05-27)  
  实践二三完整实验交付（PR #1）。包含全部网格搜索结果、报告素材、工具链与代码文档增强。  
  Complete delivery of Practices 2 & 3 (PR #1). Includes full grid search results, report assets, toolchains, and enhanced code documentation.

---

## 仓库结构 / Repository Structure

每个主要实验目录基本独立，典型包含：
- `src/` — 核心源代码
- `scripts/` / `run_*.sh` — 实验运行、监控与一键脚本
- `tools/` — 结果汇总、素材生成等辅助工具
- `requirements.txt` + `README.md`
- `runs/`、`outputs/`、`report_assets/` — 实验结果、指标、图表与答辩素材

---

## 注意事项 / Notes

- 大型模型权重（`.pt`）、原始数据集（MSTAR、CIFAR）及部分海量 runs 目录已按需 gitignore 或单独管理
- 详见各子目录 README 获取复现步骤、设计说明与关键结论
- 多数实验运行于 NVIDIA H100 / L40S GPU 环境
- 本仓库记录了从基础实现到大规模超参研究、实际应用建模与模型结构搜索的完整课程实践过程

---

**English Summary**  
This repository collects all experiments and deliverables for the USTS AI Development Practical Training II course (May 2026). It includes full implementations of the three official practices, extensive hyperparameter studies, real-world application modeling (battery SOH), neural architecture search with evolutionary algorithms, supporting tools, and rich presentation materials for reports and defenses. The latest major release is **v1.0.0**.

---

*本 README 为中英双语版本，方便课程组内审阅与对外展示。*  
*This README is provided in both Chinese and English for internal course review and external presentation.*