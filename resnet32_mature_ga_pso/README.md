# ResNet32 成熟版：重要性引导 Block-level 通道搜索 + KD

本项目是课程论文的**成熟方案**实现：

- 保持 ResNet32 深度结构不变
- 对 **15 个 residual block** 分别搜索通道数（细粒度 block-level）
- 使用 baseline 的 BN gamma 作为重要性先验，引导搜索
- 支持可选的知识蒸馏（KD）恢复精度
- 完整对比 Accuracy、Params、FLOPs、训练时间

```text
标准 ResNet32 baseline
    ↓
读取每个 block 的 BN gamma 作为重要性
    ↓
GA / PSO 搜索 15 维 block 通道配置
    ↓
训练最优 block-width 模型（可选 KD）
    ↓
与 baseline 全面对比
```

这比 stage-level（仅 3 个变量）更接近真实结构化压缩场景。

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. 查看模型复杂度

标准 ResNet32：
```bash
python model_info.py
```

任意 block-level 配置：
```bash
python model_info.py --block-channels 16,16,12,12,16,32,28,24,24,28,64,56,48,48,56
```

---

## 3. 训练标准 ResNet32 Baseline（Teacher）

快速测试：
```bash
python train_resnet32.py --epochs 2 --batch-size 128 --lr 0.05 --milestones 1 --run-name resnet32_quick
```

正式训练（推荐）：
```bash
python train_resnet32.py \
  --run-name resnet32_baseline \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --milestones 100,150 \
  --amp
```

输出：`runs/resnet32_baseline/{best.pt, summary.json, metrics.csv}`

这个 checkpoint 将同时用于：
- 搜索阶段的重要性先验
- 搜索阶段的 sliced weight inheritance（如果使用优化版）
- 最终训练阶段的 KD teacher

---

## 4. GA/PSO Block-level 搜索（15 维）

搜索空间（每个 block 独立）：
- Stage1（5 blocks）：{8, 12, 16}
- Stage2（5 blocks）：{16, 20, 24, 28, 32}
- Stage3（5 blocks）：{32, 40, 48, 56, 64}

快速搜索：
```bash
python search_block_channels_ga_pso.py \
  --algorithm both \
  --baseline-checkpoint runs/resnet32_baseline/best.pt \
  --search-epochs 1 \
  --ga-population 4 \
  --ga-generations 2 \
  --pso-particles 4 \
  --pso-iterations 2 \
  --max-train-samples 1000 \
  --max-test-samples 500
```

较正式搜索：
```bash
python search_block_channels_ga_pso.py \
  --algorithm both \
  --baseline-checkpoint runs/resnet32_baseline/best.pt \
  --search-epochs 2 \
  --ga-population 8 \
  --ga-generations 5 \
  --pso-particles 8 \
  --pso-iterations 5 \
  --max-train-samples 5000 \
  --max-test-samples 2000 \
  --amp
```

输出目录：`runs/block_channel_search_ga_pso/`
- `ga_best.json` / `pso_best.json`
- `best_result.json`
- `evaluations.csv` 等历史记录

---

## 5. 训练最终 Block-level 压缩模型（支持 KD）

搜索完成后，终端会打印推荐命令。典型用法：

**带 KD（推荐，精度恢复效果好）：**
```bash
python train_block_resnet32_kd.py \
  --block-channels 16,16,12,12,16,32,28,24,24,28,64,56,48,48,56 \
  --teacher-checkpoint runs/resnet32_baseline/best.pt \
  --kd-alpha 0.5 \
  --kd-temperature 4 \
  --run-name final_block_kd \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

**不使用 KD：**
```bash
python train_block_resnet32_kd.py \
  --block-channels 16,16,12,12,16,32,28,24,24,28,64,56,48,48,56 \
  --run-name final_block_no_kd \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

---

## 6. 结果对比

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_kd/summary.json \
  --search-result runs/block_channel_search_ga_pso/best_result.json \
  --output runs/final_comparison.json
```

对比字段示例：
- Accuracy Drop
- Params Compression Rate
- FLOPs Reduction Rate
- Search Time
- Training Time
- KD Enabled

---

## 7. 适应度函数

```text
Fitness(x) = Acc(x)
           - 100 * [λp * Params(x)/Params0 + λf * FLOPs(x)/FLOPs0]
           - λt * TimePenalty(x)
```

默认参数：
- `λp = 0.10`
- `λf = 0.15`
- `λt = 0.02`

调整建议见原 README 或实验记录。

---

## 8. 核心文件

| 文件/目录                        | 作用 |
|----------------------------------|------|
| `search_block_channels_ga_pso.py` | 15 维 block-level GA/PSO 搜索 |
| `train_block_resnet32_kd.py`     | 支持 KD 的 block-width 模型训练 |
| `src/search/importance.py`       | BN gamma 重要性计算 |
| `src/models/resnet32_blockwidth.py` | 支持任意 15 维通道配置的 ResNet32 |
| `src/utils/checkpoint.py`        | 权重继承相关工具 |

---

如需 L40S 加速版（BF16 + channels-last + 权重继承 + 更快脚本），请使用配对项目 `resnet32_l40s_mature_ga_pso`。

备份文件：`README.md.bak`
