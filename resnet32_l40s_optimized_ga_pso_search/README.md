# ResNet32 CIFAR-10 + GA/PSO（L40S Optimized）

本项目为 L40S 优化增强版，用于课程论文：

- 保持 ResNet32 深度不变，使用 GA/PSO 搜索 `[c1, c2, c3]` 通道配置
- 默认采用 **BF16 混合精度**（`--amp --amp-dtype bf16`）
- 集成 channels-last、TF32、权重继承（`--baseline-ckpt`）、改进 fitness 等加速与稳定手段
- 完整对比压缩前后 Accuracy、Params、FLOPs、训练时间

> 说明：本阶段为 stage-level 通道宽度搜索（非真剪枝）。目标是跑通 “baseline → 智能搜索 → 最优模型 → 指标对比” 闭环，并利用 L40S 硬件特性获得更好速度与稳定性。

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. 快速开始

### (1) 查看模型信息
```bash
python model_info.py
python model_info.py --channels 16,24,48
```

### (2) 训练 baseline（Batch Size 消融，推荐 BF16）

本版本推荐在 baseline 阶段测试以下 batch size：

- **128, 256, 512, 1024**

**重要**：脚本不会自动缩放学习率，建议采用 Linear Scaling Rule：

```
lr = 0.1 × (batch_size / 128)
```

推荐使用 **BF16 混合精度**（本项目默认优化方向）：

```bash
# bs=128（标准 baseline）
python train_resnet32.py \
  --run-name resnet32_bs128 \
  --batch-size 128 --lr 0.1 \
  --milestones 100,150 \
  --amp --amp-dtype bf16 --channels-last

# bs=256
python train_resnet32.py \
  --run-name resnet32_bs256 \
  --batch-size 256 --lr 0.2 \
  --milestones 100,150 \
  --amp --amp-dtype bf16 --channels-last

# bs=512
python train_resnet32.py \
  --run-name resnet32_bs512 \
  --batch-size 512 --lr 0.4 \
  --milestones 100,150 \
  --amp --amp-dtype bf16 --channels-last

# bs=1024（大 batch）
python train_resnet32.py \
  --run-name resnet32_bs1024 \
  --batch-size 1024 --lr 0.8 \
  --milestones 100,150 \
  --amp --amp-dtype bf16 --channels-last
```

**完整学习率调度（200 epochs，milestones=[100,150], gamma=0.1）**：

| Batch Size | 初始 LR | Epoch 0–99 | Epoch 100–149 | Epoch 150–199 |
|------------|---------|------------|---------------|---------------|
| 128        | 0.1     | 0.10       | 0.01          | 0.001         |
| 256        | 0.2     | 0.20       | 0.02          | 0.002         |
| 512        | 0.4     | 0.40       | 0.04          | 0.004         |
| 1024       | 0.8     | 0.80       | 0.08          | 0.008         |

每轮当前学习率会记录在 `metrics.csv`，可用于验证。

输出示例：`runs/resnet32_bs{128,256,512,1024}/{best.pt, metrics.csv, summary.json}`

---

### (3) GA/PSO 搜索通道配置（推荐优化路径）

推荐使用 L40S 优化配置（BF16 + channels-last + 权重继承 + 改进 fitness）：

```bash
# 快速搜索（已有 baseline 时推荐）
bash scripts/run_l40s_search_fast.sh

# 或手动：
python search_channels_ga_pso.py \
  --algorithm both \
  --search-epochs 1 \
  --batch-size 1536 \
  --num-workers 8 \
  --max-train-samples 10000 \
  --max-val-samples 5000 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp --amp-dtype bf16 --channels-last
```

默认搜索空间（可通过 `--space` 自定义）：
- `c1 ∈ {8,12,16}`、`c2 ∈ {16,20,24,28,32}`、`c3 ∈ {32,40,48,56,64}`

核心优化特性：
- BF16 混合精度（更稳定）
- `--baseline-ckpt` 权重继承（短训练也能得到可靠 proxy）
- 改进 fitness（相对短 baseline 的精度约束 + penalty）
- 支持 train/val split 避免 test 泄漏

输出：`runs/channel_search_fast_*/{best_result.json, final_train_command.txt, ...}`

搜索完成后直接查看推荐的最终训练命令：
```bash
cat runs/channel_search_fast_*/final_train_command.txt
```

---

### (4) 训练最优压缩模型 + 对比

按搜索输出的推荐命令执行，例如：

```bash
python train_width_resnet32.py \
  --channels 16,24,56 \
  --run-name final_pso_16-24-56 \
  --epochs 80 --milestones 40,60 \
  --batch-size 1024 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt
```

对比：
```bash
python compare_results.py \
  --baseline runs/resnet32_bs128/summary.json \
  --compressed runs/final_pso_16-24-56/summary.json \
  --output runs/final_comparison.json
```

---

## 3. 适应度函数

**基础版**（与老版本兼容）：
```
Fitness(x) = Acc(x) - 100 * [λp·Params(x)/Params0 + λf·FLOPs(x)/FLOPs0]
```

**本优化版推荐**：使用相对短训练 baseline 的约束版本（搜索脚本默认已启用），可有效避免搜索出过小的模型。

---

## 4. 主要脚本与优化特性

| 脚本 / 特性              | 说明 |
|--------------------------|------|
| `train_resnet32.py`      | baseline 训练（支持 BF16 + channels-last） |
| `search_channels_ga_pso.py` | GA/PSO 搜索（支持权重继承 + 改进 fitness） |
| `train_width_resnet32.py` | 训练最终压缩模型 |
| `run_l40s_*_fast.sh`     | L40S 一键快速脚本 |
| `src/utils/accelerate.py` | BF16/FP16 自动选择、GradScaler 智能开关、TF32 设置 |
| `src/utils/checkpoint.py` | `load_sliced_baseline_weights`（权重继承核心） |

推荐始终开启：`--amp --amp-dtype bf16 --channels-last`

---

## 5. 输出目录结构

```
runs/
├── resnet32_bs{128,256,512,1024}/     # 不同 batch size baseline（BF16）
├── channel_search_fast_*/             # 搜索过程 + 最优配置 + 推荐命令
├── final_* /                          # 最终压缩模型训练结果
└── final_comparison.json              # 论文核心对比指标
```

---

## 6. 参考文档

- `docs/l40s_optimized_notes.md` — L40S 优化改动详细说明
- `docs/optimization_model.md` — 优化问题建模
- `docs/experiment_plan.md` — 实验流程建议

如需恢复原始风格，可参考备份文件 `README.md.bak`。
