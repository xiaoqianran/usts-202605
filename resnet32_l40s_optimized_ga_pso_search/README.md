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

#### Baseline 训练结果（Batch Size 消融实验，L40S + BF16）

在 L40S 上使用 BF16 + channels-last 训练 200 epochs 后的实际结果：

| Batch Size | 最佳测试 Acc (%) | 最佳 Epoch | 训练总时长        | Params | FLOPs   | 备注          |
|------------|------------------|------------|-------------------|--------|---------|---------------|
| 128        | **92.83**        | 168        | 2373s (39.5min)   | 464154 | 68.86M  | **精度最高**  |
| 256        | 92.37            | 168        | 1813s (30.2min)   | 464154 | 68.86M  | -             |
| 512        | 91.60            | 180        | 1436s (23.9min)   | 464154 | 68.86M  | 速度较好      |
| 1024       | 91.24            | 190        | 1432s (23.9min)   | 464154 | 68.86M  | 大 batch 最快 |

**观察**：与主项目 H100 结果趋势一致，bs128 精度最高；L40S + BF16 下整体训练速度更快。推荐将 `runs/resnet32_bs128` 作为 L40S 实验的主要 baseline 参考。

**为什么 L40S 实验推荐使用自己训练的 Baseline？**

- **公平性**：L40S 上训练的 compressed 模型使用了 BF16 + `channels-last` + 特定 batch size 等优化，训练动态和收敛行为与 H100 存在差异。若直接复用主项目的 baseline 计算 accuracy_drop，会引入额外变量，导致结果不严谨。
- **一致性**：权重继承（`--baseline-ckpt`）和最终指标对比，都应在相同硬件 + 相同训练设置下进行，才能保证可比性。
- **历史教训**：之前因缺少当前环境的 baseline 做继承，导致短训练 proxy 精度仅 11.56%，搜索完全失效。
- **课程规范**：人工智能实训课程论文通常要求 baseline 与实验模型在同一实验闭环下训练和评估，跨硬件/设置混用 baseline 容易被指出问题。

因此，本目录所有 L40S 实验统一以 `resnet32_bs128`（或后续专门训的 `resnet32_baseline_l40s`）作为官方 baseline。

---

### (3) 直接使用主项目搜索结果进行完整训练（当前阶段跳过 L40S 搜索）

由于主项目（H100）已经通过多次 GA/PSO 搜索得到了可靠的最优配置，我们**不再使用** `scripts/run_l40s_search_fast.sh` 进行重复搜索。

直接选取上一步（主项目）搜索出的最好配置，在 L40S 上进行完整 80-epoch 训练 + 对比。

**推荐优先训练的三个配置**（来自主项目真实搜索结果）：
- `8,16,32` （最激进压缩，多次被 GA 选中）
- `8,20,32` （平衡型）
- `12,24,32` （某次 PSO 快速评估中精度最高）

#### 准备工作（必须先做）
```bash
# 使用我们已有的最佳 baseline（bs128 精度最高 92.83%）
mkdir -p runs/resnet32_baseline
cp runs/resnet32_bs128/best.pt runs/resnet32_baseline/best.pt
```

#### 直接训练命令（L40S 完整优化参数）

> **重要说明（Batch Size 选择）**：  
> 因为我们已决定以 `resnet32_bs128`（使用 batch=128 + lr=0.1 训练，精度最高）作为本目录 L40S 实验的官方 baseline，为了保持训练动态和学习率调度的一致性，这里**统一使用 `--batch-size 128 --lr 0.1`**。  
> 如果你想用更大 batch 加速（512/1024），建议同时切换使用对应的 `resnet32_bs512` 或 `resnet32_bs1024` 作为 baseline，并按 Linear Scaling Rule 调整 lr（例如 batch 1024 用 lr=0.8），否则 accuracy drop 的对比会不够公平。

```bash
# 1. 最激进配置 8-16-32（推荐第一个跑）
python train_width_resnet32.py \
  --channels 8,16,32 \
  --run-name final_l40s_8-16-32 \
  --epochs 80 --milestones 40,60 \
  --batch-size 128 --lr 0.1 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt
```

```bash
# 2. 平衡配置 8-20-32
python train_width_resnet32.py \
  --channels 8,20,32 \
  --run-name final_l40s_8-20-32 \
  --epochs 80 --milestones 40,60 \
  --batch-size 128 --lr 0.1 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt
```

```bash
# 3. 相对保守、快速搜索精度最高 12-24-32
python train_width_resnet32.py \
  --channels 12,24,32 \
  --run-name final_l40s_12-24-32 \
  --epochs 80 --milestones 40,60 \
  --batch-size 128 --lr 0.1 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt
```

#### 对比命令（训练完后执行）
```bash
# 对比 8-16-32
python compare_results.py \
  --baseline runs/resnet32_bs128/summary.json \
  --compressed runs/final_l40s_8-16-32/summary.json \
  --output runs/final_l40s_comparison_8-16-32.json

# 对比 8-20-32
python compare_results.py \
  --baseline runs/resnet32_bs128/summary.json \
  --compressed runs/final_l40s_8-20-32/summary.json \
  --output runs/final_l40s_comparison_8-20-32.json

# 对比 12-24-32
python compare_results.py \
  --baseline runs/resnet32_bs128/summary.json \
  --compressed runs/final_l40s_12-24-32/summary.json \
  --output runs/final_l40s_comparison_12-24-32.json
```

#### L40S 实际完整训练结果（基于主项目遗传算法配置）

本目录近期尝试运行 `run_l40s_search_fast.sh` 产生的 `channel_search_fast_*` 结果因 baseline 继承问题（proxy 精度仅 ~11%）而无效。

实际可用的结果是直接使用主项目遗传算法搜索出的三个最优配置，在本 L40S 环境下完整训练 80 epoch 后的真实性能（均使用与官方 baseline 一致的 batch=128 + lr=0.1 设置）：

| 配置       | 完整训练 Acc | 相对本地 bs128 Baseline (92.83%) | Params 压缩率 | FLOPs 缩减率 | 训练时长 (sec) | 备注             |
|------------|--------------|----------------------------------|---------------|--------------|----------------|------------------|
| 8-16-32    | 89.18%       | **-3.65%**                       | 74.82%        | 74.84%       | 770            | 最激进压缩       |
| 8-20-32    | 89.14%       | -3.69%                           | 71.98%        | 70.29%       | 802            | 平衡型           |
| **12-24-32** | **90.92%** | **-1.91%**                       | 66.80%        | 53.59%       | 799            | **精度损失最小** |

**观察**：
- 在当前 L40S + BF16 + channels-last 设置下，`12-24-32` 是三个配置里精度损失最小的（仅掉 1.91%）。
- 两个带 8 的激进配置掉点都在 3.6~3.7% 左右，压缩收益显著。
- 这些结果可直接用于课程论文的 L40S 实验部分。

对比 JSON 已生成在 `runs/final_l40s_comparison_*.json`。

> **说明**：`run_l40s_search_fast.sh` 脚本在本阶段已暂停使用（搜索工作已在主项目完成）。如需以后在 L40S 上重新搜索，可手动启用并确保 baseline checkpoint 存在。

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
