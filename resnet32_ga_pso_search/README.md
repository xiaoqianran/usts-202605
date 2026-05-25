# ResNet32 CIFAR-10 + GA/PSO 通道配置搜索

本项目用于课程论文：在保持 ResNet32 深度不变的前提下，使用 GA/PSO 搜索三个 stage 的通道配置 `[c1, c2, c3]`，实现 stage-level 结构化压缩，并完整对比压缩前后指标。

> 说明：本阶段为 stage-level 通道宽度搜索（非真剪枝），目标是跑通 "baseline → 智能搜索 → 最优模型 → 指标对比" 闭环。

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. 快速开始

### (1) 查看模型信息
```bash
python model_info.py                    # 标准 ResNet32
python model_info.py --channels 16,24,48  # 指定压缩配置
```

### (2) 训练 baseline（含 Batch Size 消融）

推荐在 baseline 阶段测试以下 batch size（CIFAR-10 实用区间）：

- **128, 256, 512, 1024**

**重要提示**：`train_resnet32.py` 不会自动根据 batch size 调整学习率。建议采用 **Linear Scaling Rule**：

```
lr = 0.1 × (batch_size / 128)
```

**完整学习率调度（200 epochs，milestones=[100,150], gamma=0.1）**：

| Batch Size | 初始 LR | Epoch 0–99 | Epoch 100–149 | Epoch 150–199 |
|------------|---------|------------|---------------|---------------|
| 128        | 0.1     | 0.10       | 0.01          | 0.001         |
| 256        | 0.2     | 0.20       | 0.02          | 0.002         |
| 512        | 0.4     | 0.40       | 0.04          | 0.004         |
| 1024       | 0.8     | 0.80       | 0.08          | 0.008         |

milestones 在第 100 和 150 epoch 触发衰减（×0.1）。不同 batch size 仅按比例放大初始 LR，衰减时机和倍率保持一致。训练过程中每轮的当前学习率会记录在 `metrics.csv` 中，可用于验证。

示例命令（200 epochs，标准 MultiStep 衰减）：

```bash
# bs=128（标准 baseline）
python train_resnet32.py --run-name resnet32_bs128 --batch-size 128 --lr 0.1 --milestones 100,150 --amp

# bs=256
python train_resnet32.py --run-name resnet32_bs256 --batch-size 256 --lr 0.2 --milestones 100,150 --amp

# bs=512
python train_resnet32.py --run-name resnet32_bs512 --batch-size 512 --lr 0.4 --milestones 100,150 --amp

# bs=1024（大 batch 上限）
python train_resnet32.py --run-name resnet32_bs1024 --batch-size 1024 --lr 0.8 --milestones 100,150 --amp
```

输出示例：`runs/resnet32_bs{128,256,512,1024}/{best.pt, metrics.csv, summary.json}`

后续的通道搜索与最终压缩模型训练，建议基于其中一个 batch size 的 baseline 结果进行。

### (3) GA/PSO 搜索通道配置
```bash
python search_channels_ga_pso.py \
  --algorithm both \
  --search-epochs 3 \
  --ga-population 8 --ga-generations 5 \
  --pso-particles 8 --pso-iterations 5 \
  --max-train-samples 5000 --max-test-samples 2000 \
  --amp
```
默认搜索空间：
- `c1 ∈ {8,12,16}`、`c2 ∈ {16,20,24,28,32}`、`c3 ∈ {32,40,48,56,64}`

输出：`runs/channel_search_ga_pso/{search_config.json, evaluations.csv, ga_best.json, pso_best.json, best_result.json}`

### (4) 训练最优压缩模型 + 对比
```bash
# 按搜索结果中的推荐命令执行，例如：
python train_width_resnet32.py \
  --channels 16,24,48 --run-name final_width_16-24-48 \
  --epochs 80 --milestones 40,60 --amp

python compare_results.py \
  --baseline runs/resnet32_bs128/summary.json \
  --compressed runs/final_width_16-24-48/summary.json \
  --output runs/final_comparison.json
```

---

## 3. 适应度函数

```
Fitness(x) = Acc(x) - 100 * [λp·Params(x)/Params0 + λf·FLOPs(x)/FLOPs0]
```

- `x = [c1, c2, c3]`
- 默认 `λp = λf = 0.15`（可通过 `--lambda-p` / `--lambda-f` 调整）
- 精度优先 → 减小 λ；压缩优先 → 增大 λ

---

## 4. 主要脚本

| 脚本 | 用途 |
|------|------|
| `train_resnet32.py` | 训练标准 ResNet32 baseline（支持不同 batch size） |
| `search_channels_ga_pso.py` | GA/PSO 搜索通道配置 |
| `train_width_resnet32.py` | 训练搜索得到的压缩模型 |
| `compare_results.py` | baseline vs compressed 指标对比 |
| `model_info.py` | 查看 Params / FLOPs |

核心模块：
- `src/models/resnet32_cifar.py` — 标准 ResNet32
- `src/models/resnet32_width.py` — 可变通道 ResNet32（深度仍为 32）
- `src/data/cifar10.py` — CIFAR-10 dataloader（支持子集加速搜索）

---

## 5. 输出目录结构

```
runs/
├── resnet32_bs{128,256,512,1024}/   # 不同 batch size 的 baseline 结果
├── channel_search_ga_pso/           # 搜索过程与最优配置
└── final_width_*/                   # 最终压缩模型训练结果
```

课程论文核心指标（`final_comparison.json`）：
- `accuracy_drop`、`params_compression_rate`、`flops_reduction_rate`、`train_time_sec` 等。
