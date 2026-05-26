# ResNet32 CIFAR-10 + GA/PSO（L40S Optimized）

本项目为 L40S 优化增强版，用于课程论文：

- 保持 ResNet32 深度不变，使用 GA/PSO 搜索 `[c1, c2, c3]` 通道配置
- 支持 **FP16/BF16 混合精度**（本轮 baseline 消融中 FP16 精度最佳）
- 集成 channels-last、TF32、权重继承（`--baseline-ckpt`）、改进 fitness 等加速与稳定手段
- 完整对比压缩前后 Accuracy、Params、FLOPs、训练时间

> 说明：本阶段为 stage-level 通道宽度搜索（非真剪枝）。本目录重点是利用 L40S 硬件特性，在已验证的最优配置上进行优化训练与对比。

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. L40S 实验结果（已完成）

**推荐配置**：`[12, 24, 32]` —— 在旧 L40S + BF16 + channels-last 对比中，三个配置里精度损失最小（仅 1.91%），综合表现最佳。

> 重要修正：以下压缩结果来自本目录旧数据管线（`torchvision.datasets.CIFAR10`）。它和 `resnet32_l40s_mature_ga_pso` 的本地 `LocalCIFAR10` 数据管线不一致，因此不应该再用来断言 “BF16 + channels-last 一定更好”。本目录已修正为与 mature 项目一致的数据读取、增强、persistent workers 和 prefetch 设置，并已完成第 3.2 节的 `128 / 256 / 512 / 1024 × FP16 / BF16` baseline 消融。新的正式 baseline 建议使用 `resnet32_l40s_b128_fp16`（93.30%）。

### 2.1 最终对比结果

Baseline 使用本地训练的 `resnet32_bs128`（L40S + BF16，200 epochs，92.83%）；压缩模型使用相同优化设置完成 80 epochs 训练。所有 compressed 模型均从本地 baseline 继承权重。

| 配置       | 完整训练 Acc | 相对本地 bs128 Baseline (92.83%) | Params 压缩率 | FLOPs 缩减率 | 训练时长 (sec) | 备注             |
|------------|--------------|----------------------------------|---------------|--------------|----------------|------------------|
| 8-16-32    | 89.18%       | -3.65%                           | 74.82%        | 74.84%       | 770            | 最激进压缩       |
| 8-20-32    | 89.14%       | -3.69%                           | 71.98%        | 70.29%       | 802            | 平衡型           |
| **12-24-32** | **90.92%** | **-1.91%**                       | 66.80%        | 53.59%       | 799            | **精度损失最小** |

**关键观察**：
- 在当前 L40S + BF16 + channels-last 设置下，`12-24-32` 是三个配置里精度损失最小的（仅掉 1.91%）。
- 两个带 8 的激进配置掉点都在 3.6~3.7% 左右，压缩收益显著。
- 这些结果可直接用于课程论文的 L40S 实验部分。

**对比 JSON** 已生成：
- `runs/final_l40s_comparison_8-16-32.json`
- `runs/final_l40s_comparison_8-20-32.json`
- `runs/final_l40s_comparison_12-24-32.json`

### 2.2 为什么 L40S 实验必须使用自己训练的 Baseline？

- **公平性**：L40S 上训练的 compressed 模型使用了 BF16 + `channels-last` + 特定 batch size 等优化，训练动态和收敛行为与 H100 存在差异。若直接复用主项目的 baseline 计算 accuracy_drop，会引入额外变量。
- **一致性**：权重继承（`--baseline-ckpt`）和最终指标对比，都应在相同硬件 + 相同训练设置下进行。
- **历史教训**：之前因缺少当前环境的 baseline 做继承，导致短训练 proxy 精度仅 11.56%，搜索完全失效。
- **课程规范**：人工智能实训课程论文通常要求 baseline 与实验模型在同一实验闭环下训练和评估。

因此，旧压缩结果仍按当时的 `resnet32_bs128`（92.83%）解释；后续新一轮 L40S 实验应统一以 `resnet32_l40s_b128_fp16`（93.30%）作为官方 baseline，并复制到 `runs/resnet32_baseline/` 用于权重继承。

### 2.3 当前问题诊断与修复

你对 BF16 效果不好的判断是对的：旧结果中 `resnet32_bs128` 只有 **92.83%**，低于普通项目 `resnet32_ga_pso_search` 的 **93.21%**，也明显低于成熟 L40S 项目 `resnet32_l40s_mature_ga_pso` 的 **93.45%**。

目前确认的问题：
- `train_resnet32.py` 和 `src/utils/accelerate.py` 与 mature 项目一致，BF16/FP16 autocast 和 GradScaler 逻辑本身不是主要问题。
- 真正不一致的是数据管线：旧版使用 `torchvision.datasets.CIFAR10`；mature 项目使用本地 `LocalCIFAR10`、reflect padding、persistent workers、prefetch。
- 旧版 `summary.json` 没记录 `batch_size / lr / amp_dtype / channels_last`，导致 README 声称和实际运行参数无法严格核对。

已修复：
- `src/data/cifar10.py` 已对齐 mature 项目的本地 CIFAR-10 loader。
- `train_resnet32.py` 和 `train_width_resnet32.py` 的 `summary.json` 现在会记录关键训练参数。
- 新增 `scripts/prepare_cifar10.sh` 和 `scripts/run_l40s_baseline_amp_sweep.sh`，用于重新跑完整 baseline 消融。

### 2.4 Baseline 消融结果（已完成）

统一设置：ResNet32，CIFAR-10，200 epochs，milestones=[100,150]，`lr=0.1`，`num_workers=8`，AMP + channels-last。Params 均为 464,154，FLOPs 均为 68,862,592。

| Batch size | AMP dtype | Best Acc | 训练时长 (sec) | Run |
|------------|-----------|----------|----------------|-----|
| **128**    | **FP16**  | **93.30%** | 2816 | `runs/resnet32_l40s_b128_fp16` |
| 128        | BF16      | 93.00%   | 2646 | `runs/resnet32_l40s_b128_bf16` |
| 256        | FP16      | 92.81%   | 2584 | `runs/resnet32_l40s_b256_fp16` |
| 256        | BF16      | 92.48%   | 2450 | `runs/resnet32_l40s_b256_bf16` |
| 512        | FP16      | 91.97%   | 1875 | `runs/resnet32_l40s_b512_fp16` |
| 512        | BF16      | 91.62%   | 1750 | `runs/resnet32_l40s_b512_bf16` |
| 1024       | FP16      | 91.01%   | 1386 | `runs/resnet32_l40s_b1024_fp16` |
| 1024       | BF16      | 91.01%   | 1442 | `runs/resnet32_l40s_b1024_bf16` |

**结论**：
- 最高精度为 `batch=128 + FP16 + channels-last`，Best Acc = **93.30%**。
- BF16 在本轮消融中没有超过 FP16；同 batch 下 FP16 均略高于或持平 BF16。
- batch size 越大，训练时间整体更短，但精度明显下降；在本课程论文的准确率优先场景下，推荐将 `runs/resnet32_l40s_b128_fp16/best.pt` 作为新的正式 baseline。

---

## 3. 快速复现指南

### 3.1 查看模型信息

```bash
python model_info.py
python model_info.py --channels 16,24,48
```

### 3.2 重新训练本地 Baseline：128 到 1024 × FP16/BF16（已完成）

先准备数据：

```bash
bash scripts/prepare_cifar10.sh
```

然后重新跑完整 baseline 消融。推荐直接执行脚本：

```bash
bash scripts/run_l40s_baseline_amp_sweep.sh
```

等价手动命令如下，顺序严格按 batch size 从 128 到 1024，每个 batch 同时跑 FP16 和 BF16：

```bash
# === Batch 128 ===
python train_resnet32.py --run-name resnet32_l40s_b128_fp16 --epochs 200 --batch-size 128 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b128_bf16 --epochs 200 --batch-size 128 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last

# === Batch 256 ===
python train_resnet32.py --run-name resnet32_l40s_b256_fp16 --epochs 200 --batch-size 256 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b256_bf16 --epochs 200 --batch-size 256 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last

# === Batch 512 ===
python train_resnet32.py --run-name resnet32_l40s_b512_fp16 --epochs 200 --batch-size 512 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b512_bf16 --epochs 200 --batch-size 512 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last

# === Batch 1024 ===
python train_resnet32.py --run-name resnet32_l40s_b1024_fp16 --epochs 200 --batch-size 1024 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b1024_bf16 --epochs 200 --batch-size 1024 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last
```

**学习率调度表**（200 epochs，milestones=[100,150]）与 mature 项目保持一致。这里先统一 `lr=0.1`，不要对大 batch 使用 `lr=0.2/0.4/0.8`，否则 FP16/BF16 对比会混入学习率变量。

本轮已完成，结果见 [2.4 Baseline 消融结果](#24-baseline-消融结果已完成)。最高精度 checkpoint 为：

```bash
runs/resnet32_l40s_b128_fp16/best.pt
```

### 3.3 准备本地 Baseline（用于权重继承）

本轮 baseline 消融中 `b128 + FP16` 最佳，因此正式 baseline 副本应来自 `runs/resnet32_l40s_b128_fp16`：

```bash
mkdir -p runs/resnet32_baseline
cp runs/resnet32_l40s_b128_fp16/best.pt runs/resnet32_baseline/best.pt
cp runs/resnet32_l40s_b128_fp16/summary.json runs/resnet32_baseline/summary.json
cp runs/resnet32_l40s_b128_fp16/last.pt runs/resnet32_baseline/last.pt
```

### 3.4 在 L40S 上完整训练已验证的优秀配置

本目录**不再重复运行 GA/PSO 搜索**（搜索工作已在主项目完成，且曾因 baseline 继承问题导致 proxy 失效）。直接使用主项目 GA/PSO 搜索得到的最优配置，在 L40S 上进行完整 80-epoch 优化训练。

以下命令即为实际产生本目录旧压缩实验结果所使用的训练命令（均使用当时本地 baseline 一致的 batch=128 + lr=0.1 + BF16）。如果基于新的 93.30% baseline 重新训练压缩模型，建议把 `--amp-dtype bf16` 改为 `--amp-dtype fp16`，并继续使用 `runs/resnet32_baseline/best.pt` 做权重继承。

```bash
# 1. 最激进配置 8-16-32
python train_width_resnet32.py \
  --channels 8,16,32 \
  --run-name final_l40s_8-16-32 \
  --epochs 80 --milestones 40,60 \
  --batch-size 128 --lr 0.1 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt

# 2. 平衡配置 8-20-32
python train_width_resnet32.py \
  --channels 8,20,32 \
  --run-name final_l40s_8-20-32 \
  --epochs 80 --milestones 40,60 \
  --batch-size 128 --lr 0.1 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt

# 3. 推荐配置 12-24-32（精度损失最小）
python train_width_resnet32.py \
  --channels 12,24,32 \
  --run-name final_l40s_12-24-32 \
  --epochs 80 --milestones 40,60 \
  --batch-size 128 --lr 0.1 --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt
```

### 3.5 生成对比 JSON（训练完成后）

旧压缩结果的历史对比仍使用 `runs/resnet32_bs128/summary.json`。新一轮压缩模型完成后，应改用 `runs/resnet32_baseline/summary.json` 作为 baseline：

```bash
python compare_results.py \
  --baseline runs/resnet32_bs128/summary.json \
  --compressed runs/final_l40s_8-16-32/summary.json \
  --output runs/final_l40s_comparison_8-16-32.json

# 同理可生成 8-20-32 和 12-24-32 的对比文件
```

实际结果和已生成的 JSON 见 [2. L40S 实验结果](#2-l40s-实验结果已完成)。

---

## 4. 方法与优化特性

### 4.1 搜索空间与适应度函数

与主项目一致：

```
Fitness(x) = Acc(x) - 100 * [λp·Params(x)/Params0 + λf·FLOPs(x)/FLOPs0]
```

本优化版额外支持相对短训练 baseline 的约束版本（搜索脚本默认已启用），可有效避免搜索出过小的模型。

### 4.2 L40S 专属优化（已全部开启）

- FP16/BF16 混合精度（本轮 baseline 消融中 `--amp-dtype fp16` 精度最佳）
- channels-last 内存布局
- TF32 + cuDNN benchmark
- 权重继承（`--baseline-ckpt` + `load_sliced_baseline_weights`）
- 更高 `num-workers`

推荐命令中已包含以上所有优化。

### 4.3 训练协议

- Baseline：200 epochs（milestones=[100,150]）
- 压缩模型：80 epochs（milestones=[40,60]）
- baseline 重跑阶段统一使用 `lr=0.1`，只比较 batch size 和 amp dtype；本轮最佳为 `batch=128 + FP16`，Best Acc = 93.30%。
- 新一轮压缩模型应使用重新选出的 `runs/resnet32_baseline/best.pt` 进行权重继承，避免继续使用旧的 92.83% baseline。

---

## 5. 脚本与输出

### 5.1 主要脚本

| 脚本 / 特性                     | 说明 |
|--------------------------------|------|
| `train_resnet32.py`            | Baseline 训练（支持 FP16/BF16 + channels-last） |
| `train_width_resnet32.py`      | 训练最终压缩模型（支持权重继承） |
| `compare_results.py`           | 生成论文核心对比 JSON |
| `scripts/prepare_cifar10.sh`   | 准备本地 CIFAR-10 数据 |
| `scripts/run_l40s_baseline_fast.sh` | 一键训练本地 baseline |
| `scripts/run_l40s_baseline_amp_sweep.sh` | 128/256/512/1024 × FP16/BF16 baseline 消融 |
| `src/utils/accelerate.py`      | BF16/FP16 自动选择、TF32 等加速 |
| `src/utils/checkpoint.py`      | 权重继承核心实现 |

**注意**：`run_l40s_search_fast.sh` 已重命名为 `.disabled`，本阶段不再使用搜索脚本。

### 5.2 输出目录结构

```
runs/
├── resnet32_l40s_b{128,256,512,1024}_{fp16,bf16}/ # 重新跑的 baseline 消融
├── resnet32_baseline/                      # 用于权重继承的官方 baseline 副本
├── final_l40s_{8-16-32,8-20-32,12-24-32}/ # L40S 完整 80-epoch 训练结果
└── final_l40s_comparison_*.json            # 论文核心对比指标（已生成）
```

---

## 6. 参考文档

- `docs/l40s_optimized_notes.md` — L40S 优化改动详细说明
- `docs/optimization_model.md` — 优化问题建模
- `docs/experiment_plan.md` — 实验流程建议

本目录实验结果可直接用于课程论文 L40S 部分的对比与分析。如需在 L40S 上进行新的 GA/PSO 搜索，请先确保 baseline 继承机制稳定后再启用搜索脚本。

如需恢复原始风格，可参考备份文件 `README.md.bak`。
