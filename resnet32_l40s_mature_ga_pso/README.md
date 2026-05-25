# ResNet32 L40S Optimized：Block-level 通道搜索 + KD（成熟版）

本项目是 **L40S 优化增强版** 的成熟方案，对应基础版 `resnet32_mature_ga_pso`：

- 15 维 block-level 通道配置搜索（每个 residual block 独立）
- 使用 baseline BN gamma 作为重要性先验
- **默认 BF16 混合精度** + channels-last + TF32
- 搜索阶段支持 sliced weight inheritance（从 teacher 继承权重）
- 支持知识蒸馏（KD）恢复精度
- 提供一键快速脚本

```text
ResNet32 Baseline Teacher
    ↓
重要性引导 + 15 维 Block-level GA/PSO 搜索（L40S 加速）
    ↓
Sliced Weight Inheritance 加速候选评估
    ↓
最优结构 + KD 训练
    ↓
全面指标对比
```

---

## 2. L40S Baseline 精度-速度消融实验结果（已完成）

我们系统性地在 L40S 上完成了 **8 组 200-epoch baseline 训练**（Batch Size × {FP16, BF16} 混合精度），目标是找到适合本项目作为 Teacher / Importance 来源的最佳基座模型。

### 2.1 最终训练结果

| 配置                  | 测试 Acc | 最佳 Epoch | 训练时长    | 相对 b128_fp16 加速 | 备注 |
|-----------------------|----------|------------|-------------|---------------------|------|
| **b128 + BF16**       | **93.45%** | 193        | 1811s      | **1.14×**           | **最高精度，强烈推荐作为主 Teacher** |
| b128 + FP16           | 93.10%   | 120        | 2070s      | 1.00× (基准)        | 非常稳定，精度优秀 |
| b256 + BF16           | 92.57%   | 186        | 1678s      | 1.23×               | 精度掉点 0.88% |
| b256 + FP16           | 92.38%   | 139        | 1862s      | 1.11×               | - |
| b512 + BF16           | 91.80%   | 137        | 1281s      | 1.62×               | 精度掉点 1.65% |
| b512 + FP16           | 91.71%   | 158        | 1405s      | 1.47×               | - |
| b1024 + BF16          | 90.95%   | 122        | 1177s      | 1.76×               | 精度掉点 2.5% |
| b1024 + FP16          | 91.01%   | 146        | 1081s      | **1.91×**           | 最快，但精度损失最大 |

**数据来源**：各 `runs/resnet32_l40s_b*/summary.json`

### 2.2 关键发现

- **意外收获**：`b128 + BF16` 达到了 **93.45%**，比之前非 L40S 版本的 93.11% 还高！说明在小 batch 下，BF16 + channels-last + TF32 组合对这个模型非常友好。
- **Batch Size 影响显著**：随着 batch size 从 128 → 1024，精度持续下降（尤其是 >256 后掉点明显）。这验证了我们之前的判断：大 batch 在 ResNet32+CIFAR-10 上优化难度大。
- **BF16 vs FP16**：在相同 batch size 下，BF16 通常收敛更好或持平（特别是 b128）。
- **速度收益**：b1024 版本训练时间只有 b128 的 ~52~55%，但精度损失 2.1~2.5%，性价比不高。
- **推荐用于后续实验**：
  - **主 Teacher / Importance 来源**：`resnet32_l40s_b128_bf16`（93.45%）
  - **速度-质量平衡备选**：`resnet32_l40s_b256_bf16`（如果后面搜索想更快）

### 2.3 课程论文使用提示

核心文件（可直接引用）：
- 最佳 baseline：`runs/resnet32_l40s_b128_bf16/summary.json` + `best.pt`
- 完整消融数据：上面 8 个 `resnet32_l40s_b*/summary.json`

### 2.4 正式基座模型确定

已将目前最好的模型（`b128 + BF16`，93.45%）复制到项目标准 baseline 目录，后续 block-level 搜索、权重继承和 KD 训练均使用统一路径：

```bash
mkdir -p runs/resnet32_baseline
cp runs/resnet32_l40s_b128_bf16/best.pt      runs/resnet32_baseline/
cp runs/resnet32_l40s_b128_bf16/summary.json runs/resnet32_baseline/
cp runs/resnet32_l40s_b128_bf16/last.pt      runs/resnet32_baseline/
```

现在项目内推荐使用以下标准路径：
- `runs/resnet32_baseline/best.pt` → 作为 Teacher 和 BN gamma 重要性先验来源
- `runs/resnet32_baseline/summary.json` → 作为 baseline 对比依据

---

## 3. 安装

```bash
pip install -r requirements.txt
```

---

## 4. 核心优化特性（L40S 版）

本版本相比基础 mature 版主要增强：

- `--amp --amp-dtype bf16`（默认推荐，数值稳定性更好）
- `--channels-last` 内存布局加速
- `setup_torch_fast`（TF32 + cudnn benchmark）
- 搜索阶段 **sliced weight inheritance**（`--baseline-ckpt`）
- 更高效的 DataLoader 配置（高 num_workers + persistent workers）
- 快速一键脚本（`run_l40s_*_fast.sh`）

---

## 5. 本次 Baseline 消融实验的执行记录（已全部完成）

**重要说明**：本节记录的是**本次实验实际执行的步骤和命令顺序**（结果已完整记录在第 2 节）。  
当你看到这里的命令时，请注意：**这些命令已经跑完**，不要误以为还需要重新执行。

本次实验严格按照以下顺序进行：

1. 数据准备
2. 按 batch size 从小到大，依次训练 FP16 和 BF16 两个精度版本（共 8 组）
3. 对比结果后，选出最佳模型并固化到 `runs/resnet32_baseline/`

### 5.1 已执行的实验步骤

#### 步骤 1: 数据准备（已执行）
```bash
bash scripts/prepare_cifar10.sh
```

#### 步骤 2: Baseline 训练（已全部执行）

我们按 batch size 递增顺序，分别训练 FP16 和 BF16 版本：

**实际执行的 8 组命令**（命名严格遵循 `resnet32_l40s_b{batch}_{fp16,bf16}`）：

```bash
# === Batch 128 ===
python train_resnet32.py --run-name resnet32_l40s_b128_fp16 --epochs 200 --batch-size 128 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b128_bf16  --epochs 200 --batch-size 128 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last

# === Batch 256 ===
python train_resnet32.py --run-name resnet32_l40s_b256_fp16 --epochs 200 --batch-size 256 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b256_bf16  --epochs 200 --batch-size 256 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last

# === Batch 512 ===
python train_resnet32.py --run-name resnet32_l40s_b512_fp16 --epochs 200 --batch-size 512 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b512_bf16  --epochs 200 --batch-size 512 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last

# === Batch 1024 ===
python train_resnet32.py --run-name resnet32_l40s_b1024_fp16 --epochs 200 --batch-size 1024 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b1024_bf16  --epochs 200 --batch-size 1024 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last
```

**执行结果**：全部 8 组已完成，对应 `runs/resnet32_l40s_b*/` 目录，结果汇总见 **第 2 节**。

#### 步骤 3: 最佳模型固化（已执行）
```bash
mkdir -p runs/resnet32_baseline
cp runs/resnet32_l40s_b128_bf16/best.pt      runs/resnet32_baseline/
cp runs/resnet32_l40s_b128_bf16/summary.json runs/resnet32_baseline/
cp runs/resnet32_l40s_b128_bf16/last.pt      runs/resnet32_baseline/
```

### 5.2 通用建议（本次实验采用）

- 所有实验均加上了 `--channels-last`
- `num-workers` 使用 8
- 每组训练后都保留了 `summary.json` 和 `metrics.csv`

### 5.3 额外可选实验（未来可做）

如果后续想进一步优化，可以考虑：
- `resnet32_l40s_b1024_bf16_lr08`（验证 Linear Scaling Rule）
- 加入 `--warmup-epochs 5`

**注意**：以上为未来可能的工作，不是本次已完成的实验。
  - 精度最高的 → 作为 KD Teacher + BN gamma 重要性来源
  - 速度与精度平衡最好的 → 可作为另一个对比点

---

**当前脚本提示**（可选）：
项目也提供了快速脚本（命名带 h100，但逻辑通用）：
```bash
bash scripts/run_h100_baseline_fast.sh   # 默认 1024+BF16
```
但**强烈建议优先手动跑上面的消融命令**，以获得完整对比数据。

---

### 3.3 Baseline 用途说明

最终选定的最佳 checkpoint（通常命名为 `runs/resnet32_baseline/best.pt` 或你自己重命名后复制）将用于：
- Block-level 搜索的 **BN gamma 重要性先验**
- 搜索阶段的 **sliced weight inheritance**（`--baseline-ckpt`）
- 最终 block-width 模型训练的 **KD Teacher**

---

## 6. Block-level 搜索（15 维，L40S 加速推荐）

快速一键搜索：
```bash
bash scripts/run_h100_block_search_fast.sh
```

等价手动命令（推荐配置）：
```bash
python search_block_channels_ga_pso.py \
  --algorithm both \
  --search-epochs 1 \
  --ga-population 8 \
  --ga-generations 5 \
  --pso-particles 8 \
  --pso-iterations 5 \
  --max-train-samples 10000 \
  --max-val-samples 5000 \
  --batch-size 1536 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp --amp-dtype bf16 --channels-last
```

搜索输出：
- `runs/block_channel_search_fast_*/`
- `best_result.json`
- `final_train_command.txt`（已自动生成带 KD 的推荐命令）

---

## 7. 训练最终 Block-level 模型（默认推荐 KD）

搜索完成后直接查看自动生成的命令：
```bash
cat runs/block_channel_search_fast_*/final_train_command.txt
```

典型带 KD 训练示例：
```bash
python train_block_width_resnet32.py \
  --block-channels 16,16,16,16,16,32,28,24,24,28,64,56,48,48,56 \
  --run-name final_block_kd \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --teacher-ckpt runs/resnet32_baseline/best.pt \
  --kd-mode logits \
  --kd-alpha 0.7 \
  --kd-temperature 4.0 \
  --amp --amp-dtype bf16 --channels-last
```

### 7.1 最新三次搜索得到的最佳配置（已完成 80-epoch KD 训练）

以下三个配置来自最近 block-level 搜索的 PSO 最优解，已全部用 KD 完成 80 epochs 训练。实际训练结果如下（数据来源于各 `summary.json`）：

```bash
# 配置1
python train_block_width_resnet32.py \
  --block-channels 12,8,8,8,8,16,16,16,20,16,48,40,32,32,64 \
  --run-name final_block_12-8-8-8-8-16-16-16-20-16-48-40-32-32-64 \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --teacher-ckpt runs/resnet32_baseline/best.pt \
  --kd-mode logits \
  --kd-alpha 0.7 \
  --kd-temperature 4.0 \
  --amp --amp-dtype bf16 --channels-last
```

**训练结果（来自 runs/final_block_12-8-8-8-8-16-16-16-20-16-48-40-32-32-64/summary.json）：**
- Best Acc: **89.38%** (epoch 73)
- Params: 187,990 (压缩率 59.5%)
- FLOPs: 24.07M (减少 65.0%)
- 训练时长: 214s
- KD: logits, α=0.7, T=4.0

```bash
# 配置2（该次搜索中 fitness 最高）
python train_block_width_resnet32.py \
  --block-channels 8,8,12,8,8,20,20,16,16,16,48,32,40,40,64 \
  --run-name final_block_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64 \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --teacher-ckpt runs/resnet32_baseline/best.pt \
  --kd-mode logits \
  --kd-alpha 0.7 \
  --kd-temperature 4.0 \
  --amp --amp-dtype bf16 --channels-last
```

**训练结果（来自 runs/final_block_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/summary.json）：**
- Best Acc: **88.63%** (epoch 70)
- Params: 201,314 (压缩率 56.6%)
- FLOPs: 24.85M (减少 63.9%)
- 训练时长: 245s
- KD: logits, α=0.7, T=4.0

```bash
# 配置3
python train_block_width_resnet32.py \
  --block-channels 12,8,8,8,8,20,20,16,16,16,48,40,32,40,64 \
  --run-name final_block_12-8-8-8-8-20-20-16-16-16-48-40-32-40-64 \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --teacher-ckpt runs/resnet32_baseline/best.pt \
  --kd-mode logits \
  --kd-alpha 0.7 \
  --kd-temperature 4.0 \
  --amp --amp-dtype bf16 --channels-last
```

**训练结果（来自 runs/final_block_12-8-8-8-8-20-20-16-16-16-48-40-32-40-64/summary.json）：**
- Best Acc: **88.80%** (epoch 74)
- Params: 202,438 (压缩率 56.4%)
- FLOPs: 25.44M (减少 63.1%)
- 训练时长: 243s
- KD: logits, α=0.7, T=4.0

## 8. 结果对比

### 通用命令模板

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/<你的最终训练目录>/summary.json \
  --output runs/final_comparison_<配置名>.json
```

**注意**：当前 `compare_results.py` 不支持 `--search-result` 参数（搜索结果已包含在 `best_result.json` 等文件中，可手动合并）。

### 8.1 本次三个新配置的对比结果（已生成）

三个最终模型的 80-epoch KD 训练已全部完成，对比 JSON 已通过 `compare_results.py` 生成在 `runs/` 目录下：

- `runs/final_comparison_12-8-8-8-8-16-16-16-20-16-48-40-32-32-64.json`
- `runs/final_comparison_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64.json`
- `runs/final_comparison_12-8-8-8-8-20-20-16-16-16-48-40-32-40-64.json`

**实际对比摘要**（来自生成的 JSON）：

| 配置 | Best Acc | Acc Drop | Params | 压缩率 | FLOPs | FLOPs减少 | 训练时长 |
|------|----------|----------|--------|--------|-------|-----------|----------|
| Baseline (b128+BF16) | 93.45% | - | 464,154 | - | 68.86M | - | 1811s |
| Config1 (12-8-...-64) | **89.38%** | 4.07% | 187,990 | 59.5% | 24.07M | 65.0% | 214s |
| Config2 (8-8-...-64) | 88.63% | 4.82% | 201,314 | 56.6% | 24.85M | 63.9% | 245s |
| Config3 (12-8-...-64) | 88.80% | 4.65% | 202,438 | 56.4% | 25.44M | 63.1% | 243s |

**已执行的对比命令**（供参考，实际已运行）：

**配置1：**
```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_12-8-8-8-8-16-16-16-20-16-48-40-32-32-64/summary.json \
  --output runs/final_comparison_12-8-8-8-8-16-16-16-20-16-48-40-32-32-64.json
```

**配置2（原搜索中 fitness 最高）：**
```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/summary.json \
  --output runs/final_comparison_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64.json
```

**配置3：**
```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_12-8-8-8-8-20-20-16-16-16-48-40-32-40-64/summary.json \
  --output runs/final_comparison_12-8-8-8-8-20-20-16-16-16-48-40-32-40-64.json
```

**说明：**
- 所有最终模型均使用相同 KD 设置（logits, α=0.7, T=4.0）和 baseline teacher。
- 精度损失约 4.1~4.8%，但参数量减少 ~56-60%，FLOPs 减少 ~63-65%，训练速度提升显著（~7-8x 更快）。
- 推荐 Config1 作为精度-压缩最佳平衡点（89.38%）。

---

## 9. 主要脚本与优化

| 脚本 / 模块                        | 说明 |
|------------------------------------|------|
| `prepare_cifar10.sh`               | 一键下载 + 解压 CIFAR-10（运行训练前必须执行） |
| `run_h100_baseline_fast.sh`        | L40S/H100 优化 baseline 训练（默认 1024+BF16） |
| `run_h100_block_search_fast.sh`    | L40S/H100 优化 15 维 block 搜索 |
| `search_block_channels_ga_pso.py`  | 核心搜索脚本（支持 importance + inheritance） |
| `train_block_width_resnet32.py`    | 支持 KD 的 block-width 训练 |
| `src/utils/accelerate.py`          | BF16 / channels-last / TF32 工具 |
| `src/utils/checkpoint.py`          | Sliced weight inheritance 实现 |

---

## 10. 配对关系说明

- **基础成熟版**：`resnet32_mature_ga_pso`（干净实现，适合理解算法）
- **L40S 优化成熟版**：`resnet32_l40s_mature_ga_pso`（本项目，推荐在 L40S 上使用，速度与稳定性更好）

两个版本算法逻辑一致，只是加速手段和默认配置不同。

---

## 11. 参考文档

- `docs/block_level_search.md` — 15 维 block-level 搜索详细说明
- `docs/experiment_plan.md` — 实验流程
- `docs/l40s_optimized_notes.md`（如存在）— L40S 优化细节

备份文件：`README.md.bak`

如需调整搜索空间、KD 参数或 batch size，请参考对应脚本的 `--help`。
