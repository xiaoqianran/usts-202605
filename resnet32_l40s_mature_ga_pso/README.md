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

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. 核心优化特性（L40S 版）

本版本相比基础 mature 版主要增强：

- `--amp --amp-dtype bf16`（默认推荐，数值稳定性更好）
- `--channels-last` 内存布局加速
- `setup_torch_fast`（TF32 + cudnn benchmark）
- 搜索阶段 **sliced weight inheritance**（`--baseline-ckpt`）
- 更高效的 DataLoader 配置（高 num_workers + persistent workers）
- 快速一键脚本（`run_l40s_*_fast.sh`）

---

## 3. 训练 Baseline / Teacher（推荐 L40S 配置）

```bash
bash scripts/run_l40s_baseline_fast.sh
```

或手动（BF16 + channels-last）：
```bash
python train_resnet32.py \
  --run-name resnet32_baseline \
  --epochs 200 \
  --batch-size 1024 \
  --lr 0.1 \
  --milestones 100,150 \
  --num-workers 8 \
  --amp --amp-dtype bf16 --channels-last
```

输出 `runs/resnet32_baseline/best.pt` 将用于：
- 搜索的重要性先验
- 搜索阶段权重继承
- 最终 KD 的 teacher

---

## 4. Block-level 搜索（15 维，L40S 加速推荐）

快速一键搜索：
```bash
bash scripts/run_l40s_block_search_fast.sh
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

## 5. 训练最终 Block-level 模型（默认推荐 KD）

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

---

## 6. 结果对比

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_kd/summary.json \
  --search-result runs/block_channel_search_fast_*/best_result.json \
  --output runs/final_comparison.json
```

---

## 7. 主要脚本与优化

| 脚本 / 模块                        | 说明 |
|------------------------------------|------|
| `run_l40s_baseline_fast.sh`        | L40S 优化 baseline 训练 |
| `run_l40s_block_search_fast.sh`    | L40S 优化 15 维 block 搜索 |
| `search_block_channels_ga_pso.py`  | 核心搜索脚本（支持 importance + inheritance） |
| `train_block_width_resnet32.py`    | 支持 KD 的 block-width 训练 |
| `src/utils/accelerate.py`          | BF16 / channels-last / TF32 工具 |
| `src/utils/checkpoint.py`          | Sliced weight inheritance 实现 |

---

## 8. 配对关系说明

- **基础成熟版**：`resnet32_mature_ga_pso`（干净实现，适合理解算法）
- **L40S 优化成熟版**：`resnet32_l40s_mature_ga_pso`（本项目，推荐在 L40S 上使用，速度与稳定性更好）

两个版本算法逻辑一致，只是加速手段和默认配置不同。

---

## 9. 参考文档

- `docs/block_level_search.md` — 15 维 block-level 搜索详细说明
- `docs/experiment_plan.md` — 实验流程
- `docs/l40s_optimized_notes.md`（如存在）— L40S 优化细节

备份文件：`README.md.bak`

如需调整搜索空间、KD 参数或 batch size，请参考对应脚本的 `--help`。
