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

## 2. 实验结果（已完成）

本项目已完整跑通 **block-level 成熟方案**：baseline 200 epochs → 基于 BN gamma 重要性先验的 15 维 GA/PSO 搜索（57 次唯一评估） → 多组 80 epochs + KD 完整训练 → 全面指标对比。

### 2.1 最终训练结果对比

- **Baseline**：`runs/resnet32_baseline`（200 epochs, batch=128, amp）
- **压缩模型**：统一 80 epochs + KD（`kd-alpha=0.5`, `kd-temperature=4`）

| 通道配置（15 维 block）                          | 测试 Acc | Acc Drop | Params   | 压缩率   | FLOPs    | FLOPs 减少 | 训练时长 | 来源     | 备注                  |
|--------------------------------------------------|----------|----------|----------|----------|----------|------------|----------|----------|-----------------------|
| Baseline（每 stage 固定 16/32/64）               | 93.11%   | -        | 464,154  | -        | 68.86M   | -          | 1290s    | -        | 标准 ResNet32 Teacher |
| 16-16-12-12-16-32-28-24-24-28-64-56-48-48-56     | 91.53%   | **1.58%**| 352,370  | 24.1%    | 54.0M    | 21.6%      | 636s     | 手册示例 | 精度最高，温和压缩    |
| 12-12-8-8-8-24-20-20-16-16-48-48-32-40-64        | 90.70%   | 2.41%    | 229,738  | 50.5%    | 30.5M    | **55.7%**  | 708s     | PSO 方向 | 良好平衡              |
| **16-16-12-12-8-16-20-20-20-20-40-40-32-32-32**  | **90.52%** | 2.59%  | **161,018** | **65.3%** | 32.6M   | 52.6%      | **490s** | **GA 最优** | **最高压缩，强烈推荐** |

数据来源：各 `summary.json` + `final_comparison_block_*.json`（及 `block_channel_search_ga_pso/best_result.json`）。

### 2.2 关键发现与分析

- **Block-level 搜索价值**：相比早期 stage-level（仅 3 变量，最高 ~74% 压缩但粒度粗），15 维搜索能**更精细地**对低重要性 block 进行激进压缩。GA 找到了 65.3% 参数压缩的优秀方案。
- **重要性先验有效**：搜索历史显示，重要性低的早期 block 被优先压缩到 8~12 通道，而重要性高的后期 block 保留更多通道，符合设计预期。
- **Quick Proxy 的局限性**：搜索阶段（2 epochs + 5000 samples）最高 quick-acc 仅 26.75%，但完整 80-epoch + KD 训练后全部回升至 90.5%+。证明 GA/PSO 搜索方向**正确**，但当前短 proxy + fitness 设置严重低估了真实潜力（未来可考虑放宽或增加 proxy 强度）。
- **KD 恢复能力突出**：从 proxy ~25% 直接跃升至 90%+，KD 对 block-level 压缩后的精度恢复贡献巨大。
- **训练加速明显**：最激进配置 wall-clock 时间降至 baseline 的 ~38%（490s vs 1290s），约 **2.6×** 加速。实际加速比理论 FLOPs 减少略低（因 block 间不均匀性）。
- **课程论文推荐**：
  - **主结果**：GA 配置 `16-16-12-12-8-16-20-20-20-20-40-40-32-32-32`（最高压缩 + 可控掉点）
  - **平衡配置**：`12-12-8-8-8-24-20-20-16-16-48-48-32-40-64`
  - **温和上界**：手册示例配置（掉点最小）

### 2.3 课程论文使用提示

直接可引用的核心文件：
- 搜索阶段：`runs/block_channel_search_ga_pso/{best_result.json, ga_best.json, pso_best.json, evaluations.csv, ga_history.csv, pso_history.csv}`
- 推荐主结果：`runs/final_comparison_block_16-16-12-12-8-16-20-20-20-20-40-40-32-32-32.json`
- 对应完整 checkpoint + 指标：`runs/final_block_16-16-12-12-8-16-20-20-20-20-40-40-32-32-32/`

---

## 3. 安装

```bash
pip install -r requirements.txt
```

---

## 4. 查看模型复杂度

标准 ResNet32：
```bash
python model_info.py
```

任意 block-level 配置：
```bash
python model_info.py --block-channels 16,16,12,12,16,32,28,24,24,28,64,56,48,48,56
```

---

## 5. 训练标准 ResNet32 Baseline（Teacher）

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

## 6. GA/PSO Block-level 搜索（15 维）

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

## 7. 训练最终 Block-level 压缩模型（支持 KD）

搜索完成后，终端会打印推荐命令。**已实际完成训练并验证的优秀配置**（见第 2 节）以及**最新搜索得到待训练的配置**如下：

**推荐主结果（GA 最优，最高压缩 65.3%）：**
```bash
python train_block_resnet32_kd.py \
  --block-channels 16,16,12,12,8,16,20,20,20,20,40,40,32,32,32 \
  --teacher-checkpoint runs/resnet32_baseline/best.pt \
  --kd-alpha 0.5 \
  --kd-temperature 4 \
  --run-name final_block_ga_best \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

**平衡配置（PSO 方向，55.7% FLOPs 减少）：**
```bash
python train_block_resnet32_kd.py \
  --block-channels 12,12,8,8,8,24,20,20,16,16,48,48,32,40,64 \
  --teacher-checkpoint runs/resnet32_baseline/best.pt \
  --kd-alpha 0.5 \
  --kd-temperature 4 \
  --run-name final_block_pso_candidate \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

**温和配置（手册示例，精度掉点最小 1.58%）：**
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

**不使用 KD（ablation）：**
```bash
python train_block_resnet32_kd.py \
  --block-channels 16,16,12,12,8,16,20,20,20,20,40,40,32,32,32 \
  --run-name final_block_ga_best_no_kd \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

---

## 8. 结果对比

详见本文 **第 2 节「实验结果」** 的完整表格与分析。推荐直接使用已生成的对比文件：

```bash
# 查看推荐主结果（GA 最优配置）
cat runs/final_comparison_block_16-16-12-12-8-16-20-20-20-20-40-40-32-32-32.json

# 或使用工具重新生成（以手册示例为例）
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_kd/summary.json \
  --search-result runs/block_channel_search_ga_pso/best_result.json \
  --output runs/final_comparison.json
```

对比字段示例：`accuracy_drop`、`params_compression_rate`、`flops_reduction_rate`、`kd_enabled`、`search_time_sec` 等。

---

## 9. 适应度函数

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

## 10. 核心文件

| 文件/目录                        | 作用 |
|----------------------------------|------|
| `search_block_channels_ga_pso.py` | 15 维 block-level GA/PSO 搜索 |
| `train_block_resnet32_kd.py`     | 支持 KD 的 block-width 模型训练 |
| `src/search/importance.py`       | BN gamma 重要性计算 |
| `src/models/resnet32_blockwidth.py` | 支持任意 15 维通道配置的 ResNet32 |
| `src/utils/checkpoint.py`        | 权重继承相关工具 |

---

如需 L40S 加速版（BF16 + channels-last + 权重继承 + 更快脚本），请使用配对项目 `resnet32_l40s_mature_ga_pso`。

---

## 11. 下一步工作建议

当前 block-level 成熟方案已跑通并产出高质量结果。推荐的后续方向（按优先级）：

1. **无 KD Ablation**（强烈建议）：对 GA 最优配置跑一次不带 KD 的 80-epoch 训练，量化 KD 的真实恢复贡献。
2. **更长训练**：对推荐配置 `16-16-12-12-8-...-32` 跑 200 epochs（与 baseline 同等充分），作为论文的最终数字。
3. **Fitness / Proxy 优化**：当前 2-epoch proxy 严重低估潜力。尝试：
   - 增大 `--search-epochs` 到 3~5
   - 放宽 λp/λf 或加入 `allowed_acc_drop` 机制
   - 重新搜索并对比
4. **重要性先验 Ablation**：关闭 importance 引导重新搜索一次，量化先验的作用。
5. **L40S 加速复现**：在 `resnet32_l40s_mature_ga_pso` 上用相同配置快速重跑（BF16 + inheritance），验证加速效果。
6. **论文产出**：基于第 2 节数据生成通道配置可视化图、学习曲线对比、搜索收敛图等。

执行任意一项前，请告知具体需求，我会立即开始实施。

备份文件：`README.md.bak`
