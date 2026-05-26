# ResNet32 L40S Mature GA/PSO

本项目是在 L40S 上完成的 ResNet32 CIFAR-10 压缩实验，核心流程是：

```text
ResNet32 baseline teacher
    -> BN gamma importance + 15 维 block-level GA/PSO 搜索
    -> sliced weight inheritance 快速候选评估
    -> block-width ResNet32 最终训练
    -> KD / no-KD / batch size 对照
```

当前实验已经全部跑完。最终推荐结果是：

- Teacher / baseline：`runs/resnet32_baseline/best.pt`
- 最佳压缩模型：`final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64`
- 最佳压缩精度：**90.20%**
- 相对 baseline 精度下降：**3.25%**
- 参数量：**201,314**，减少 **56.6%**
- FLOPs：**24.85M**，减少 **63.9%**

---

## 1. 环境与安装

```bash
pip install -r requirements.txt
bash scripts/prepare_cifar10.sh
```

推荐运行配置：

- CUDA GPU：L40S 或同级别显卡
- AMP：`--amp --amp-dtype bf16`
- 内存布局：`--channels-last`
- DataLoader：`--num-workers 8`

---

## 2. 代码结构

| 路径 | 说明 |
|---|---|
| `train_resnet32.py` | 训练标准 ResNet32 baseline |
| `train_block_width_resnet32.py` | 训练 15 维 block-width ResNet32，支持 KD |
| `search_block_channels_ga_pso.py` | block-level GA/PSO 通道搜索 |
| `compare_results.py` | baseline 与压缩模型指标对比 |
| `model_info.py` | 查看模型参数量 / FLOPs |
| `src/models/resnet32_cifar.py` | 标准 ResNet32 |
| `src/models/resnet32_block_width.py` | 每个 residual block 独立通道数的 ResNet32 |
| `src/utils/checkpoint.py` | sliced weight inheritance |
| `src/utils/kd.py` | logits KD / attention transfer 辅助逻辑 |
| `docs/` | 实验计划、搜索方法和优化说明 |
| `runs/` | 所有已完成实验输出 |

---

## 3. Baseline 消融结果

本项目先在 L40S 上完成了 `batch size x {FP16, BF16}` 共 8 组 200-epoch baseline 训练，用于确定 teacher 和 importance 来源。

| 配置 | Test Acc | Best Epoch | 训练时长 | 相对 b128_fp16 加速 | 结论 |
|---|---:|---:|---:|---:|---|
| `b128 + BF16` | **93.45%** | 193 | 1811s | 1.14x | 最佳 teacher |
| `b128 + FP16` | 93.10% | 120 | 2070s | 1.00x | 稳定基准 |
| `b256 + BF16` | 92.57% | 186 | 1678s | 1.23x | 速度/精度备选 |
| `b256 + FP16` | 92.38% | 139 | 1862s | 1.11x | - |
| `b512 + BF16` | 91.80% | 137 | 1281s | 1.62x | 精度掉点明显 |
| `b512 + FP16` | 91.71% | 158 | 1405s | 1.47x | - |
| `b1024 + BF16` | 90.95% | 122 | 1177s | 1.76x | 大 batch 精度下降 |
| `b1024 + FP16` | 91.01% | 146 | 1081s | 1.91x | 最快但精度损失最大 |

数据来源：`runs/resnet32_l40s_b*/summary.json`。

最终已将最佳模型固化为统一 baseline：

```bash
runs/resnet32_baseline/best.pt
runs/resnet32_baseline/last.pt
runs/resnet32_baseline/summary.json
```

该 baseline 同时用于：

- 搜索阶段 BN gamma importance prior
- 搜索阶段 sliced weight inheritance 的权重来源
- 最终压缩模型 KD teacher
- 所有最终对比的 baseline 参考

---

## 4. Baseline 复现实验命令

以下命令已经执行完毕，保留在 README 中用于复现。

```bash
python train_resnet32.py --run-name resnet32_l40s_b128_fp16  --epochs 200 --batch-size 128  --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b128_bf16  --epochs 200 --batch-size 128  --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b256_fp16  --epochs 200 --batch-size 256  --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b256_bf16  --epochs 200 --batch-size 256  --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b512_fp16  --epochs 200 --batch-size 512  --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b512_bf16  --epochs 200 --batch-size 512  --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b1024_fp16 --epochs 200 --batch-size 1024 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype fp16 --channels-last
python train_resnet32.py --run-name resnet32_l40s_b1024_bf16 --epochs 200 --batch-size 1024 --lr 0.1 --milestones 100,150 --num-workers 8 --amp --amp-dtype bf16 --channels-last
```

---

## 5. Block-Level 搜索

搜索使用 15 维 block-level 通道配置，每个 residual block 独立选择通道数。

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

也可以直接运行脚本：

```bash
bash scripts/run_h100_block_search_fast.sh
```

搜索输出位于：

```text
runs/block_channel_search_fast_*/
  best_result.json
  ga_best.json
  pso_best.json
  ga_history.csv
  pso_history.csv
  evaluations.csv
  final_train_command.txt
```

本轮最终挑选了 3 个候选结构进入完整训练：

| 简称 | Block channels | Params | FLOPs | Params 减少 | FLOPs 减少 |
|---|---|---:|---:|---:|---:|
| Config1 | `12,8,8,8,8,16,16,16,20,16,48,40,32,32,64` | 187,990 | 24.07M | 59.5% | 65.0% |
| Config2 | `8,8,12,8,8,20,20,16,16,16,48,32,40,40,64` | 201,314 | 24.85M | 56.6% | 63.9% |
| Config3 | `12,8,8,8,8,20,20,16,16,16,48,40,32,40,64` | 202,438 | 25.44M | 56.4% | 63.1% |

---

## 6. 最终训练设置

最终训练统一使用：

- `epochs=80`
- `milestones=40,60`
- `amp dtype=bf16`
- `channels-last`
- `num-workers=8`
- KD teacher：`runs/resnet32_baseline/best.pt`
- KD 设置：`--kd-mode logits --kd-alpha 0.7 --kd-temperature 4.0`

KD 示例：

```bash
python train_block_width_resnet32.py \
  --block-channels 8,8,12,8,8,20,20,16,16,16,48,32,40,40,64 \
  --run-name final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64 \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 128 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --teacher-ckpt runs/resnet32_baseline/best.pt \
  --kd-mode logits \
  --kd-alpha 0.7 \
  --kd-temperature 4.0 \
  --amp --amp-dtype bf16 --channels-last
```

无 KD 示例：

```bash
python train_block_width_resnet32.py \
  --block-channels 8,8,12,8,8,20,20,16,16,16,48,32,40,40,64 \
  --run-name final_block_nokd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64 \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 128 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp --amp-dtype bf16 --channels-last
```

---

## 7. 完整最终结果

Baseline：`resnet32_l40s_b128_bf16`，best acc **93.45%**，params **464,154**，FLOPs **68.86M**。

| Config | Batch | KD | Best Acc | Drop | Best Epoch | Time |
|---|---:|---|---:|---:|---:|---:|
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 128 | yes | 90.09% | 3.36% | 77 | 1264s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 128 | no | 89.58% | 3.87% | 75 | 1268s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 256 | yes | 89.68% | 3.77% | 76 | 1984s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 256 | no | 89.40% | 4.05% | 77 | 1493s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 512 | yes | 89.46% | 3.99% | 64 | 1751s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 512 | no | 88.33% | 5.12% | 69 | 1550s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 1024 | yes | 89.33% | 4.12% | 77 | 1579s |
| `12-8-8-8-8-16-16-16-20-16-48-40-32-32-64` | 1024 | no | 87.18% | 6.27% | 68 | 1332s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 128 | yes | 89.80% | 3.65% | 80 | 2389s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 128 | no | 89.65% | 3.80% | 76 | 1568s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 256 | yes | 89.46% | 3.99% | 71 | 2358s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 256 | no | 89.04% | 4.41% | 62 | 2120s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 512 | yes | 88.81% | 4.64% | 75 | 1803s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 512 | no | 88.35% | 5.10% | 69 | 1622s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 1024 | yes | 89.18% | 4.27% | 70 | 1584s |
| `12-8-8-8-8-20-20-16-16-16-48-40-32-40-64` | 1024 | no | 87.91% | 5.54% | 62 | 1336s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 128 | yes | **90.20%** | **3.25%** | 69 | 2224s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 128 | no | 89.06% | 4.39% | 78 | 1383s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 256 | yes | 89.71% | 3.74% | 72 | 2345s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 256 | no | 88.85% | 4.60% | 68 | 2112s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 512 | yes | 88.76% | 4.69% | 68 | 1796s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 512 | no | 86.65% | 6.80% | 60 | 1576s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 1024 | yes | 89.21% | 4.24% | 78 | 1585s |
| `8-8-12-8-8-20-20-16-16-16-48-32-40-40-64` | 1024 | no | 86.04% | 7.41% | 72 | 1328s |

说明：

- 表中 `Drop` 是相对 baseline 93.45% 的绝对精度下降。
- 上表使用完整 batch size x KD/no-KD 对照实验的 `final_block_kd_b*` 和 `final_block_nokd_b*` 目录。
- 早期还保留了 3 个 `final_block_<channels>/` 快速 KD 结果，用于生成第一版 comparison JSON；完整结论以上表为准。

---

## 8. 关键结论

1. **最佳模型是 Config2 + batch 128 + KD**：90.20%，比 baseline 低 3.25%，同时减少 56.6% 参数和 63.9% FLOPs。
2. **KD 明显有效**：同一结构和 batch 下，KD 通常带来 0.15% 到 3.17% 精度收益；收益在大 batch 下更明显。
3. **小 batch 更适合最终精度**：三个候选结构的最佳结果都出现在 batch 128 + KD。
4. **搜索阶段 fitness 不能完全替代完整训练**：Config2 的完整训练结果最好，说明最终选择仍应以 80-epoch 训练结果为准。
5. **大 batch 训练速度不一定更优**：本轮完整最终训练中，batch 1024 的墙钟时间没有稳定优于小 batch，可能受数据加载、GPU 利用率、运行环境波动影响；最终论文/报告建议主要引用精度、参数量和 FLOPs。

推荐论文/报告主表：

| Model | Acc | Acc Drop | Params | Params 减少 | FLOPs | FLOPs 减少 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet32 baseline | 93.45% | - | 464,154 | - | 68.86M | - |
| Config1 best | 90.09% | 3.36% | 187,990 | 59.5% | 24.07M | 65.0% |
| Config2 best | **90.20%** | **3.25%** | 201,314 | 56.6% | 24.85M | 63.9% |
| Config3 best | 89.80% | 3.65% | 202,438 | 56.4% | 25.44M | 63.1% |

---

## 9. 结果文件索引

Baseline：

```text
runs/resnet32_baseline/summary.json
runs/resnet32_baseline/best.pt
```

最佳压缩模型：

```text
runs/final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/summary.json
runs/final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/best.pt
```

完整最终训练结果：

```text
runs/final_block_kd_b{128,256,512,1024}_*/summary.json
runs/final_block_nokd_b{128,256,512,1024}_*/summary.json
```

早期 comparison JSON：

```text
runs/final_comparison_12-8-8-8-8-16-16-16-20-16-48-40-32-32-64.json
runs/final_comparison_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64.json
runs/final_comparison_12-8-8-8-8-20-20-16-16-16-48-40-32-40-64.json
```

搜索结果：

```text
runs/block_channel_search_fast_*/best_result.json
runs/block_channel_search_fast_*/evaluations.csv
```

---

## 10. 常用检查命令

查看最佳模型 summary：

```bash
cat runs/final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/summary.json
```

重新生成 baseline 与压缩模型对比：

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/summary.json \
  --output runs/final_comparison_best_kd_b128_config2.json
```

评估 checkpoint：

```bash
python evaluate_block_width_resnet32.py \
  --checkpoint runs/final_block_kd_b128_8-8-12-8-8-20-20-16-16-16-48-32-40-40-64/best.pt \
  --block-channels 8,8,12,8,8,20,20,16,16,16,48,32,40,40,64 \
  --batch-size 512 \
  --num-workers 8
```

---

## 11. 参考文档

- `docs/block_level_search.md`：15 维 block-level 搜索说明
- `docs/experiment_plan.md`：实验流程
- `docs/optimization_model.md`：优化目标建模
- `docs/next_ga_pso_plan.md`：GA/PSO 后续计划记录
- `docs/h100_fast_notes.md`：高端 GPU 快速运行说明

备份文件：`README.md.bak`
