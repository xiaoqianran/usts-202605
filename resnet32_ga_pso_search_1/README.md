# ResNet32 CIFAR-10 + GA/PSO 通道配置搜索

本项目用于《智能信息技术与应用》课程论文：

1. 先训练标准 CIFAR-10 ResNet32 baseline。
2. 在不改变 ResNet32 深度结构的前提下，搜索三个 stage 的通道配置 `[c1, c2, c3]`。
3. 使用 GA（遗传算法）和 PSO（粒子群优化）求解通道配置优化问题。
4. 完整比较压缩前后 Accuracy、Params、FLOPs、运行时间。

> 注意：本阶段是 **stage-level 结构化通道宽度压缩**。它不是 BN-gamma 真通道索引剪枝。目的：先把“baseline → 智能优化搜索 → 最优压缩模型 → 指标对比”的闭环跑通。

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. 查看标准 ResNet32 参数量/FLOPs

```bash
python model_info.py
```

查看某个压缩通道配置：

```bash
python model_info.py --channels 16,24,48
```

---

## 3. 训练标准 ResNet32 baseline

快速测试：

```bash
python train_resnet32.py --epochs 2 --batch-size 128 --lr 0.05 --milestones 1 --run-name resnet32_quick
```

正式训练：

```bash
python train_resnet32.py \
  --run-name resnet32_baseline \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --milestones 100,150 \
  --amp
```

输出：

```text
runs/resnet32_baseline/best.pt
runs/resnet32_baseline/last.pt
runs/resnet32_baseline/metrics.csv
runs/resnet32_baseline/summary.json
```

---

## 4. GA/PSO 搜索通道配置

默认搜索空间：

```text
c1 ∈ {8, 12, 16}
c2 ∈ {16, 20, 24, 28, 32}
c3 ∈ {32, 40, 48, 56, 64}
```

即仍然是 ResNet32：

```text
conv1 + stage1(5 blocks) + stage2(5 blocks) + stage3(5 blocks) + GAP + FC
```

只是 stage 通道数从原始 `[16, 32, 64]` 变成搜索得到的 `[c1, c2, c3]`。

快速搜索：

```bash
python search_channels_ga_pso.py \
  --algorithm both \
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
python search_channels_ga_pso.py \
  --algorithm both \
  --search-epochs 3 \
  --ga-population 8 \
  --ga-generations 5 \
  --pso-particles 8 \
  --pso-iterations 5 \
  --max-train-samples 5000 \
  --max-test-samples 2000 \
  --amp
```

输出：

```text
runs/channel_search_ga_pso/search_config.json
runs/channel_search_ga_pso/evaluations.csv
runs/channel_search_ga_pso/ga_history.csv
runs/channel_search_ga_pso/pso_history.csv
runs/channel_search_ga_pso/ga_best.json
runs/channel_search_ga_pso/pso_best.json
runs/channel_search_ga_pso/best_result.json
```

---

## 5. 训练搜索得到的最优压缩模型

搜索完成后，终端会打印推荐命令。也可以手动运行：

```bash
python train_width_resnet32.py \
  --channels 16,24,48 \
  --run-name final_width_16-24-48 \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

输出：

```text
runs/final_width_16-24-48/best.pt
runs/final_width_16-24-48/metrics.csv
runs/final_width_16-24-48/summary.json
```

---

## 6. 对比 baseline 和压缩模型

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_width_16-24-48/summary.json \
  --output runs/final_comparison.json
```

输出字段包括：

```text
baseline_best_acc
compressed_best_acc
accuracy_drop
params_compression_rate
flops_reduction_rate
baseline_train_time_sec
compressed_train_time_sec
```

这几个指标可以直接放进课程论文结果表。

---

## 7. 适应度函数

搜索阶段的适应度函数：

```text
Fitness(x) = Acc(x) - 100 * [ λp * Params(x)/Params0 + λf * FLOPs(x)/FLOPs0 ]
```

其中：

```text
x = [c1, c2, c3]
Acc(x)：候选模型短训练后的验证准确率
Params0 / FLOPs0：原始 ResNet32 的参数量和 FLOPs
λp / λf：参数量和计算量惩罚权重
```

默认：

```text
λp = 0.15
λf = 0.15
```

如果你更重视精度，减小 λ；如果你更重视压缩率，增大 λ。

---

## 8. 文件结构

```text
train_resnet32.py             # 标准 ResNet32 baseline 训练
evaluate_resnet32.py          # 标准 ResNet32 checkpoint 评估
model_info.py                 # 查看标准/压缩配置的 Params 与 FLOPs
search_channels_ga_pso.py     # GA/PSO 搜索通道配置
train_width_resnet32.py       # 训练搜索得到的压缩 ResNet32
evaluate_width_resnet32.py    # 评估压缩 ResNet32 checkpoint
compare_results.py            # baseline vs compressed 指标对比

src/models/resnet32_cifar.py  # 标准 ResNet32
src/models/resnet32_width.py  # 可变通道 ResNet32，深度仍为 32
src/data/cifar10.py           # CIFAR-10 dataloader，支持子集搜索
src/utils/                    # seed/checkpoint/metrics

docs/optimization_model.md    # 优化问题建模说明
docs/experiment_plan.md       # 实验流程说明
```
