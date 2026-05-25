# ResNet32 成熟版：重要性引导 GA/PSO Block-level 通道搜索 + KD

这个版本是在干净 ResNet32 baseline 上升级的可实现成熟方案：

```text
标准 ResNet32 baseline
→ 读取 baseline 的 BN gamma 作为 block 重要性先验
→ GA / PSO 搜索 15 个 residual block 的通道配置
→ 训练搜索得到的最优 block-width ResNet32
→ 可选使用 baseline 作为 teacher 做知识蒸馏恢复精度
→ 对比 Accuracy、Params、FLOPs、运行时间
```

不是量化，不是 NAS，不是全家桶。重点是把课程任务里的“计算智能方法求解模型压缩最优化问题”做扎实。

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

某个 block-level 配置：

```bash
python model_info.py --block-channels 16,16,12,12,16,32,28,24,24,28,64,56,48,48,56
```

---

## 3. 训练标准 ResNet32 baseline

快速冒烟测试：

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
runs/resnet32_baseline/summary.json
runs/resnet32_baseline/metrics.csv
```

---

## 4. GA/PSO 搜索 15 维 block 通道配置

搜索变量：

```text
x = [c1_1,c1_2,c1_3,c1_4,c1_5,
     c2_1,c2_2,c2_3,c2_4,c2_5,
     c3_1,c3_2,c3_3,c3_4,c3_5]
```

默认搜索空间：

```text
stage1 每个 block: {8, 12, 16}
stage2 每个 block: {16, 20, 24, 28, 32}
stage3 每个 block: {32, 40, 48, 56, 64}
```

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

正式一点：

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

输出：

```text
runs/block_channel_search_ga_pso/evaluations.csv
runs/block_channel_search_ga_pso/ga_history.csv
runs/block_channel_search_ga_pso/pso_history.csv
runs/block_channel_search_ga_pso/ga_best.json
runs/block_channel_search_ga_pso/pso_best.json
runs/block_channel_search_ga_pso/best_result.json
```

---

## 5. 训练最终压缩模型，并可选 KD

搜索完成后终端会打印推荐命令。手动示例：

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

不用 KD：

```bash
python train_block_resnet32_kd.py \
  --block-channels 16,16,12,12,16,32,28,24,24,28,64,56,48,48,56 \
  --run-name final_block_no_kd \
  --epochs 80 \
  --milestones 40,60
```

---

## 6. 对比 baseline 与压缩模型

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_kd/summary.json \
  --search-result runs/block_channel_search_ga_pso/best_result.json \
  --output runs/final_comparison.json
```

输出指标可直接放报告：

```text
Accuracy Drop
Params Compression Rate
FLOPs Reduction Rate
Search Time
Training Time
KD Enabled
```

---

## 7. 适应度函数

```text
Fitness(x) = Acc(x)
           - 100 * [λp * Params(x)/Params0 + λf * FLOPs(x)/FLOPs0]
           - λt * TimePenalty(x)
```

默认：

```text
λp = 0.10
λf = 0.15
λt = 0.02
```

调整建议：

```text
更重视精度：减小 λp / λf
更重视压缩率：增大 λp / λf
机器慢：减小 population / iterations / search-epochs
```
