已经按“成熟但可实现”的方案重做成代码包了：

[下载 resnet32_mature_ga_pso.zip](sandbox:/mnt/data/resnet32_mature_ga_pso.zip)

这版核心升级是：

```text
标准 ResNet32 baseline
→ 读取 baseline 的 BN gamma 作为 block 重要性先验
→ GA / PSO 搜索 15 个 residual block 的通道配置
→ 训练搜索得到的最优 block-width ResNet32
→ 可选用 baseline teacher 做 KD 蒸馏恢复精度
→ 输出 Accuracy / Params / FLOPs / Search Time / Training Time 对比
```

我做了语法检查，并实际跑了模型复杂度统计。标准 ResNet32 结果：

```text
Params: 464.154K
FLOPs : 68.863M
```

示例压缩配置：

```text
16,16,12,12,16,32,28,24,24,28,64,56,48,48,56
```

统计结果：

```text
Params: 352.370K
FLOPs : 53.986M
```

也就是说，这种 block-level 配置已经能明显压缩，而不是之前那种粗糙的 3 维 stage 搜索。

## 主要文件

```text
train_resnet32.py                 # 训练标准 ResNet32 baseline
evaluate_resnet32.py              # 评估 baseline
model_info.py                     # 查看 Params / FLOPs

search_block_channels_ga_pso.py   # 重点：GA/PSO 15维 block 通道搜索
train_block_resnet32_kd.py        # 训练最终压缩模型，可选 KD 蒸馏
compare_results.py                # baseline vs 压缩模型对比

src/models/resnet32_cifar.py      # 标准 ResNet32
src/models/resnet32_blockwidth.py # block-level 可变通道 ResNet32
src/search/importance.py          # BN gamma 重要性引导
docs/方案说明.md
```

## 推荐运行顺序

## 1. 先训练 baseline：

```bash
python train_resnet32.py \
  --run-name resnet32_baseline \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --milestones 100,150 \
  --amp
```


## 2 然后快速搜索，先确认流程：

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

## 正式一点：

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

## 3 搜索结束后，它会打印最终训练命令。形式类似：

【这个是我们找到的】
```bash 
python train_block_resnet32_kd.py \
  --block-channels 12,12,8,8,8,24,20,20,16,16,48,48,32,40,64 \
  --teacher-checkpoint runs/resnet32_baseline/best.pt \
  --run-name final_block_12-12-8-8-8-24-20-20-16-16-48-48-32-40-64 \
  --epochs 80 \
  --milestones 40,60 \
  --amp
```

python train_block_resnet32_kd.py \
  --block-channels 16,16,12,12,8,16,20,20,20,20,40,40,32,32,32 \
  --teacher-checkpoint runs/resnet32_baseline/best.pt \
  --run-name final_block_16-16-12-12-8-16-20-20-20-20-40-40-32-32-32 \
  --epochs 80 \
  --milestones 40,60 \
  --amp


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

## 4 最后对比：

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_12-12-8-8-8-24-20-20-16-16-48-48-32-40-64/summary.json \
  --search-result runs/block_channel_search_ga_pso/best_result.json \
  --output runs/final_comparison_block_12-12-8-8-8-24-20-20-16-16-48-48-32-40-64.json

python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_16-16-12-12-8-16-20-20-20-20-40-40-32-32-32/summary.json \
  --search-result runs/block_channel_search_ga_pso/best_result.json \
  --output runs/final_comparison_block_16-16-12-12-8-16-20-20-20-20-40-40-32-32-32.json

python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_kd/summary.json \
  --search-result runs/block_channel_search_ga_pso/best_result.json \
  --output runs/final_comparison_block_kd.json
```

## 这版比上一版强在哪

上一版：

```text
x = [c1, c2, c3]
```

这版：

```text
x = [
  c1_1,c1_2,c1_3,c1_4,c1_5,
  c2_1,c2_2,c2_3,c2_4,c2_5,
  c3_1,c3_2,c3_3,c3_4,c3_5
]
```

从 3 维搜索升级到 15 维搜索，而且加入了 baseline 的 BN gamma 重要性引导。
这就更像“成熟的结构化通道压缩”，不是随便改宽度。

我没有把量化加进去。这个判断是刻意的：现在最重要的是把 **GA/PSO + ResNet32 压缩 + 结果对比** 做稳定，别把任务复杂度炸穿。
