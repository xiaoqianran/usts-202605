# ResNet32 CIFAR-10：H100 快速版 + 15维 GA/PSO Block-Level 通道搜索

这版已经从旧的 3 维 stage-level 搜索：

```text
x = [c1, c2, c3]
```

升级为新版 15 维 block-level 搜索：

```text
x = [
  c1_1, c1_2, c1_3, c1_4, c1_5,
  c2_1, c2_2, c2_3, c2_4, c2_5,
  c3_1, c3_2, c3_3, c3_4, c3_5
]
```

每个变量对应 ResNet32 中一个 residual block 的输出通道数。网络深度仍然是 ResNet32，不改变 block 数量。

本包是纯代码包，不包含：

```text
data/
runs/
*.pt
checkpoint
CIFAR-10 数据
```

---

## 安装

```bash
pip install -r requirements.txt
```

项目默认读取：

```text
data/cifar-10-batches-py/
```

如果你原项目里已经有 `data/` 和 `runs/`，把这个代码包覆盖过去即可，不要删除原来的数据和 checkpoint。

---

## 1. 训练标准 ResNet32 baseline

```bash
bash scripts/run_h100_baseline_fast.sh
```

或手动运行：

```bash
python train_resnet32.py \
  --run-name resnet32_baseline_bf16 \
  --epochs 200 \
  --batch-size 1024 \
  --lr 0.1 \
  --milestones 100,150 \
  --num-workers 8 \
  --amp \
  --amp-dtype bf16 \
  --channels-last
```

输出：

```text
runs/resnet32_baseline/best.pt
runs/resnet32_baseline/summary.json
```

---

## 2. 运行新版 15 维 GA/PSO 搜索

```bash
bash scripts/run_h100_search_fast.sh
```

等价于：

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
  --batch-size 2048 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp \
  --amp-dtype bf16 \
  --channels-last
```

搜索输出在：

```text
runs/block_channel_search_fast_*/
```

里面包括：

```text
evaluations.csv
ga_history.csv
pso_history.csv
ga_best.json
pso_best.json
best_result.json
final_train_command.txt
```

## 2.2 重新search
python train_block_width_resnet32.py \
  --block-channels 16,16,16,16,16,32,28,28,20,20,56,56,56,48,56 \
  --run-name final_block_pso_16-16-16-16-16-32-28-28-20-20-56-56-56-48-56 \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --amp \
  --amp-dtype bf16 \
  --channels-last \
  --baseline-ckpt runs/resnet32_baseline/best.pt



---

## 2.2 完整训练搜索得到的最优 block-level 压缩模型

搜索结束后直接查看：

```bash
cat runs/block_channel_search_fast_*/final_train_command.txt
```

复制其中命令运行即可。

也可以手动训练一个 15 维配置，例如：

python train_block_width_resnet32.py --block-channels 16,16,16,16,16,32,28,28,20,20,56,56,56,48,56 --run-name final_block_pso_16-16-16-16-16-32-28-28-20-20-56-56-56-48-56 --epochs 80 --milestones 40,60 --batch-size 1024 --num-workers 8 --amp --amp-dtype bf16 --channels-last --baseline-ckpt runs/resnet32_baseline/best.pt

```bash
python train_block_width_resnet32.py \
  --block-channels 16,16,16,16,16,32,28,24,24,28,64,56,48,48,56 \
  --run-name final_block_example \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp \
  --amp-dtype bf16 \
  --channels-last
```

也支持输入 3 个 stage 通道，自动扩展为 15 维：

```bash
python train_block_width_resnet32.py --block-channels 16,24,48 ...
```

会自动变为：

```text
[16,16,16,16,16, 24,24,24,24,24, 48,48,48,48,48]
```

---

## 3. 对比 baseline 和最终压缩模型

```bash
python compare_results.py \
  --baseline runs/resnet32_baseline/summary.json \
  --compressed runs/final_block_example/summary.json \
  --output runs/final_block_comparison.json
```

输出包括：

```text
accuracy_drop
params_compression_rate
flops_reduction_rate
baseline_train_time_sec
compressed_train_time_sec
compressed_block_channels
```

---

## 4. 查看某个 15 维配置的 Params/FLOPs

```bash
python model_info.py \
  --block-channels 16,16,16,16,16,32,28,24,24,28,64,56,48,48,56
```

baseline 等价配置：

```bash
python model_info.py --block-channels 16,32,64
```

会自动扩展为：

```text
[16,16,16,16,16, 32,32,32,32,32, 64,64,64,64,64]
```

---

## 5. 新增/保留的主要文件

```text
search_block_channels_ga_pso.py       # 新版 15维 GA/PSO 搜索
train_block_width_resnet32.py         # 新版 15维压缩模型完整训练
evaluate_block_width_resnet32.py      # 新版 15维模型评估
src/models/resnet32_block_width.py    # BlockWidthResNet32 模型
src/data/cifar10.py                   # 本地 CIFAR-10 读取，不强依赖 torchvision
scripts/run_h100_search_fast.sh       # 默认已切换到新版 15维搜索
```

旧的 3 维 stage-level 脚本仍保留：

```text
search_channels_ga_pso.py
train_width_resnet32.py
```

用于对照实验。
