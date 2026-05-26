# BF16 与 FP16 混合精度实验差异分析

本文用于解释 `resnet32_l40s_optimized_ga_pso_search` 与 `resnet32_l40s_mature_ga_pso` 两个目录中 FP16/BF16 baseline 结果方向不一致，以及最终压缩通道配置不一致的原因。

## 1. 结论摘要

两个现象需要分开解释：

1. Baseline 中 BF16/FP16 谁更高，发生在标准 ResNet32 `[16, 32, 64]` 上，和压缩通道搜索无关。
2. 最终压缩通道数不同，主要来自两个项目的搜索空间和模型参数化不同，不是由 BF16/FP16 直接决定。

更准确的结论是：

- `resnet32_l40s_mature_ga_pso` 的 baseline 消融中，`batch=128 + BF16` 最好，Best Acc = 93.45%。
- `resnet32_l40s_optimized_ga_pso_search` 的新 baseline 消融中，`batch=128 + FP16` 最好，Best Acc = 93.30%。
- 这两个结果来自非确定性训练设置下的两轮独立运行，不能直接推出 “BF16 一定优于 FP16” 或 “FP16 一定优于 BF16”。
- 后续论文或报告中应表述为：在各自实验闭环内选择最高精度 baseline，而不泛化混合精度类型的绝对优劣。

## 2. Baseline 实验结果对比

两个目录的 baseline 都是标准 ResNet32，通道为 `[16, 32, 64]`，不是压缩模型。因此 baseline 的 FP16/BF16 差异不能用压缩通道数解释。

| Batch | mature FP16 | mature BF16 | optimized FP16 | optimized BF16 |
|---:|---:|---:|---:|---:|
| 128 | 93.10% | **93.45%** | **93.30%** | 93.00% |
| 256 | 92.38% | **92.57%** | **92.81%** | 92.48% |
| 512 | 91.71% | **91.80%** | **91.97%** | 91.62% |
| 1024 | **91.01%** | 90.95% | 91.01% | **91.01%** |

可以看到，`mature` 也不是所有 batch 下 BF16 都高于 FP16；`batch=1024` 时 FP16 略高。`optimized` 中 FP16 整体更高，但差距同样不大。

## 3. 真实运行配置是否一致

从 checkpoint 的 `config` 字段检查，两个目录的 `b128_fp16` 和 `b128_bf16` baseline 训练参数是一致的：

- `epochs=200`
- `batch_size=128`
- `lr=0.1`
- `milestones=[100,150]`
- `momentum=0.9`
- `weight_decay=1e-4`
- `num_workers=8`
- `seed=42`
- `amp=True`
- `channels_last=True`
- `compile=False`
- 模型均为标准 ResNet32，`stage_channels=[16,32,64]`

因此，结果反转不是因为显式超参数不同。

## 4. 为什么同配置仍会反转

代码中训练并不是严格确定性的：

- `set_seed(args.seed, deterministic=False)`
- `torch.backends.cudnn.benchmark = True`
- DataLoader 使用 `num_workers=8`
- 训练增强包含随机 crop 和随机 horizontal flip
- FP16 与 BF16 会触发不同的 CUDA/cuDNN kernel
- FP16 使用 GradScaler，BF16 不使用 GradScaler

这些因素会让同一 seed 下的训练轨迹也不能保证 bitwise 一致。实际曲线也证明两轮运行从第 1 个 epoch 就已经不同。

例如 `batch=128 + FP16`：

| 项目 | Epoch 1 Test Acc | Best Epoch | Best Acc |
|---|---:|---:|---:|
| mature | 43.98% | 120 | 93.10% |
| optimized | 41.57% | 193 | 93.30% |

例如 `batch=128 + BF16`：

| 项目 | Epoch 1 Test Acc | Best Epoch | Best Acc |
|---|---:|---:|---:|
| mature | 45.87% | 193 | 93.45% |
| optimized | 46.41% | 175 | 93.00% |

这说明两边虽然配置相同，但训练过程没有严格复现。最终 `0.3% ~ 0.45%` 量级的差异，属于这种非确定性训练设置下可能出现的波动范围。

## 5. FP16 与 BF16 的数值差异

FP16 和 BF16 都是混合精度训练，但数值特性不同：

- FP16 尾数精度更高，但指数范围较小，训练时通常需要 GradScaler 降低 overflow 风险。
- BF16 指数范围更接近 FP32，通常不需要 GradScaler，但尾数精度低于 FP16。

在 ResNet32 + CIFAR-10 这种规模较小的任务中，两者的优势并没有绝对方向。FP16 经过 GradScaler 后可能获得更细的数值分辨率；BF16 可能在某些运行中更稳定。最终谁略高，容易受初始化、增强随机性、kernel 选择和训练噪声影响。

因此，这里的实验结论不应写成 “BF16 更好” 或 “FP16 更好”，而应写成：

> 在本实验的单次消融闭环中，选择该闭环内最高精度的 mixed precision 设置作为 baseline。

## 6. 通道数为什么不一样

通道数不一样的根本原因是两个项目的搜索空间不同。

### 6.1 optimized 是 3 维 stage-level 搜索

`resnet32_l40s_optimized_ga_pso_search` 使用 3 维通道配置：

```text
[c1, c2, c3]
```

含义是每个 stage 内 5 个 residual block 使用同一个通道数。例如：

```text
[12, 24, 32]
```

表示：

```text
stage1: 12,12,12,12,12
stage2: 24,24,24,24,24
stage3: 32,32,32,32,32
```

这是粗粒度的 stage-level 宽度搜索。

### 6.2 mature 是 15 维 block-level 搜索

`resnet32_l40s_mature_ga_pso` 使用 15 维 block-level 通道配置：

```text
[c1_1,c1_2,c1_3,c1_4,c1_5,
 c2_1,c2_2,c2_3,c2_4,c2_5,
 c3_1,c3_2,c3_3,c3_4,c3_5]
```

例如 mature 最佳模型：

```text
[8,8,12,8,8, 20,20,16,16,16, 48,32,40,40,64]
```

它可以在同一个 stage 内保留不同 block 的通道宽度。这样既能压缩浅层和部分中间 block，又能在深层保留较大通道，例如 `48/40/64`。

注意 mature summary 中的：

```text
stage_channels=[8,20,48]
```

只是从 15 维 block_channels 中取第 1、6、11 个通道作为摘要，不代表整个 stage 都是 `[8,20,48]`。因此不能把它和 optimized 的 `[8,20,32]` 或 `[12,24,32]` 直接比较。

## 7. mature 还引入了 KD

`mature` 最终训练使用了 KD：

```text
teacher: runs/resnet32_baseline/best.pt
kd_mode: logits
kd_alpha: 0.7
kd_temperature: 4.0
```

`optimized` 的旧压缩模型没有 KD，只是用 sliced weight inheritance 初始化后训练。因此 mature 的最终压缩精度不仅来自 block-level 搜索，也来自 teacher distillation。

这也是两个项目最终压缩模型不能直接横向比较的重要原因。

## 8. provenance 问题

`optimized` 旧压缩结果使用的是旧 baseline：

```text
resnet32_bs128: 92.83%
```

而新补跑的 optimized baseline sweep 得到：

```text
resnet32_l40s_b128_fp16: 93.30%
```

旧压缩模型 summary 中缺少 `batch_size / lr / amp_dtype / channels_last / baseline_ckpt` 等字段，因此它和新 baseline sweep 不属于完全一致的实验闭环。后续如果要严谨比较，应使用新的 `resnet32_l40s_b128_fp16` baseline 重新训练压缩模型。

`mature` 则把：

```text
resnet32_l40s_b128_bf16: 93.45%
```

固化为 `runs/resnet32_baseline`，并用于：

- 权重继承
- BN gamma / importance 来源
- KD teacher
- 最终 comparison baseline

## 9. 建议写法

报告中建议采用如下表述：

> 两个目录对应不同实验阶段。`resnet32_l40s_optimized_ga_pso_search` 使用 3 维 stage-level 通道宽度搜索；`resnet32_l40s_mature_ga_pso` 使用 15 维 block-level 通道宽度搜索，并在最终训练中引入 logits KD。因此二者的压缩通道配置不可直接等价比较。
>
> 对 FP16/BF16 baseline，两个目录均使用标准 ResNet32 `[16,32,64]`，但训练设置不是严格确定性的：`deterministic=False`、`cudnn.benchmark=True`、多 worker 数据加载、随机数据增强以及不同 AMP kernel 都会导致训练轨迹变化。因此 FP16/BF16 的单次运行结果可能出现方向反转。本文只在各自实验闭环内选择最高精度 baseline，而不声称 BF16 或 FP16 在所有设置下稳定优于另一方。

## 10. 后续严谨验证建议

如果需要证明 FP16/BF16 谁更稳定，应额外做多 seed 实验：

```text
seed = 42, 43, 44
batch = 128
amp_dtype = fp16 / bf16
```

然后报告 mean ± std，而不是只报告单次 best acc。

如果需要严格复现，应关闭非确定性因素：

```text
deterministic=True
cudnn.benchmark=False
固定 DataLoader worker seed
固定数据增强随机源
```

但这种设置通常会降低训练速度，不一定适合作为 L40S 性能优化实验的默认配置。
