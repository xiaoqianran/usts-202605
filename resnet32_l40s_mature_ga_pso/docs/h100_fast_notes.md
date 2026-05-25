# H100 加速版改动说明

本版重点解决 ResNet32/CIFAR-10 在 H100 上显存占用低、GPU 吃不满、搜索很慢的问题。

## 已改动

1. 默认 batch size 提高到 1024/2048。
2. DataLoader 默认 num_workers=8，并启用 persistent_workers、prefetch_factor、pin_memory。
3. 新增 BF16 AMP：`--amp --amp-dtype bf16`。
4. 新增 channels_last：`--channels-last`。
5. 新增 TF32/cudnn benchmark 设置。
6. 新增可选 `--compile`，但搜索阶段默认不建议开，因为每个 candidate 都会重新编译。
7. 搜索阶段新增 baseline checkpoint 权重继承：`--baseline-ckpt runs/resnet32_baseline/best.pt`。
8. 搜索阶段改为 train/val split，不再用 test set 作为 fitness。
9. fitness 改为 baseline-short 约束，避免 GA/PSO 只选择极小模型。
10. 默认搜索空间已恢复为与第一版一致的 aggressive 版本（包含 8）：`{8,12,16};{16,20,24,28,32};{32,40,48,56,64}`（block-level 会自动扩展为 15 维）。

## 推荐命令

先保证 baseline checkpoint 存在：

```bash
bash scripts/run_h100_baseline_fast.sh
```

如果已经有 `runs/resnet32_baseline/best.pt`，可直接搜索：

```bash
bash scripts/run_h100_search_fast.sh
```

搜索完成后，查看：

```bash
cat runs/channel_search_fast_*/final_train_command.txt
```

然后复制该命令训练最终压缩模型。

## 注意

`torch.compile` 对最终训练可能有效，但对搜索阶段不一定有效，因为每个候选模型都要单独编译，可能反而变慢。
