# L40S Optimized 加速版改动说明

本版针对 L40S（以及其他支持 BF16 的现代 GPU）进行了多项优化，目标是：
- 提升训练和搜索吞吐
- 让 GA/PSO 搜索阶段的候选模型评估更稳定可靠（通过权重继承 + 更好的 fitness）

## 已改动 / 推荐开启的优化

1. 默认使用 BF16 混合精度（`--amp --amp-dtype bf16`），比 FP16 数值更稳定。
2. 支持 `--channels-last`（NHWC 内存布局，通常有加速）。
3. 启用 TF32 + cudnn benchmark（`setup_torch_fast`）。
4. DataLoader 支持较高 `num_workers` + `persistent_workers`（按实际 CPU 核数调整）。
5. 搜索阶段支持从标准 baseline checkpoint 继承权重（`--baseline-ckpt` + `load_sliced_baseline_weights`），让短训练（1 epoch）也能得到有意义的 proxy 性能。
6. 改进的 fitness 函数：引入 `allowed_short_acc_drop` + 相对短 baseline 的精度约束 + penalty，避免搜索出过小的模型。
7. 搜索阶段默认使用 train/val split（避免 test set 泄漏）。
8. 可选 `--compile`（最终完整训练时可能有效，搜索阶段通常不建议开）。

## 推荐命令

先保证 baseline checkpoint 存在：

```bash
bash scripts/run_l40s_baseline_fast.sh
```

如果已经有 `runs/resnet32_baseline/best.pt`，可直接搜索：

```bash
bash scripts/run_l40s_search_fast.sh
```

搜索完成后，查看：

```bash
cat runs/channel_search_fast_*/final_train_command.txt
```

然后复制该命令训练最终压缩模型。

## 注意事项

- 搜索时 batch size 建议根据实际 GPU 显存和 CPU 调整。L40S 上 1536~2048 通常可行，但 step 数太少会影响短训练 proxy 的可靠性。
- `torch.compile` 对搜索阶段帮助有限（每个候选都要编译），建议只在最终训练时尝试。
- 所有优化在 L40S 上实测有效，同时也兼容其他 Ada / Ampere 及以上架构的 GPU。
