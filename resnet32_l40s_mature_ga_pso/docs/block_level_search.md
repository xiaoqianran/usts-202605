# 15维 block-level ResNet32 通道配置搜索与 KD

新版搜索变量为：

```text
x = [
  c1_1, c1_2, c1_3, c1_4, c1_5,
  c2_1, c2_2, c2_3, c2_4, c2_5,
  c3_1, c3_2, c3_3, c3_4, c3_5
]
```

其中每个变量表示一个 residual block 的输出通道数。模型深度仍然是 CIFAR-10 ResNet32：每个 stage 5 个 residual blocks，总共 15 个 blocks。

默认搜索空间已恢复为与第一版主项目一致的 aggressive 设置（包含 8）：

```text
stage1 每个 block: {8, 12, 16}
stage2 每个 block: {16, 20, 24, 28, 32}
stage3 每个 block: {32, 40, 48, 56, 64}
```

搜索阶段使用：

```text
1. baseline checkpoint 的 sliced weight inheritance
2. train/val split 的 proxy evaluation
3. GA 与 PSO 两种计算智能算法
```

最终训练阶段使用 KD：

```text
teacher: 原始 ResNet32 baseline best.pt
student: GA/PSO 搜到的 block-level 压缩 ResNet32
loss: (1-alpha) * CE + alpha * T^2 * KL(student/T, teacher/T)
```

推荐默认：

```text
--kd-mode logits
--kd-alpha 0.7
--kd-temperature 4.0
```

可选增强：

```text
--kd-mode logits_at
--at-weight 25.0
```

`logits_at` 会额外对齐 stage1/stage2/stage3 的空间 attention map。由于该模式依赖 forward hooks，不建议同时使用 `--compile`。

H100 推荐搜索命令：

```bash
bash scripts/run_h100_search_fast.sh
```

搜索结束后查看：

```bash
cat runs/block_channel_search_fast_*/final_train_command.txt
```

输出命令已经默认包含 teacher KD。复制运行即可完整训练最终压缩模型。
