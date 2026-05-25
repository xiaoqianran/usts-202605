# 15维 block-level ResNet32 通道配置搜索

新版搜索变量为：

```text
x = [
  c1_1, c1_2, c1_3, c1_4, c1_5,
  c2_1, c2_2, c2_3, c2_4, c2_5,
  c3_1, c3_2, c3_3, c3_4, c3_5
]
```

其中每个变量表示一个 residual block 的输出通道数。模型深度仍然是 CIFAR-10 ResNet32：每个 stage 5 个 residual blocks，总共 15 个 blocks。

默认搜索空间采用保守设置：

```text
stage1 每个 block: {12, 16}
stage2 每个 block: {20, 24, 28, 32}
stage3 每个 block: {40, 48, 56, 64}
```

也可以用 `--space` 自定义。如果只给 3 组，会自动扩展成 15 维：

```bash
--space '12,16;20,24,28,32;40,48,56,64'
```

如果给 15 组，则逐 block 指定搜索空间。

H100 推荐搜索命令：

```bash
bash scripts/run_h100_search_fast.sh
```

搜索结束后查看：

```bash
cat runs/block_channel_search_fast_*/final_train_command.txt
```

然后复制输出命令完整训练最终压缩模型。

注意：搜索阶段默认使用 baseline checkpoint 做 sliced weight inheritance。这样候选模型不再从随机初始化开始，速度和排序稳定性都会更好。
