# 下一步：GA / PSO 搜索通道配置的接入计划

本版本只保留标准 ResNet32 baseline，不包含压缩模型、不包含通道搜索代码。

下一步加 GA/PSO 时，建议只新增以下模块，不破坏当前 baseline：

```text
src/search/
├── search_space.py      # 通道配置搜索空间，例如 [c1, c2, c3]
├── fitness.py           # Accuracy、Params、FLOPs 的适应度函数
├── ga_search.py         # 遗传算法
└── pso_search.py        # 粒子群优化

train_width_resnet32.py  # 可变通道 ResNet32 训练入口，和当前 baseline 分开
```

推荐先搜索 stage-level 通道配置：

```text
x = [c1, c2, c3]
原始 ResNet32: [16, 32, 64]
候选范围：
c1 ∈ {8, 12, 16}
c2 ∈ {16, 20, 24, 28, 32}
c3 ∈ {32, 40, 48, 56, 64}
```

适应度函数可以先写成单目标：

```text
Fitness(x) = Acc(x) - λ1 * Params(x)/Params0 - λ2 * FLOPs(x)/FLOPs0
```

搜索阶段不要完整训练每个候选模型，否则太慢。建议每个候选只训练少量 epoch，选出最优候选后再完整训练。
