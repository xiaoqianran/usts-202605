# MSTAR 深度半监督学习实践二

这是对原 `mstar_ssl_experiment` / `mstar_ssl_experiment_add` 的重写版。项目目标不是只跑训练，而是完整支撑《人工智能开发实训 II》实践二和答辩展示。

## 实验设计

三组实验直接对应任务书要求：

| 实验 | 标签使用方式 | 目的 |
| --- | --- | --- |
| 全标签监督 | 使用全部 `mstar-train` 标签 | 得到分类网络参考上限 |
| 10% 标签监督 | 每类只保留约 10% 有标签样本 | 观察标签有限导致的性能下降 |
| FixMatch 半监督 | 同样 10% 有标签 + 其余训练图像作为无标签 | 验证半监督方法能否恢复精度 |

默认模型为 `SmallResNet`，输入 1 通道 MSTAR SAR 灰度图，输出 10 类目标分类结果。

## 目录结构

```text
mstar_ssl_practice2_clean/
  train.py                         # 统一训练入口：supervised / fixmatch
  src/mstar_ssl/
    data.py                        # 数据读取、分层有限标签划分、FixMatch 双视图数据集
    transforms.py                  # 弱增强、强增强、归一化
    models.py                      # SmallResNet / SmallCNN
    eval.py                        # 测试与混淆矩阵
    utils.py                       # 随机种子、JSON/CSV 工具
  tools/
    inspect_dataset.py             # 数据集统计
    extract_mstar.py               # 解压 MSTAR.zip
    make_presentation_assets.py    # 生成答辩图表和摘要
  scripts/run_all.sh               # 一键跑三组实验并生成答辩素材
  docs/答辩展示提纲.md
```

## 环境安装

```bash
cd mstar_ssl_practice2_clean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

期望数据结构：

```text
data/MSTAR/
  mstar-train/
    2S1/
    BMP2/
    ...
  mstar-test/
    2S1/
    BMP2/
    ...
```

如果手头是 `MSTAR.zip`：

```bash
python tools/extract_mstar.py --zip MSTAR.zip --out data/MSTAR
```

检查数据：

```bash
python tools/inspect_dataset.py --data-root data/MSTAR
```

## 运行实验

一键运行：

```bash
bash scripts/run_all.sh data/MSTAR
```

快速自检可减少 epoch：

```bash
EPOCHS=2 bash scripts/run_all.sh data/MSTAR
```

单独运行三组实验：

```bash
python train.py --mode supervised --data-root data/MSTAR --label-ratio 1.0 --epochs 80 --out runs/01_supervised_full
python train.py --mode supervised --data-root data/MSTAR --label-ratio 0.1 --epochs 80 --out runs/02_supervised_10percent
python train.py --mode fixmatch --data-root data/MSTAR --label-ratio 0.1 --epochs 80 --batch-size 64 --mu 2 --out runs/03_fixmatch_10percent
```

为保证有限标签监督实验与 FixMatch 半监督实验之间的可比性，本实验将两者的有标签 batch size 均设置为 64。FixMatch 额外引入无标签样本，其无标签 batch size 由参数 μ 控制。本实验设置 μ=2，因此每个训练 step 同时使用 64 张有标签样本和 128 张无标签样本。这样既保证了监督信号规模一致，又体现了半监督方法利用额外无标签数据的特点。

| 实验名称 | 模式 | 有标签 batch size | μ | 无标签 batch size |
| --- | --- | ---: | ---: | ---: |
| 全标签监督 | supervised | 64 | - | - |
| 10%标签监督 | supervised | 64 | - | - |
| FixMatch半监督 | fixmatch | 64 | 2 | 128 |

### batch32 补充实验

为进一步排除 batch size 对结果解释的影响，项目追加一组全部使用有标签 batch size 32 的补充实验。该组实验不覆盖默认 `runs/`，而是输出到 `runs_batch32/`：

```bash
bash scripts/run_all_batch32.sh data/MSTAR
```

batch32 组的学习率按线性缩放规则从默认 batch64 的 `0.001` 调整为 `0.0005`。三组实验使用相同学习率，保证主要变量仍然是标签数量和是否使用无标签数据。

| 实验名称 | 模式 | 有标签 batch size | μ | 无标签 batch size | 学习率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 全标签监督 | supervised | 32 | - | - | 0.0005 |
| 10%标签监督 | supervised | 32 | - | - | 0.0005 |
| FixMatch半监督 | fixmatch | 32 | 4 | 128 | 0.0005 |

FixMatch 中 `batch-size` 仍表示有标签 batch size，`μ=4` 表示每个 step 额外使用 `32×4=128` 张无标签样本。该设置保证三组实验的监督信号 batch size 都是 32，同时保留半监督方法额外利用无标签样本的特点。

### batch96 补充实验

项目还追加一组全部使用有标签 batch size 96 的补充实验，输出到 `runs_batch96/`：

```bash
bash scripts/run_all_batch96.sh data/MSTAR
```

batch96 组的学习率同样按线性缩放规则从 batch64 的 `0.001` 调整为 `0.0015`。三组实验使用相同学习率，保证组内对照公平。

| 实验名称 | 模式 | 有标签 batch size | μ | 无标签 batch size | 学习率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 全标签监督 | supervised | 96 | - | - | 0.0015 |
| 10%标签监督 | supervised | 96 | - | - | 0.0015 |
| FixMatch半监督 | fixmatch | 96 | 2 | 192 | 0.0015 |

FixMatch 中 `batch-size` 表示有标签 batch size，`μ=2` 表示每个 step 额外使用 `96×2=192` 张无标签样本。该组用于观察更大有标签 batch size 和线性缩放学习率下，有限标签监督与半监督方法的表现是否稳定。

## 输出结果

每个实验目录固定输出：

```text
args.json              # 训练参数
model_summary.json     # 网络结构摘要
split_ratio*.json      # 有标签/无标签划分
history.csv            # 每个 epoch 的 loss/acc
metrics.json           # 最优测试精度、最优 epoch、样本数
confusion_matrix.csv   # 测试集混淆矩阵
best.pt                # 最优模型权重
```

生成答辩素材：

```bash
python tools/make_presentation_assets.py --runs runs --out presentation_assets
```

生成文件包括：

- `accuracy_recovery.png`：基线、有限标签、FixMatch 精度对比。
- `training_curves.png`：训练/测试损失和精度曲线。
- `confusion_matrices.png`：三组实验混淆矩阵。
- `fixmatch_flowchart.png`：半监督方法流程图。
- `presentation_summary.md`：可直接写入报告或答辩稿的摘要表。

## 已有旧实验结果可迁移

旧目录 `mstar_ssl_experiment_add/runs` 已经有完整三组实验结果，可先复制到新目录验证图表生成：

```bash
cp -r ../mstar_ssl_experiment_add/runs ./runs
python tools/make_presentation_assets.py --runs runs --out presentation_assets
```
