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
python train.py --mode fixmatch --data-root data/MSTAR --label-ratio 0.1 --epochs 80 --batch-size 32 --out runs/03_fixmatch_10percent
```

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

