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
    summarize_fixmatch_grid.py     # 汇总 FixMatch batch/mu 网格结果
    summarize_supervised_grid.py   # 汇总监督学习 batch/lr 网格结果
  scripts/
    run_all.sh                     # 一键跑基础三组
    run_fixmatch_grid.sh           # 跑 FixMatch 8组网格 (fixed/scaled)
    run_supervised_grid.sh         # 跑监督学习 8组网格 (full + 10% x 32/64/96/128)
    monitor_training.sh            # 实时监控 GPU + 各 run 进度 + OOM 检测
    babysit_supervised_grids.sh    # 顺序跑完监督 fixed + scaled 两套网格
  docs/答辩展示提纲.md
  docs/实验结果汇总.md             # 已完成基础对照和四套网格实验结果
  docs/监控提示词.md               # 给其他 agent 的看管 prompt（本文件）
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

为保证有限标签监督实验与 FixMatch 半监督实验之间的可比性，本实验将两者的有标签 batch size 均设置为 64。FixMatch 额外引入无标签样本，其无标签 batch size 由参数 μ 控制。基础组设置 μ=2，因此每个训练 step 同时使用 64 张有标签样本和 128 张无标签样本。这样既保证了监督信号规模一致，又体现了半监督方法利用额外无标签数据的特点。

| 实验名称 | 模式 | 有标签 batch size | μ | 无标签 batch size |
| --- | --- | ---: | ---: | ---: |
| 全标签监督 | supervised | 64 | - | - |
| 10%标签监督 | supervised | 64 | - | - |
| FixMatch半监督 | fixmatch | 64 | 2 | 128 |

### FixMatch batch/mu 网格补充实验

为了让半监督实验更严谨，除基础三组对照外，追加 FixMatch 的二维网格实验。网格因素如下：

- 有标签 batch size：32、64、96、128。
- μ：2、4。
- 标签划分：固定 `--label-ratio 0.1 --seed 42`，确保所有实验使用同一批有标签样本和同一批无标签样本。
- 训练轮数、模型、阈值和无标签损失权重：统一为 `80` epoch、`SmallResNet`、`threshold=0.95`、`lambda_u=1.0`。

这样一共得到 `4 × 2 = 8` 组 FixMatch 实验，专门分析 batch size 和无标签样本比例 μ 对半监督效果的影响。

#### 主对照：固定学习率

```bash
LR_POLICY=fixed PARALLEL_JOBS=4 bash scripts/run_fixmatch_grid.sh data/MSTAR
```

固定学习率组全部使用 `lr=0.001`，这是最严格的主对照：学习率不变，只改变 batch size 和 μ，便于判断性能差异是否来自 batch 或无标签比例本身。输出目录为 `runs_fixmatch_grid_fixed/`。

| 编号 | 有标签 batch size | μ | 无标签 batch size | 学习率 | 输出目录 |
| --- | ---: | ---: | ---: | ---: | --- |
| F1 | 32 | 2 | 64 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b32_mu2` |
| F2 | 32 | 4 | 128 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b32_mu4` |
| F3 | 64 | 2 | 128 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b64_mu2` |
| F4 | 64 | 4 | 256 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b64_mu4` |
| F5 | 96 | 2 | 192 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b96_mu2` |
| F6 | 96 | 4 | 384 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b96_mu4` |
| F7 | 128 | 2 | 256 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b128_mu2` |
| F8 | 128 | 4 | 512 | 0.001 | `runs_fixmatch_grid_fixed/fixmatch_b128_mu4` |

#### 学习率敏感性分析：线性缩放学习率

固定学习率可以隔离变量，但大 batch 往往需要更大的学习率才能保持相近的参数更新尺度。因此再追加一组线性缩放学习率实验，作为敏感性分析：

```bash
LR_POLICY=scaled PARALLEL_JOBS=4 bash scripts/run_fixmatch_grid.sh data/MSTAR
```

线性缩放规则以 batch64 的 `lr=0.001` 为基准：

| 有标签 batch size | 学习率 |
| ---: | ---: |
| 32 | 0.0005 |
| 64 | 0.001 |
| 96 | 0.0015 |
| 128 | 0.002 |

该组输出目录为 `runs_fixmatch_grid_scaled/`。如果固定学习率和线性缩放学习率得出的最优 batch/μ 一致，说明结论更稳定；如果不一致，则报告中应明确说明 batch size 与学习率存在耦合，不能只用单一学习率结论解释半监督效果。

#### 推荐执行顺序

1. 先跑基础三组对照，得到全标签上限、10% 标签监督下限和 FixMatch 基础结果。
2. 再跑固定学习率 8 组网格，把它作为正式报告的主要 batch/mu 消融实验。
3. 最后跑线性缩放学习率 8 组网格，作为学习率敏感性分析。如果时间或算力有限，至少完成固定学习率 8 组。

`PARALLEL_JOBS` 控制同时训练的实验数量。H100 80GB 上不建议直接设置为 8，因为 `batch=96, μ=4` 和 `batch=128, μ=4` 与其他实验同时运行时容易把显存占满；推荐先用 `PARALLEL_JOBS=4` 跑完整实验。

网格实验完成后会自动生成汇总：

```text
runs_fixmatch_grid_fixed/summary/fixmatch_grid_summary.csv
runs_fixmatch_grid_fixed/summary/fixmatch_grid_summary.md
runs_fixmatch_grid_scaled/summary/fixmatch_grid_summary.csv
runs_fixmatch_grid_scaled/summary/fixmatch_grid_summary.md
```

也可以手动汇总任意网格目录：

```bash
python tools/summarize_fixmatch_grid.py --runs runs_fixmatch_grid_fixed --out runs_fixmatch_grid_fixed/summary
```

## Supervised batch/lr 网格对标实验

为了与 FixMatch 网格形成严谨对照，补充了**纯监督学习**在相同 batch size 下的表现：

- 全标签监督（label_ratio=1.0）：全部 2746 个训练样本有标签，作为性能上限参考。
- 10% 标签监督（label_ratio=0.1）：仅保留 ~268 个有标签样本，其余 2478 张作为“无标签但实际未使用”（模拟有限标签场景的下限）。

网格因素与 FixMatch 完全对齐：
- batch size：32 / 64 / 96 / 128
- 学习率策略：**fixed** (全部 0.001) 和 **scaled** (按 batch 线性缩放：32→0.0005, 64→0.001, 96→0.0015, 128→0.002)
- 统一 80 epoch、SmallResNet、seed=42

共 2 策略 × 4 batch × 2 标签比例 = **16 组** 监督实验。

运行命令（推荐用 babysit 脚本，它会按顺序跑 fixed 再 scaled，并自动跳过已完成的）：

```bash
# 方式一：一键顺序跑两套（推荐，带监控）
bash scripts/babysit_supervised_grids.sh data/MSTAR

# 方式二：手动分开跑（支持断点续跑）
LR_POLICY=fixed  PARALLEL_JOBS=4 bash scripts/run_supervised_grid.sh data/MSTAR
LR_POLICY=scaled PARALLEL_JOBS=4 bash scripts/run_supervised_grid.sh data/MSTAR
```

实时监控（另一个终端或 agent）：

```bash
watch -n 20 'bash scripts/monitor_training.sh --once'
# 或
bash scripts/monitor_training.sh --watch 30
```

监控脚本会显示：
- 当前 GPU 显存/利用率/功耗
- 每个 run 的状态（DONE / RUN）、当前/最优 epoch、测试精度
- 自动扫描 train.log 中的 OOM / CUDA / killed 等致命错误

结果汇总自动生成在：

```text
runs_supervised_grid_fixed/summary/supervised_grid_summary.md   (固定 lr)
runs_supervised_grid_scaled/summary/supervised_grid_summary.md  (线性缩放 lr)
```

这两张表 + FixMatch 的两张表，共同构成完整的 batch size / 学习率 / 监督 vs 半监督 四维对照实验。

## 已完成实验结果

当前目录已经完成基础对照、batch96 补充对照、FixMatch 网格和监督网格，共 38 个训练 run。所有结果均使用 MSTAR 10 类分类数据，训练集 2746 张、测试集 2425 张；10% 标签设置下按类别分层保留 268 张有标签样本，剩余 2478 张训练图像在纯监督实验中不使用，在 FixMatch 中作为无标签样本使用。

### 基础三组对照

`runs/` 使用 batch size 64、lr=0.001；FixMatch 使用 μ=2。

| 实验 | 有标签样本 | 无标签/未使用样本 | batch size | μ | 学习率 | 最优测试精度 | 最优 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全标签监督 | 2746 | 0 | 64 | - | 0.001 | 97.07% | 71 |
| 10% 标签监督 | 268 | 2478 | 64 | - | 0.001 | 73.65% | 54 |
| FixMatch 半监督 | 268 | 2478 | 64 | 2 | 0.001 | 82.60% | 68 |

结论：标签从全量降到 10% 后，精度从 97.07% 降到 73.65%，下降 23.42 个百分点；FixMatch 在相同 268 个有标签样本基础上利用剩余无标签样本，提升到 82.60%，相对 10% 标签监督提高 8.95 个百分点，恢复了约 38.22% 的精度缺口。

### batch96 补充对照

`runs_batch96/` 使用 batch size 96、线性缩放 lr=0.0015；FixMatch 使用 μ=2。

| 实验 | 有标签样本 | 无标签/未使用样本 | batch size | μ | 学习率 | 最优测试精度 | 最优 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全标签监督 | 2746 | 0 | 96 | - | 0.0015 | 98.31% | 37 |
| 10% 标签监督 | 268 | 2478 | 96 | - | 0.0015 | 72.78% | 61 |
| FixMatch 半监督 | 268 | 2478 | 96 | 2 | 0.0015 | 85.57% | 52 |

结论：batch96 下 FixMatch 比 10% 标签监督提高 12.79 个百分点，恢复约 50.10% 的精度缺口，说明半监督收益不是 batch64 单一设置下的偶然结果。

### 监督学习 batch/lr 网格

完整表格见：

- `runs_supervised_grid_fixed/summary/supervised_grid_summary.md`
- `runs_supervised_grid_scaled/summary/supervised_grid_summary.md`

固定 lr=0.001 的最佳结果：

| 监督设置 | 最佳 batch size | 学习率 | 最优测试精度 | 最优 epoch |
| --- | ---: | ---: | ---: | ---: |
| 全标签监督 | 32 | 0.001 | 98.23% | 63 |
| 10% 标签监督 | 64 | 0.001 | 78.27% | 29 |

线性缩放 lr 的最佳结果：

| 监督设置 | 最佳 batch size | 学习率 | 最优测试精度 | 最优 epoch |
| --- | ---: | ---: | ---: | ---: |
| 全标签监督 | 32 | 0.0005 | 98.47% | 71 |
| 10% 标签监督 | 96 | 0.0015 | 80.87% | 35 |

结论：全标签监督在不同 batch/lr 下整体稳定，最高达到 98.47%；10% 标签监督对 batch 和学习率更敏感，固定 lr 下 batch64 最好，线性缩放 lr 下 batch96 最好，但仍显著低于全标签上限。

### FixMatch batch/mu 网格

完整表格见：

- `runs_fixmatch_grid_fixed/summary/fixmatch_grid_summary.md`
- `runs_fixmatch_grid_scaled/summary/fixmatch_grid_summary.md`

固定 lr=0.001 的最佳结果：

| batch size | μ | 无标签 batch size | 学习率 | 最优测试精度 | 最优 epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 128 | 0.001 | 88.62% | 60 |

线性缩放 lr 的最佳结果：

| batch size | μ | 无标签 batch size | 学习率 | 最优测试精度 | 最优 epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 64 | 0.0005 | 90.93% | 56 |

结论：FixMatch 在两套学习率策略下的最优点都出现在 batch size 32，说明本任务中较小有标签 batch 更有利于半监督训练。μ 的最优值会随学习率策略变化：固定 lr 下 μ=4 最好，线性缩放 lr 下 μ=2 最好，说明无标签比例与学习率存在耦合。全量网格中的最高 FixMatch 精度为 90.93%，比同一 batch/lr 下的 10% 标签监督 71.96% 高 18.97 个百分点。

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
