## 1. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 uv：

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. 解压数据

把 `MSTAR.zip` 放在项目根目录，然后执行：

```bash
python src/extract_mstar.py --zip MSTAR.zip --out data/MSTAR
```

也可以直接把数据解压为：

```text
data/MSTAR/mstar-train
数据/MSTAR/mstar-test
```

## 3. 检查数据集

```bash
python src/inspect_dataset.py --data-root data/MSTAR
```

本次上传的数据统计为：训练集 2746 张、测试集 2425 张、共 10 类。

## 4. 跑三组实验

推荐先用 50 epoch 跑通流程。如果机器较好，可改成 100 epoch。

```bash
bash scripts/run_all.sh data/MSTAR
```

等价于分别执行：

```bash
# A. 全标签监督基线
python train_supervised.py \
  --data-root data/MSTAR \
  --label-ratio 1.0 \
  --epochs 80 \
  --batch-size 64 \
  --lr 0.001 \
  --out runs/01_supervised_full

# B. 10% 以下标签监督学习
python train_supervised.py \
  --data-root data/MSTAR \
  --label-ratio 0.1 \
  --epochs 80 \
  --batch-size 64 \
  --lr 0.001 \
  --out runs/02_supervised_10percent

# C. 10% 标签 + 90% 无标签半监督 FixMatch
python train_fixmatch.py \
  --data-root data/MSTAR \
  --label-ratio 0.1 \
  --epochs 80 \
  --batch-size 32 \
  --mu 4 \
  --lr 0.001 \
  --threshold 0.95 \
  --lambda-u 1.0 \
  --out runs/03_fixmatch_10percent
```

## 5. 输出文件

每个实验目录会保存：

```text
runs/实验名/
  args.json              # 实验参数
  split_*.json           # 有标签/无标签划分，保证复现
  history.csv            # 每个 epoch 的 loss/acc
  metrics.json           # 最优测试精度等摘要结果
  best.pt                # 最优模型权重
  confusion_matrix.csv   # 测试集混淆矩阵
```

## 6. 报告建议结论写法

不要一上来承诺半监督一定超过全标签。报告重点是：

- 全标签监督学习给出该网络在 MSTAR 上的参考上限。
- 标签减少到 10% 以下后，监督学习精度通常下降，说明深度网络对标签数量敏感。
- 半监督方法使用剩余 90% 未标注样本，若测试精度高于 10% 标签监督模型，说明无标签样本中的数据分布信息能够缓解标签不足问题。
- 如果 FixMatch 没有明显提升，需要分析可能原因：伪标签阈值过高导致无标签利用率低，阈值过低导致错误伪标签累积；SAR 图像增强过强可能破坏目标散射特征。
