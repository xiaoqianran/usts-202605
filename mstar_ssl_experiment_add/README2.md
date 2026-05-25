mstar_ssl_experiment/
  train_supervised.py
  train_fixmatch.py
  src/
  scripts/
  runs/
  generate_practice2_report.py        # 新增
  requirements_report.txt             # 新增
  scripts/run_all_with_report.sh       # 新增，可选


## 1. 已经跑完实验的话，直接执行：
pip install -r requirements_report.txt

python generate_practice2_report.py \
  --runs runs \
  --out 组号-人工智能实训II-实践2.docx


## 它会基于你原来的：

runs/01_supervised_full/history.csv
runs/01_supervised_full/confusion_matrix.csv
runs/01_supervised_full/metrics.json

runs/02_supervised_10percent/history.csv
runs/02_supervised_10percent/confusion_matrix.csv
runs/02_supervised_10percent/metrics.json

runs/03_fixmatch_10percent/history.csv
runs/03_fixmatch_10percent/confusion_matrix.csv
runs/03_fixmatch_10percent/metrics.json

## 自动生成你要的这些内容：
实践二部分（2~3页）
1. 基线 vs 有限标签：
   - 精度对比表
   - 柱状图
   - 训练/测试 loss 曲线
   - 训练/测试 accuracy 曲线
   - 混淆矩阵

2. 半监督方法：
   - FixMatch 方法简介
   - 原理说明
   - 实现框图

3. 结果对比：
   - 加入半监督后的精度恢复情况
   - 表格/折线图
   - 三组训练/测试曲线
   - 三组混淆矩阵

## 2. 最小修改方案就是：
```
# 原代码不动，只加这个文件
generate_practice2_report.py
```