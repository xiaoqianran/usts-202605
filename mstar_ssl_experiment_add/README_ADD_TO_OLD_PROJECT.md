# 在原 mstar_ssl_experiment 版本中加入实践二报告生成

原来的训练代码已经会生成以下实验记录：

```text
runs/实验名/
  metrics.json
  history.csv
  confusion_matrix.csv
```

这些文件已经足够支持实践二要求的图表，包括训练/测试曲线和混淆矩阵。
本补丁只是在原项目基础上新增“报告生成”功能，不改变原来的训练逻辑。

## 1. 复制文件

把本补丁中的文件复制到你的原项目根目录：

```text
mstar_ssl_experiment/
  generate_practice2_report.py      # 新增
  requirements_report.txt           # 新增
  scripts/run_all_with_report.sh     # 新增，可选
```

## 2. 安装报告生成依赖

```bash
pip install -r requirements_report.txt
```

原项目训练依赖仍使用原来的 `requirements.txt`。

## 3. 已经跑完实验时，直接生成报告

确保原项目已有：

```text
runs/
  01_supervised_full/metrics.json history.csv confusion_matrix.csv
  02_supervised_10percent/metrics.json history.csv confusion_matrix.csv
  03_fixmatch_10percent/metrics.json history.csv confusion_matrix.csv
```

然后运行：

```bash
python generate_practice2_report.py --runs runs --out 组号-人工智能实训II-实践2.docx
```

## 4. 从头训练并自动生成报告

也可以使用新增脚本：

```bash
bash scripts/run_all_with_report.sh data/MSTAR
```

它会先跑原来的三组实验，再自动调用 `generate_practice2_report.py` 生成 Word 报告。

## 5. 生成内容

报告会自动包含：

- 基线 vs 有限标签精度对比表
- 精度柱状图/恢复图
- 三组训练集与测试集 loss / accuracy 曲线
- 三组混淆矩阵
- FixMatch 方法简介与原理
- FixMatch 实现框图
- 半监督加入后的精度恢复情况分析
- 环境问题与解决过程

## 6. 原代码需要改哪里？

训练代码基本不用改，因为原来的 `train_supervised.py` 和 `train_fixmatch.py` 已经保存了报告需要的记录。
只建议做两个小改动：

### requirements.txt 增加

```text
pandas
matplotlib
python-docx
```

### scripts/run_all.sh 末尾可选增加

```bash
python generate_practice2_report.py --runs runs --out 组号-人工智能实训II-实践2.docx
```

如果不想改原脚本，直接使用本补丁里的 `scripts/run_all_with_report.sh`。
