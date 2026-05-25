# 实验流程建议

## 第一阶段：baseline

```bash
python train_resnet32.py --run-name resnet32_baseline --epochs 200 --milestones 100,150 --amp
```

记录：

```text
Accuracy
Params
FLOPs
Training time
```

## 第二阶段：GA/PSO 搜索

```bash
python search_channels_ga_pso.py --algorithm both --search-epochs 3 --ga-population 8 --ga-generations 5 --pso-particles 8 --pso-iterations 5 --amp
```

记录：

```text
GA best channels
PSO best channels
GA search time
PSO search time
candidate evaluations
```

## 第三阶段：训练最终压缩模型

假设搜索得到 `[16, 24, 48]`：

```bash
python train_width_resnet32.py --channels 16,24,48 --run-name final_width_16-24-48 --epochs 80 --milestones 40,60 --amp
```

## 第四阶段：对比

```bash
python compare_results.py --baseline runs/resnet32_baseline/summary.json --compressed runs/final_width_16-24-48/summary.json --output runs/final_comparison.json
```

论文结果表建议列：

```text
模型 / 算法
通道配置
Accuracy
Params
参数压缩率
FLOPs
FLOPs下降率
搜索时间
训练时间
```
