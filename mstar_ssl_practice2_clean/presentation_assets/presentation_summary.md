# 实践二答辩素材摘要

| 实验 | 有标签样本 | 无标签/未使用样本 | 最优测试精度 | 最优 epoch |
| --- | ---: | ---: | ---: | ---: |
| Full supervised | 2746 | 0 | 97.61% | 71 |
| 10% supervised | 268 | 2478 | 78.68% | 60 |
| FixMatch | 268 | 2478 | 87.46% | 34 |

建议答辩顺序：先说明全标签基线，再说明标签减少到 10% 后的精度下降，最后展示 FixMatch 如何利用剩余无标签样本恢复精度。

可插入图片：`accuracy_recovery.png`、`training_curves.png`、`confusion_matrices.png`、`fixmatch_flowchart.png`。