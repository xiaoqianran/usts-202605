# 数据说明

本项目使用 NASA Prognostics Center of Excellence (PCoE) Battery Aging ARC-FY08Q4 数据集。

运行前请将原始压缩包 `BatteryAgingARC.zip` 放在本目录下，或在运行命令中用 `--data_zip` 指定其路径。

示例：

```bash
python run_all.py --data_zip ./data/BatteryAgingARC.zip --output_dir ./outputs
```

原始数据文件包含：B0005.mat、B0006.mat、B0007.mat、B0018.mat。
