# 数据说明

本项目使用 NASA PCoE Battery Aging 数据中的 `B0005.mat`、`B0006.mat`、`B0007.mat`、`B0018.mat`。

运行方式二选一：

1. 将 `BatteryAgingARC.zip` 放在 `data/BatteryAgingARC.zip`。
2. 将四个 `.mat` 文件放在 `data/extracted/` 或 `data/extracted/1. BatteryAgingARC-FY08Q4/`。

当前仓库已有旧实验目录时，`run_pipeline.py` 也会自动尝试复用：

- `../battery_soh_practice3_strict_code/data/extracted/1. BatteryAgingARC-FY08Q4`
- `../battery_soh_practice3/data/extracted/1. BatteryAgingARC-FY08Q4`

