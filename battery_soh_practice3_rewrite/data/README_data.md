# 数据说明

本项目使用 NASA PCoE Battery Aging 数据中的 `B0005.mat`、`B0006.mat`、`B0007.mat`、`B0018.mat`。

运行方式二选一：

1. 将 `BatteryAgingARC.zip` 放在 `data/BatteryAgingARC.zip`。
2. 或将四个 `.mat` 文件直接放在 `data/extracted/`。

运行 `run_pipeline.py` 时，如果 `data/extracted/` 里还没有 `.mat` 文件，程序会自动解压 `data/BatteryAgingARC.zip` 到该目录。
