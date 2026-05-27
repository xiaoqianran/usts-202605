from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BATTERIES = ("B0005", "B0006", "B0007", "B0018")
RATED_CAPACITY_AH = 2.0


@dataclass(frozen=True)
class PipelineConfig:
    """管道配置数据类，集中管理 SOH 预测全流程的输入输出路径与超参数。

    Attributes:
        data_zip: NASA 电池老化数据集 zip 文件路径（BatteryAgingARC.zip）。
        data_dir: 解压后 .mat 文件存放目录。
        output_dir: 所有输出（报告、图片、CSV）根目录。
        top_k: Pearson 相关系数特征选择时保留的 Top-K 特征数量。
        seed: 全局随机种子，确保实验可复现。
        epochs: 神经网络最大训练轮数。
        patience: 早停耐心值（验证损失若干轮不下降则停止）。
        learning_rate: AdamW 优化器学习率。
        save_individual_plots: 是否为每个场景单独保存详细预测/损失曲线图。
    """

    data_zip: Path
    data_dir: Path
    output_dir: Path
    top_k: int = 8
    seed: int = 42
    epochs: int = 600
    patience: int = 60
    learning_rate: float = 2e-3
    save_individual_plots: bool = False

    @property
    def asset_dir(self) -> Path:
        """返回存放报告插图（PNG）和中间结果的子目录路径。"""
        return self.output_dir / "report_assets"

