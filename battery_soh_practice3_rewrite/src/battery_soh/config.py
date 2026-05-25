from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BATTERIES = ("B0005", "B0006", "B0007", "B0018")
RATED_CAPACITY_AH = 2.0


@dataclass(frozen=True)
class PipelineConfig:
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
        return self.output_dir / "report_assets"

