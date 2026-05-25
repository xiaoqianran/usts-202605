from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, **kwargs: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(kwargs, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
