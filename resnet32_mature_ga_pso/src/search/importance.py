from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


STANDARD_BN2_KEYS = [
    *[f"layer1.{i}.bn2.weight" for i in range(5)],
    *[f"layer2.{i}.bn2.weight" for i in range(5)],
    *[f"layer3.{i}.bn2.weight" for i in range(5)],
]


def load_block_importance_from_baseline(checkpoint_path: str | Path | None) -> list[float] | None:
    """Return 15 normalized block importance scores from BN gamma.

    Scores are based on mean(abs(gamma)) of each block's second BN layer.
    If no checkpoint is given or keys cannot be found, return None.
    """
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    scores = []
    for key in STANDARD_BN2_KEYS:
        if key not in state:
            return None
        gamma = state[key]
        scores.append(float(gamma.abs().mean().item()))
    if not scores:
        return None
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-12:
        return [0.5 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def default_block_search_space() -> list[list[int]]:
    return [[8, 12, 16] for _ in range(5)] + [[16, 20, 24, 28, 32] for _ in range(5)] + [[32, 40, 48, 56, 64] for _ in range(5)]


def candidate_key(channels: Sequence[int]) -> str:
    return "-".join(map(str, channels))
