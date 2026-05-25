from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, **kwargs: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(kwargs, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def _unwrap_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj  # type: ignore[return-value]


def _baseline_key_for_width_key(key: str) -> str:
    # WidthResNet32 uses stage1/stage2/stage3; baseline ResNet32 uses layer1/layer2/layer3.
    return (
        key.replace("stage1.", "layer1.")
        .replace("stage2.", "layer2.")
        .replace("stage3.", "layer3.")
        .replace("avg_pool.", "avgpool.")
    )


def load_sliced_baseline_weights(model: nn.Module, checkpoint_path: str | Path, verbose: bool = True) -> Dict[str, int]:
    """Initialize a width-compressed ResNet32 from a standard ResNet32 checkpoint.

    Tensors are copied by common prefix slices in every dimension. This is a
    pragmatic and fast weight inheritance strategy for stage-width search.
    It avoids training every candidate from scratch during GA/PSO evaluation.
    """
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    src_state = _unwrap_state_dict(ckpt)
    dst_state = model.state_dict()
    new_state = dict(dst_state)

    copied = sliced = skipped = 0
    for dst_key, dst_tensor in dst_state.items():
        src_key = _baseline_key_for_width_key(dst_key)
        if src_key not in src_state or not torch.is_tensor(src_state[src_key]):
            skipped += 1
            continue
        src_tensor = src_state[src_key]
        if src_tensor.shape == dst_tensor.shape:
            new_state[dst_key] = src_tensor.to(dtype=dst_tensor.dtype)
            copied += 1
        elif src_tensor.ndim == dst_tensor.ndim:
            patched = dst_tensor.clone()
            if src_tensor.ndim == 0:
                patched.copy_(src_tensor.to(dtype=dst_tensor.dtype))
            else:
                common_shape = tuple(min(a, b) for a, b in zip(src_tensor.shape, dst_tensor.shape))
                slices = tuple(slice(0, d) for d in common_shape)
                patched[slices] = src_tensor[slices].to(dtype=dst_tensor.dtype)
            new_state[dst_key] = patched
            sliced += 1
        else:
            skipped += 1

    model.load_state_dict(new_state, strict=True)
    stats = {"copied": copied, "sliced": sliced, "skipped": skipped}
    if verbose:
        print(f"[weight inheritance] {checkpoint_path}: {stats}")
    return stats
