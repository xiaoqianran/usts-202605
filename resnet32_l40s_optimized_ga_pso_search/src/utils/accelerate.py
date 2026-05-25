from __future__ import annotations

import torch
import torch.nn as nn


def setup_torch_fast(tf32: bool = True, benchmark: bool = True) -> None:
    """Speed-oriented PyTorch settings for fixed-shape CIFAR training."""
    torch.backends.cudnn.benchmark = benchmark
    if torch.cuda.is_available() and tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def maybe_channels_last(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    return model


def maybe_compile(model: nn.Module, enabled: bool, mode: str = "reduce-overhead") -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode=mode)
    except Exception as exc:
        print(f"[warning] torch.compile disabled because it failed: {exc}")
        return model


def move_images(images: torch.Tensor, device: torch.device, channels_last: bool = False) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last and images.ndim == 4 and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def autocast_context(device: torch.device, enabled: bool, dtype: str = "bf16"):
    if device.type != "cuda" or not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    amp_dtype = torch.bfloat16 if dtype.lower() in {"bf16", "bfloat16"} else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool, dtype: str = "bf16"):
    # BF16 does not need GradScaler. FP16 benefits from it.
    use_scaler = bool(enabled and device.type == "cuda" and dtype.lower() in {"fp16", "float16"})
    return torch.amp.GradScaler(device="cuda", enabled=use_scaler)
