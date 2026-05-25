from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def human_number(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n / 1e9:.3f}G"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.3f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.3f}K"
    return f"{n:.0f}"


@torch.no_grad()
def measure_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 32, 32),
    device: str | torch.device = "cpu",
) -> int:
    """Approximate multiply-add FLOPs for one image.

    Counts Conv2d and Linear layers only. BatchNorm/ReLU/residual addition are ignored,
    which is sufficient for a clean coursework baseline comparison.
    """
    model = model.to(device)
    model.eval()
    flops: Dict[str, int] = {"total": 0}
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        batch_size = out.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total_ops = batch_size * out_h * out_w * module.out_channels * kernel_ops
        flops["total"] += int(total_ops / batch_size)

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        batch_size = inputs[0].shape[0]
        total_ops = batch_size * module.in_features * module.out_features
        flops["total"] += int(total_ops / batch_size)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, *input_size, device=device)
    model(dummy)

    for hook in hooks:
        hook.remove()

    return flops["total"]
