from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KDLossParts:
    total: torch.Tensor
    ce: torch.Tensor
    kd: torch.Tensor
    at: torch.Tensor


class FeatureCapture:
    """Capture selected module outputs during forward passes.

    Used for attention-transfer KD. It is intentionally lightweight and only
    stores detached teacher features and normal student features for the current
    batch. Call clear() before every batch.
    """

    def __init__(self, model: nn.Module, module_names: Sequence[str], detach: bool = False) -> None:
        self.model = model
        self.module_names = list(module_names)
        self.detach = bool(detach)
        self.features: list[torch.Tensor] = []
        name_to_module = dict(model.named_modules())
        missing = [name for name in self.module_names if name not in name_to_module]
        if missing:
            raise ValueError(f"modules not found for feature capture: {missing}")
        self.handles = [name_to_module[name].register_forward_hook(self._hook) for name in self.module_names]

    def _hook(self, _module: nn.Module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.features.append(output.detach() if self.detach else output)

    def clear(self) -> None:
        self.features.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attention_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Channel-agnostic spatial attention map. Works even when teacher/student
    # channel counts differ.
    a = x.pow(2).mean(dim=1, keepdim=True)
    a = a.flatten(1)
    return F.normalize(a, p=2, dim=1, eps=eps)


def attention_transfer_loss(student_features: Iterable[torch.Tensor], teacher_features: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = []
    for s, t in zip(student_features, teacher_features):
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.mse_loss(attention_map(s), attention_map(t)))
    if not losses:
        return torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    return torch.stack(losses).sum()


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    ce_criterion: nn.Module,
    alpha: float = 0.7,
    temperature: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy + standard logit distillation KL loss."""
    ce = ce_criterion(student_logits, targets)
    # Compute KL in fp32 for numerical stability when training with bf16/fp16 AMP.
    s_logits = student_logits.float()
    t_logits = teacher_logits.float()
    t = float(temperature)
    kd = F.kl_div(
        F.log_softmax(s_logits / t, dim=1),
        F.softmax(t_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)
    total = (1.0 - float(alpha)) * ce + float(alpha) * kd
    return total, ce, kd
