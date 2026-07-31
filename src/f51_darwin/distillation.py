from __future__ import annotations

import torch
from torch.nn import functional as F


def combine_teacher_logits(
    teacher_logits: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    if not teacher_logits:
        raise ValueError("teacher_logits must not be empty.")
    if weights is None:
        weights = [1.0 / len(teacher_logits)] * len(teacher_logits)
    if len(weights) != len(teacher_logits):
        raise ValueError("weights and teacher_logits must have the same length.")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must have positive sum.")
    normalized = [float(weight) / total_weight for weight in weights]
    combined = torch.zeros_like(teacher_logits[0])
    for weight, logits in zip(normalized, teacher_logits):
        combined = combined + logits * weight
    return combined


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature**2)

