from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from f51_darwin.darwin_x_core.model import DarwinXModel


ORGAN_ARMS = (
    "control",
    "gaba",
    "ihs",
    "heartbeat",
    "ttm",
    "spider",
    "jepa",
    "mtp",
)

_LANGUAGE_MARKERS = (
    "token_embedding",
    ".attention.",
    ".ffn.",
    ".norm1.",
    ".norm2.",
    "lm_head",
)


@dataclass(frozen=True)
class OrganSelection:
    arm: str
    named: tuple[tuple[str, torch.nn.Parameter], ...]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.named)


def language_brain_named_parameters(
    model: DarwinXModel,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    def is_language(name: str) -> bool:
        return (
            name == "token_embedding.weight"
            or name.startswith("norm.")
            or name.startswith("lm_head.")
            or ".attention." in name
            or ".ffn." in name
            or ".norm1." in name
            or ".norm2." in name
        )

    return tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if is_language(name)
    )


def build_answer_only_example(
    prompt_ids: torch.Tensor,
    *,
    answer_ids: Sequence[int],
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        prompt_ids.ndim != 2
        or prompt_ids.size(0) != 1
        or prompt_ids.size(1) == 0
    ):
        raise ValueError("prompt_ids must contain one non-empty sequence")
    answer = tuple(int(token) for token in answer_ids)
    if not answer:
        raise ValueError("answer_ids must not be empty")
    suffix = torch.tensor(
        [(*answer, int(eos_token_id))],
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    inputs = torch.cat((prompt_ids, suffix), dim=1)
    labels = torch.full_like(inputs, -100)
    labels[:, prompt_ids.size(1) :] = suffix
    return inputs, labels


def _module_parameters(
    prefix: str,
    module: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (f"{prefix}.{name}", parameter)
        for name, parameter in module.named_parameters()
    ]


def select_organ_parameters(
    model: DarwinXModel,
    arm: str,
) -> OrganSelection:
    if arm not in ORGAN_ARMS:
        raise ValueError(f"unknown organ arm: {arm}")
    model_named = list(model.named_parameters())
    selected: list[tuple[str, torch.nn.Parameter]]
    if arm == "control":
        selected = []
    elif arm == "gaba":
        selected = [
            (name, parameter)
            for name, parameter in model_named
            if ".gaba." in name
        ]
    elif arm == "ihs":
        selected = [
            (name, parameter)
            for name, parameter in model_named
            if name.startswith("inter_hemispheric")
        ]
    elif arm == "spider":
        selected = [
            (name, parameter)
            for name, parameter in model_named
            if name.startswith("_spider_sense_module.")
        ]
    elif arm == "jepa":
        selected = [
            (name, parameter)
            for name, parameter in model_named
            if name.startswith("jepa_predictor.")
        ]
    elif arm == "mtp":
        selected = [
            (name, parameter)
            for name, parameter in model_named
            if name.startswith("mtp_heads.")
        ]
    elif arm == "heartbeat":
        heartbeat = model.heartbeat
        selected = []
        if heartbeat is not None:
            selected.extend(
                _module_parameters("heartbeat.ff_stack", heartbeat.ff_stack)
            )
            selected.extend(
                _module_parameters("heartbeat.thinker", heartbeat.thinker)
            )
    else:
        selected = [
            (name, parameter)
            for name, parameter in model_named
            if name == "ttm_residual_gate"
        ]
        heartbeat = model.heartbeat
        if heartbeat is not None:
            selected.extend(
                _module_parameters(
                    "heartbeat.tt_memory",
                    heartbeat.tt_memory,
                )
            )

    unique: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for name, parameter in selected:
        identity = id(parameter)
        if identity in seen:
            continue
        if any(marker in name for marker in _LANGUAGE_MARKERS):
            raise ValueError(
                f"organ selection crossed into language brain: {name}"
            )
        unique.append((name, parameter))
        seen.add(identity)
    return OrganSelection(arm=arm, named=tuple(unique))


def shuffled_orders(
    item_ids: Sequence[str],
    *,
    seeds: Iterable[int],
) -> tuple[tuple[str, ...], ...]:
    base = tuple(item_ids)
    if len(set(base)) != len(base):
        raise ValueError("QA item IDs must be unique")
    orders = []
    for seed in seeds:
        current = list(base)
        random.Random(int(seed)).shuffle(current)
        orders.append(tuple(current))
    return tuple(orders)


def classify_arm(report: dict[str, Any]) -> str:
    if int(report.get("frozen_brain_drift", 0)) != 0:
        return "invalid_brain_drift"
    nonzero = int(report.get("nonzero_gradient_parameters", 0))
    state_changed = bool(report.get("state_changed", False))
    if nonzero == 0 and not state_changed:
        return "no_language_gradient"
    corrected = int(report.get("corrected", 0))
    forgotten = int(report.get("forgotten", 0))
    if corrected > 0 and forgotten == 0:
        return "improved"
    if corrected > 0 or forgotten > 0:
        return "changed_but_regressed"
    if state_changed:
        return "state_only"
    return "no_effect"
