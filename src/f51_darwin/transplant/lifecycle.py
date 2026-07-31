from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import torch

from .recipient import TwoDonorRecipient


class OrganState(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ADAPTER_ACTIVE = "adapter_active"
    ORGAN_UNFROZEN = "organ_unfrozen"
    ACTIVE = "active"
    FROZEN = "frozen"
    QUARANTINE = "quarantine"


_ALLOWED_TRANSITIONS: dict[OrganState, frozenset[OrganState]] = {
    OrganState.CANDIDATE: frozenset(
        {OrganState.SHADOW, OrganState.QUARANTINE}
    ),
    OrganState.SHADOW: frozenset(
        {OrganState.ADAPTER_ACTIVE, OrganState.QUARANTINE}
    ),
    OrganState.ADAPTER_ACTIVE: frozenset(
        {OrganState.ORGAN_UNFROZEN, OrganState.QUARANTINE}
    ),
    OrganState.ORGAN_UNFROZEN: frozenset(
        {OrganState.ACTIVE, OrganState.FROZEN, OrganState.QUARANTINE}
    ),
    OrganState.ACTIVE: frozenset(
        {OrganState.FROZEN, OrganState.QUARANTINE}
    ),
    OrganState.FROZEN: frozenset(
        {OrganState.SHADOW, OrganState.QUARANTINE}
    ),
    OrganState.QUARANTINE: frozenset(),
}


def transition_organ(current: OrganState, target: OrganState) -> OrganState:
    current = OrganState(current)
    target = OrganState(target)
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(
            f"invalid organ transition: {current.value} -> {target.value}"
        )
    return target


def configure_trainable_state(
    recipient: TwoDonorRecipient,
    *,
    organ_name: str,
    state: OrganState,
) -> frozenset[str]:
    return configure_trainable_states(
        recipient,
        organ_states={organ_name: state},
    )


def configure_trainable_states(
    recipient: TwoDonorRecipient,
    *,
    organ_states: Mapping[str, OrganState],
) -> frozenset[str]:
    resolved = {
        name: OrganState(state)
        for name, state in organ_states.items()
    }
    slots = {
        name: recipient.organ_slot(name)
        for name in resolved
    }

    for parameter in recipient.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    declared_parameters: set[int] = set()
    for name, state in resolved.items():
        slot = slots[name]
        if state in {OrganState.ADAPTER_ACTIVE, OrganState.ORGAN_UNFROZEN}:
            declared_parameters.update(
                id(parameter) for parameter in slot.adapter.parameters()
            )
            external_gate = getattr(slot, "external_gate", None)
            if external_gate is not None:
                declared_parameters.add(id(external_gate))
        if state is OrganState.ORGAN_UNFROZEN:
            declared_parameters.update(
                id(parameter) for parameter in slot.organ.parameters()
            )

    declared_names: set[str] = set()
    for name, parameter in recipient.named_parameters():
        enabled = id(parameter) in declared_parameters
        parameter.requires_grad_(enabled)
        if enabled:
            declared_names.add(name)

    return frozenset(declared_names)


def audit_gradients(
    recipient: TwoDonorRecipient,
    declared: frozenset[str],
) -> None:
    known = {name for name, _ in recipient.named_parameters()}
    unknown = set(declared) - known
    if unknown:
        raise ValueError(f"unknown declared parameters: {sorted(unknown)}")

    for name, parameter in recipient.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if name not in declared:
            raise RuntimeError(f"undeclared gradient: {name}")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite gradient: {name}")
