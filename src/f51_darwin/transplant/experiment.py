from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from .lifecycle import OrganState, configure_trainable_state
from .recipient import TwoDonorRecipient


@dataclass(frozen=True)
class ArmBudget:
    tokens: int
    steps: int
    seed: int
    starts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.tokens <= 0 or self.steps <= 0:
            raise ValueError("arm tokens and steps must be positive")
        if not self.starts or any(start < 0 for start in self.starts):
            raise ValueError("arm starts must be non-empty and non-negative")

    @property
    def identity(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ArmResult:
    arm: str
    budget_identity: str
    train_nll: float
    holdout_nll: float
    elapsed_seconds: float
    tokens_processed: int


@dataclass
class ExperimentalArm:
    name: str
    recipient: TwoDonorRecipient
    organ_mode: str
    declared_parameters: frozenset[str]
    budget: ArmBudget


def validate_matched_arms(arms: Mapping[str, ArmBudget]) -> None:
    if set(arms) != {"A", "B", "C", "D"}:
        raise ValueError("matched-arm budget mismatch: expected A/B/C/D")
    identities = {budget.identity for budget in arms.values()}
    if len(identities) != 1:
        raise ValueError("matched-arm budget mismatch")


def _assert_independent_storage(
    arms: Mapping[str, ExperimentalArm],
) -> None:
    storage_by_arm: dict[str, set[int]] = {}
    for name, arm in arms.items():
        storage_by_arm[name] = {
            parameter.data_ptr() for parameter in arm.recipient.parameters()
        }
    ordered = tuple(sorted(storage_by_arm))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            shared = storage_by_arm[left] & storage_by_arm[right]
            if shared:
                raise RuntimeError(
                    f"shared parameter storage between arms {left} and {right}"
                )


def build_first_slice_arms(
    recipient_factory: Callable[[], TwoDonorRecipient],
    *,
    organ_name: str,
    budgets: Mapping[str, ArmBudget],
) -> dict[str, ExperimentalArm]:
    validate_matched_arms(budgets)
    base = recipient_factory()
    recipients = {
        name: copy.deepcopy(base)
        for name in ("A", "B", "C", "D")
    }

    declared_a = configure_trainable_state(
        recipients["A"],
        organ_name=organ_name,
        state=OrganState.CANDIDATE,
    )

    configure_trainable_state(
        recipients["B"],
        organ_name=organ_name,
        state=OrganState.SHADOW,
    )
    declared_b: set[str] = set()
    for name, parameter in recipients["B"].named_parameters():
        if name.startswith("language_model."):
            parameter.requires_grad_(True)
            declared_b.add(name)

    declared_c = configure_trainable_state(
        recipients["C"],
        organ_name=organ_name,
        state=OrganState.ADAPTER_ACTIVE,
    )
    declared_d = configure_trainable_state(
        recipients["D"],
        organ_name=organ_name,
        state=OrganState.ORGAN_UNFROZEN,
    )

    arms = {
        "A": ExperimentalArm(
            "A", recipients["A"], "disabled", declared_a, budgets["A"]
        ),
        "B": ExperimentalArm(
            "B",
            recipients["B"],
            "shadow",
            frozenset(declared_b),
            budgets["B"],
        ),
        "C": ExperimentalArm(
            "C", recipients["C"], "adapter_active", declared_c, budgets["C"]
        ),
        "D": ExperimentalArm(
            "D", recipients["D"], "organ_unfrozen", declared_d, budgets["D"]
        ),
    }
    _assert_independent_storage(arms)
    return arms

