from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from f51_darwin.curiosity import CuriosityMemory
from f51_darwin.organism.causal_bus import StepIdentity
from f51_darwin.organism.organ_adapters import CuriosityPriorityAdapter
from f51_darwin.organism.organ_frames import OrganObservationFrame


def _identity(step: int) -> StepIdentity:
    return StepIdentity(
        run_id="curiosity-run",
        cycle=2,
        optimizer_step=step,
        accumulation_window=0,
        batch_digest=f"batch-{step}",
        rng_digest=f"rng-{step}",
        base_checkpoint_id="base-curiosity",
        training_contract_id="contract-curiosity",
    )


def _priority_frame(
    *,
    source_step: int = 10,
    valid_through_step: int = 11,
) -> OrganObservationFrame:
    return OrganObservationFrame.create(
        source=_identity(source_step),
        signals={
            "curiosity": {
                "sample_ids": ("row-a", "row-b"),
                "priorities": (0.75, 1.75),
                "observation_id": "curiosity-observation-v1:test",
            }
        },
        valid_from_step=source_step + 1,
        valid_through_step=valid_through_step,
    )


def test_curiosity_is_batch_permutation_independent() -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=8)
    hidden = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
        ]
    )

    a = memory.observe(hidden, torch.tensor([0.2, 0.7]), ("a", "b"))
    b = memory.observe(
        hidden.flip(0),
        torch.tensor([0.7, 0.2]),
        ("b", "a"),
    )

    assert a.novelty == tuple(reversed(b.novelty))
    assert a.priority == tuple(reversed(b.priority))
    assert memory.bank_size == 0


def test_observe_is_pure_and_observation_is_frozen_detached() -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=8)
    before = copy.deepcopy(memory.state_dict())
    hidden = torch.randn(2, 3, 4, requires_grad=True)

    observation = memory.observe(
        hidden,
        torch.tensor([0.25, 0.5]),
        ("a", "b"),
    )

    assert memory.state_dict() == before
    assert observation.representations.device.type == "cpu"
    assert observation.representations.dtype == torch.float32
    assert observation.representations.requires_grad is False
    assert torch.isfinite(observation.representations).all()
    with pytest.raises(FrozenInstanceError):
        observation.sample_ids = ("changed",)


def test_curiosity_bank_updates_only_after_commit() -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=8)
    hidden = torch.ones(1, 2, 4)

    observation = memory.observe(hidden, torch.tensor([0.5]), ("sample",))

    assert memory.bank_size == 0
    memory.commit(observation)
    assert memory.bank_size == 1
    next_observation = memory.observe(
        hidden,
        torch.tensor([0.5]),
        ("sample-2",),
    )
    assert next_observation.novelty[0] < observation.novelty[0]


def test_commit_rejects_duplicate_and_tensor_tamper() -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=8)
    duplicate = memory.observe(
        torch.ones(1, 2, 4),
        torch.tensor([0.5]),
        ("sample",),
    )
    memory.commit(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        memory.commit(duplicate)

    fresh_memory = CuriosityMemory(d_model=4, bank_capacity=8)
    tampered = fresh_memory.observe(
        torch.ones(1, 2, 4),
        torch.tensor([0.5]),
        ("tampered",),
    )
    tampered.representations[0, 0] = 99.0
    with pytest.raises(ValueError, match="tamper|digest"):
        fresh_memory.commit(tampered)
    assert fresh_memory.bank_size == 0


def test_persistent_surprise_is_downweighted_as_noise() -> None:
    memory = CuriosityMemory(
        d_model=4,
        bank_capacity=8,
        noisy_repeat_limit=3,
    )
    hidden = torch.ones(1, 2, 4)
    priorities = []
    for index in range(5):
        observation = memory.observe(
            hidden,
            torch.tensor([9.0]),
            (f"same-{index}",),
            content_digests=("same-content",),
        )
        priorities.append(observation.priority[0])
        memory.commit(observation)

    assert priorities[-1] < priorities[1]
    assert all(0.5 <= value <= 2.0 for value in priorities)


def test_recorded_improvement_prevents_noisy_tv_decay() -> None:
    memory = CuriosityMemory(
        d_model=4,
        bank_capacity=8,
        noisy_repeat_limit=2,
    )
    hidden = torch.ones(1, 2, 4)
    for index in range(3):
        observation = memory.observe(
            hidden,
            torch.tensor([4.0]),
            (f"repeat-{index}",),
            content_digests=("learnable-content",),
        )
        memory.commit(observation, loss_improved=index == 2)

    after_improvement = memory.observe(
        hidden,
        torch.tensor([4.0]),
        ("after-improvement",),
        content_digests=("learnable-content",),
    )
    assert after_improvement.priority[0] > 0.5


def test_curiosity_state_roundtrip_is_exact_and_tamper_evident() -> None:
    source = CuriosityMemory(d_model=4, bank_capacity=3)
    first = source.observe(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        torch.tensor([2.5]),
        ("first",),
        content_digests=("content",),
    )
    source.commit(first)
    state = source.state_dict()

    restored = CuriosityMemory(d_model=4, bank_capacity=3)
    restored.load_state_dict(copy.deepcopy(state))

    assert restored.state_dict() == state
    assert restored.observe(
        torch.ones(1, 1, 4),
        torch.tensor([1.0]),
        ("next",),
    ) == source.observe(
        torch.ones(1, 1, 4),
        torch.tensor([1.0]),
        ("next",),
    )

    tampered = copy.deepcopy(state)
    tampered["bank"][0][0] = 99.0
    with pytest.raises(ValueError, match="digest"):
        CuriosityMemory(d_model=4, bank_capacity=3).load_state_dict(tampered)


@pytest.mark.parametrize(
    ("hidden", "error"),
    [
        (torch.full((1, 1, 4), float("nan")), torch.tensor([1.0])),
        (torch.ones(1, 1, 4), torch.tensor([float("inf")])),
    ],
)
def test_curiosity_rejects_nonfinite_observations(
    hidden: torch.Tensor,
    error: torch.Tensor,
) -> None:
    memory = CuriosityMemory(d_model=4, bank_capacity=3)
    with pytest.raises(ValueError, match="finite"):
        memory.observe(hidden, error, ("sample",))
    assert memory.bank_size == 0


def test_curiosity_adapter_consumes_only_exact_next_step_frame() -> None:
    adapter = CuriosityPriorityAdapter()
    frame = _priority_frame(valid_through_step=12)

    priorities = adapter.consume(frame=frame, target=_identity(11))

    assert dict(priorities) == {"row-a": 0.75, "row-b": 1.75}
    assert not hasattr(priorities, "loss_scale")
    with pytest.raises(ValueError, match="temporal_barrier"):
        adapter.consume(frame=frame, target=_identity(10))
    with pytest.raises(ValueError, match="exact_next_step"):
        adapter.consume(frame=frame, target=_identity(12))


def test_curiosity_adapter_rejects_tampered_or_invalid_priority_frame() -> None:
    adapter = CuriosityPriorityAdapter()
    frame = _priority_frame()
    with pytest.raises(ValueError, match="digest"):
        adapter.consume(
            frame=replace(frame, digest="organ-frame-v1:tampered"),
            target=_identity(11),
        )

    invalid = OrganObservationFrame.create(
        source=_identity(20),
        signals={
            "curiosity": {
                "sample_ids": ("row-a",),
                "priorities": (2.5,),
                "observation_id": "curiosity-observation-v1:test",
            }
        },
        valid_from_step=21,
        valid_through_step=21,
    )
    with pytest.raises(ValueError, match="priority"):
        adapter.consume(frame=invalid, target=_identity(21))
