from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.moe import DeepSeekStyleMoE
from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
from f51_darwin.organism.causal_bus import AblationArm, OrganCausalBus
from f51_darwin.organism.causal_ledger import CausalLedger
from f51_darwin.organism.config import DarwinOrganismConfig
from f51_darwin.organism.lifecycle import _DarwinLifecycleMixin
from f51_darwin.organism.support import restore_rng_state, rng_state_dict
from f51_darwin.organism.unified_mesh import run_protected_sleep
from f51_darwin.replay_buffer import ReplayBuffer


def _metrics(fresh: float, replay: float, heldout: float) -> dict[str, float]:
    return {
        "fresh": fresh,
        "replay": replay,
        "heldout": heldout,
    }


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _model_optimizer() -> tuple[nn.Linear, torch.optim.SGD]:
    torch.manual_seed(51)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_protected_sleep_accepts_improvement_across_all_three_metrics() -> None:
    model, optimizer = _model_optimizer()
    evaluations = iter(
        [_metrics(2.0, 3.0, 4.0), _metrics(1.9, 2.8, 3.9)]
    )
    organs = {"memory": [1]}

    def consolidate() -> None:
        with torch.no_grad():
            model.weight.add_(0.25)
        organs["memory"].append(2)

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        consolidate=consolidate,
        evaluate=lambda: next(evaluations),
        snapshot_organs=lambda: organs,
        restore_organs=lambda state: (
            organs.clear(),
            organs.update(state),
        ),
    )

    assert report.executed is True
    assert report.accepted is True
    assert report.reverted is False
    assert report.metrics_before == _metrics(2.0, 3.0, 4.0)
    assert report.metrics_after == _metrics(1.9, 2.8, 3.9)
    assert organs == {"memory": [1, 2]}


def test_protected_sleep_rolls_back_model_optimizer_rng_and_organs() -> None:
    model, optimizer = _model_optimizer()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = copy.deepcopy(rng_state_dict())
    organs = {"memory": [1], "pressure": 0.5}
    organs_before = copy.deepcopy(organs)
    evaluations = iter(
        [_metrics(2.0, 3.0, 4.0), _metrics(1.9, 3.1, 3.9)]
    )

    def regress() -> None:
        with torch.no_grad():
            model.weight.mul_(0.0)
        optimizer.param_groups[0]["lr"] = 9.0
        random.random()
        np.random.random()
        torch.rand(())
        organs["memory"].append(999)
        organs["pressure"] = 9.0

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        consolidate=regress,
        evaluate=lambda: next(evaluations),
        snapshot_organs=lambda: organs,
        restore_organs=lambda state: (
            organs.clear(),
            organs.update(state),
        ),
    )

    assert report.accepted is False
    assert report.reverted is True
    assert report.regressions == ("replay",)
    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert organs == organs_before

    actual_probes = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(rng_before)
    expected_probes = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual_probes == expected_probes


def test_protected_sleep_disabled_is_exact_noop() -> None:
    model, optimizer = _model_optimizer()
    before = copy.deepcopy(model.state_dict())
    calls: list[str] = []

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=False,
        consolidate=lambda: calls.append("consolidate"),
        evaluate=lambda: calls.append("evaluate"),
    )

    assert report.executed is False
    assert report.accepted is False
    assert report.reverted is False
    assert calls == []
    _assert_nested_equal(model.state_dict(), before)


def test_protected_sleep_exception_recovers_full_snapshot() -> None:
    model, optimizer = _model_optimizer()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    organs = {"memory": [1]}
    evaluations = iter([_metrics(1.0, 1.0, 1.0)])

    def explode() -> None:
        with torch.no_grad():
            model.bias.add_(10.0)
        optimizer.param_groups[0]["lr"] = 10.0
        organs["memory"].append(2)
        raise RuntimeError("sleep exploded")

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        consolidate=explode,
        evaluate=lambda: next(evaluations),
        snapshot_organs=lambda: organs,
        restore_organs=lambda state: (
            organs.clear(),
            organs.update(state),
        ),
    )

    assert report.reverted is True
    assert report.accepted is False
    assert report.error == "RuntimeError:sleep exploded"
    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert organs == {"memory": [1]}


def test_protected_sleep_evaluations_are_isolated_and_rng_paired() -> None:
    model, optimizer = _model_optimizer()
    organs = {"eval_calls": 0, "consolidated": False, "prepared": False}
    probes: list[tuple[float, float, float]] = []

    def prepare() -> None:
        organs["prepared"] = True

    def evaluate() -> dict[str, float]:
        organs["eval_calls"] += 1
        probe = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        probes.append(probe)
        return _metrics(*probe)

    def consolidate() -> None:
        organs["consolidated"] = True
        with torch.no_grad():
            model.weight.add_(0.1)

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        prepare=prepare,
        consolidate=consolidate,
        evaluate=evaluate,
        snapshot_organs=lambda: organs,
        restore_organs=lambda state: (
            organs.clear(),
            organs.update(state),
        ),
    )

    assert report.accepted is True
    assert probes[0] == probes[1]
    assert organs == {
        "eval_calls": 0,
        "consolidated": True,
        "prepared": True,
    }


@pytest.mark.parametrize("regress", [False, True])
def test_protected_sleep_replay_sampling_rng_is_transactional(
    regress: bool,
) -> None:
    model, optimizer = _model_optimizer()
    replay = ReplayBuffer(capacity=8, seed=51)
    for token in range(6):
        replay.add([token])
    initial_state = copy.deepcopy(replay.state_dict())
    expected_accepted = ReplayBuffer(capacity=8, seed=0)
    expected_accepted.load_state_dict(initial_state)
    expected_accepted.sample(2)
    evaluations = iter(
        [
            _metrics(1.0, 1.0, 1.0),
            _metrics(1.0, 1.1 if regress else 1.0, 1.0),
        ]
    )

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        prepare=lambda: replay.sample(2),
        consolidate=lambda: None,
        evaluate=lambda: next(evaluations),
        snapshot_organs=lambda: replay.state_dict(),
        restore_organs=replay.load_state_dict,
    )

    expected = (
        initial_state if regress else expected_accepted.state_dict()
    )
    _assert_nested_equal(replay.state_dict(), expected)
    assert report.reverted is regress
    assert report.accepted is (not regress)


def test_protected_sleep_reports_original_and_partial_rollback_failure() -> None:
    model, optimizer = _model_optimizer()
    restore_calls = 0

    def explode() -> None:
        raise RuntimeError("sleep exploded")

    def fail_restore(_state) -> None:
        nonlocal restore_calls
        restore_calls += 1
        if restore_calls > 1:
            raise ValueError("organ restore failed")

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        consolidate=explode,
        evaluate=lambda: _metrics(1.0, 1.0, 1.0),
        snapshot_organs=lambda: {},
        restore_organs=fail_restore,
    )

    assert report.accepted is False
    assert report.reverted is False
    assert report.rollback_partial is True
    assert report.error == "RuntimeError:sleep exploded"
    assert report.rollback_error == "ValueError:organ restore failed"


class _HeartbeatModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.5)
        self.heartbeat_slot = torch.tensor([3.0])

    def heartbeat_state_dict(self):
        return {"slot": self.heartbeat_slot.clone()}

    def load_heartbeat_state_dict(self, state):
        self.heartbeat_slot = state["slot"].clone()


def test_rollback_restores_external_heartbeat_and_mixed_training_modes() -> None:
    model = _HeartbeatModel()
    model.train()
    model.dropout.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    evaluations = iter(
        [_metrics(1.0, 1.0, 1.0), _metrics(1.1, 1.0, 1.0)]
    )

    def regress() -> None:
        model.eval()
        model.dropout.train()
        model.heartbeat_slot.add_(10.0)

    report = run_protected_sleep(
        model,
        optimizer,
        enabled=True,
        consolidate=regress,
        evaluate=lambda: next(evaluations),
    )

    assert report.reverted is True
    assert torch.equal(model.heartbeat_slot, torch.tensor([3.0]))
    assert model.training is True
    assert model.dropout.training is False


def test_heartbeat_state_roundtrip_removes_extra_explorer_domains() -> None:
    heartbeat = Heartbeat(
        8,
        HeartbeatConfig(
            think_interval=100,
            explore_interval=100,
            self_reward_interval=100,
            ff_layers=1,
            memory_capacity=4,
        ),
    )
    heartbeat.explorer.register_domain("original", difficulty=0.75)
    heartbeat.explorer.explored["original"].questions = ["q1"]
    heartbeat.explorer.explored["original"].discoveries = 2
    heartbeat.explorer._history.append({"domain": "original", "questions": 1})
    state = copy.deepcopy(heartbeat.state_dict())

    heartbeat.explorer.register_domain("protected_sleep_fresh")
    heartbeat.explorer.explored["original"].questions.append("mutated")
    heartbeat.explorer._history.append({"domain": "extra", "questions": 0})
    heartbeat.load_state_dict(state)

    assert set(heartbeat.explorer.explored) == {"original"}
    target = heartbeat.explorer.explored["original"]
    assert target.difficulty == pytest.approx(0.75)
    assert target.questions == ["q1"]
    assert target.discoveries == 2
    assert heartbeat.explorer._history == [
        {"domain": "original", "questions": 1}
    ]
    _assert_nested_equal(heartbeat.state_dict(), state)


@pytest.mark.parametrize("value", [-0.1, float("inf"), True])
def test_sleep_regression_threshold_is_explicit_and_validated(value) -> None:
    with pytest.raises(ValueError, match="sleep_max_regression"):
        DarwinOrganismConfig(sleep_max_regression=value)

    assert DarwinOrganismConfig(
        sleep_max_regression=0.05
    ).sleep_max_regression == pytest.approx(0.05)


class _SleepPressure:
    def state(self):
        return {"cortisol": 0.0, "bdnf": [0.0], "ach": 0.0}

    def should_sleep(self):
        return False


class _DisabledSleepModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        moe = nn.Module()
        moe.neuroendocrine = _SleepPressure()
        self.blocks = [SimpleNamespace(moe=moe)]


class _NoSampleReplay:
    def __len__(self):
        return 1

    def sample(self, size):
        raise AssertionError("disabled sleep must not sample replay")


def test_disabled_sleep_does_not_advance_replay_before_snapshot() -> None:
    organism = _DarwinLifecycleMixin.__new__(_DarwinLifecycleMixin)
    organism.model = _DisabledSleepModel()
    organism.optimizer = torch.optim.SGD(
        organism.model.parameters(), lr=0.1
    )
    organism.cfg = SimpleNamespace(sleep_max_regression=0.0)
    organism.replay = _NoSampleReplay()
    report: dict = {}

    organism._sleep_cycle(report)

    assert report["sleep"]["executed"] is False
    assert "degraded_organs" not in report


def test_sleep_organ_snapshot_restores_dae_and_autonomic_accumulators() -> None:
    organism = _DarwinLifecycleMixin.__new__(_DarwinLifecycleMixin)
    moe = nn.Module()
    moe._dae_observation_accumulator = {"count": 2}
    moe._autonomic_usage_accumulator = torch.tensor([0.25, 0.75])
    moe._autonomic_entropy_sum = 1.25
    moe._autonomic_entropy_observations = 3
    moe._autonomic_forward_uses = 4
    moe._replay_buffer = [torch.tensor([1, 2])]
    moe._replay_count = 7
    moe._signature_inputs = [torch.tensor([3.0])]
    moe._signature_buffer = [torch.tensor([4.0])]
    moe._ghost_predator_buffer = [torch.tensor([5.0])]
    organism.model = SimpleNamespace(
        blocks=[SimpleNamespace(moe=moe)]
    )
    organism.replay = None

    snapshot = organism._snapshot_sleep_organs()
    moe._dae_observation_accumulator["count"] = 99
    moe._autonomic_usage_accumulator.zero_()
    moe._autonomic_entropy_sum = 0.0
    moe._autonomic_entropy_observations = 0
    moe._autonomic_forward_uses = 0
    moe._replay_buffer.append(torch.tensor([9, 9]))
    moe._replay_count = 99
    moe._signature_inputs.append(torch.tensor([9.0]))
    moe._signature_buffer.clear()
    moe._ghost_predator_buffer.clear()

    organism._restore_sleep_organs(snapshot)

    assert moe._dae_observation_accumulator == {"count": 2}
    assert torch.equal(
        moe._autonomic_usage_accumulator,
        torch.tensor([0.25, 0.75]),
    )
    assert moe._autonomic_entropy_sum == pytest.approx(1.25)
    assert moe._autonomic_entropy_observations == 3
    assert moe._autonomic_forward_uses == 4
    assert len(moe._replay_buffer) == 1
    assert torch.equal(moe._replay_buffer[0], torch.tensor([1, 2]))
    assert moe._replay_count == 7
    assert len(moe._signature_inputs) == 1
    assert torch.equal(moe._signature_inputs[0], torch.tensor([3.0]))
    assert len(moe._signature_buffer) == 1
    assert torch.equal(moe._signature_buffer[0], torch.tensor([4.0]))
    assert len(moe._ghost_predator_buffer) == 1
    assert torch.equal(
        moe._ghost_predator_buffer[0], torch.tensor([5.0])
    )


def test_real_moe_runtime_state_roundtrip_restores_forward_side_buffers() -> None:
    config = DarwinXConfig(
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        ghost_enabled=False,
    )
    moe = DeepSeekStyleMoE(config)
    moe._replay_buffer = [torch.tensor([1, 2])]
    moe._replay_count = 3
    moe._signature_inputs = [torch.tensor([4.0])]
    moe._signature_buffer = [torch.tensor([5.0])]
    moe._ghost_predator_buffer = [torch.tensor([6.0])]
    snapshot = moe.runtime_state_dict()

    moe._replay_buffer.append(torch.tensor([9, 9]))
    moe._replay_count = 99
    moe._signature_inputs.clear()
    moe._signature_buffer.clear()
    moe._ghost_predator_buffer.clear()
    moe.load_runtime_state_dict(snapshot)

    assert len(moe._replay_buffer) == 1
    assert torch.equal(moe._replay_buffer[0], torch.tensor([1, 2]))
    assert moe._replay_count == 3
    assert torch.equal(moe._signature_inputs[0], torch.tensor([4.0]))
    assert torch.equal(moe._signature_buffer[0], torch.tensor([5.0]))
    assert torch.equal(
        moe._ghost_predator_buffer[0], torch.tensor([6.0])
    )


@pytest.mark.parametrize(
    ("proposal", "expected_detail", "terminal_status", "fine_experts"),
    [
        (
            {
                "neurogenesis": True,
                "prune_experts": [],
                "expand_experts": [],
            },
            "proposed:create:B0",
            "birth",
            2,
        ),
        (
            {
                "neurogenesis": False,
                "prune_experts": [3],
                "expand_experts": [],
            },
            "proposed:prune:B0_E3",
            "apoptosis",
            4,
        ),
    ],
)
def test_real_moe_structural_terminal_events_keep_proposal_correlatable(
    proposal: dict,
    expected_detail: str,
    terminal_status: str,
    fine_experts: int,
) -> None:
    config = DarwinXConfig(
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        fine_experts=fine_experts,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        ghost_enabled=False,
    )
    moe = DeepSeekStyleMoE(config)
    moe._structural_event_log.append(
        {
            "status": "proposed_not_applied",
            "plasticity": {},
            **proposal,
        }
    )

    result = moe.execute_structural_actions()

    assert any(
        event.get("status") == terminal_status
        for event in moe._structural_event_log
    )
    details = _DarwinLifecycleMixin._structural_log_details(
        SimpleNamespace(blocks=[SimpleNamespace(moe=moe)]),
        status="executed",
    )
    assert details == (expected_detail,)
    assert sum(result.values()) == 1


def test_real_moe_rejects_prune_at_floor_without_false_execution() -> None:
    config = DarwinXConfig(
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        ghost_enabled=False,
    )
    moe = DeepSeekStyleMoE(config)
    for expert_index in (3, 2):
        moe._prune_expert(expert_index)
        assert moe._apoptosis(expert_index) is True
    assert len(moe.fine_experts) == 2
    moe._structural_event_log.clear()
    proposal = {
        "status": "proposed_not_applied",
        "plasticity": {},
        "neurogenesis": False,
        "prune_experts": [1],
        "expand_experts": [],
    }
    moe._structural_event_log.append(proposal)

    result = moe.execute_structural_actions()

    assert result == {"neurogenesis": 0, "pruned": 0, "expanded": 0}
    assert len(moe.fine_experts) == 2
    assert proposal["status"] == "rejected_not_applied"
    assert proposal["rejection_reason"] == "minimum_expert_floor"


def test_protected_eval_disables_heartbeat_when_forward_supports_it() -> None:
    heartbeat_values: list[bool | None] = []

    class Model(nn.Module):
        def forward(
            self,
            input_ids,
            labels=None,
            *,
            domain="train",
            heartbeat=None,
        ):
            heartbeat_values.append(heartbeat)
            return SimpleNamespace(loss=torch.tensor(1.0))

    organism = _DarwinLifecycleMixin.__new__(_DarwinLifecycleMixin)
    organism.model = Model()
    batch = torch.ones((1, 2), dtype=torch.long)

    organism._protected_eval_forward(
        batch,
        domain="protected_sleep_fresh",
    )

    assert heartbeat_values == [False]


class _Neuroendocrine:
    def __init__(self, *, proposed: bool = True) -> None:
        self.proposed = proposed

    def state(self):
        return {"cortisol": 0.5, "bdnf": [0.1], "ach": 0.2}

    def plasticity_decision(self):
        return {"expand": [0] if self.proposed else []}


class _StructuralModel(nn.Module):
    def __init__(self, *, proposed: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.blocks = [
            SimpleNamespace(
                moe=SimpleNamespace(
                    neuroendocrine=_Neuroendocrine(proposed=proposed)
                )
            )
        ]
        self.structural_calls = 0
        self.blocks[0].moe._structural_event_log = []
        if proposed:
            self.blocks[0].moe._structural_event_log.append(
                {
                    "status": "proposed_not_applied",
                    "expand_experts": [0],
                    "prune_experts": [],
                    "neurogenesis": False,
                }
            )

    def execute_structural_actions(self):
        self.structural_calls += 1
        if self.blocks[0].moe._structural_event_log:
            self.blocks[0].moe._structural_event_log[-1]["status"] = "executed"
            return [
                {
                    "pruned": 0,
                    "expanded": 1,
                    "neurogenesis": 0,
                }
            ]
        return []


def _structural_harness(
    arm: AblationArm | None,
    *,
    proposed: bool = True,
):
    organism = _DarwinLifecycleMixin.__new__(_DarwinLifecycleMixin)
    organism.model = _StructuralModel(proposed=proposed)
    organism.optimizer = torch.optim.SGD(
        organism.model.parameters(), lr=0.1
    )
    organism.cfg = SimpleNamespace(
        causal_mode="disabled" if arm is None else arm.value.lower(),
        optimizer_name="adamw",
        weight_decay=0.0,
    )
    organism.causal_bus = (
        None if arm is None else OrganCausalBus(arm=arm, adapters=())
    )
    organism.causal_ledger = None
    organism.cycle = 1
    organism.total_steps = 1
    organism.base_checkpoint_id = "base"
    organism.training_contract_id = "contract"
    organism._apply_lr_schedule = lambda _step: None
    return organism


@pytest.mark.parametrize(
    ("arm", "expected_calls"),
    [
        (None, 1),
        (AblationArm.CONTROL, 0),
        (AblationArm.SHADOW, 0),
        (AblationArm.APPLY, 1),
    ],
)
def test_structural_actions_require_effective_causal_intervention_when_enabled(
    arm: AblationArm | None,
    expected_calls: int,
) -> None:
    organism = _structural_harness(arm)
    report: dict = {}

    organism._execute_structural_boundary(report)

    assert organism.model.structural_calls == expected_calls
    assert report["structural"]["authorized"] is (expected_calls == 1)


def test_causal_boundary_rejects_real_prune_at_floor_before_mutation() -> None:
    config = DarwinXConfig(
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=1,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        ghost_enabled=False,
    )
    moe = DeepSeekStyleMoE(config)
    for expert_index in (3, 2):
        moe._prune_expert(expert_index)
        assert moe._apoptosis(expert_index) is True
    moe._structural_event_log.clear()
    moe._structural_event_log.append(
        {
            "status": "proposed_not_applied",
            "plasticity": {},
            "neurogenesis": False,
            "prune_experts": [1],
            "expand_experts": [],
        }
    )
    moe.neuroendocrine.plasticity_decision = lambda: {"prune": [1]}
    organism = _structural_harness(AblationArm.APPLY)
    organism.model.blocks[0].moe = moe
    calls: list[str] = []
    organism.model.execute_structural_actions = (
        lambda: calls.append("execute")
    )
    report: dict = {}

    organism._execute_structural_boundary(report)

    assert calls == []
    assert len(moe.fine_experts) == 2
    assert report["structural"]["authorized"] is False
    assert report["structural"]["error"] == "structural_preflight_rejected"
    assert report["structural"]["preflight_errors"] == [
        "B0:minimum_expert_floor"
    ]


@pytest.mark.parametrize("terminal_status", ["birth", "apoptosis"])
def test_structural_validation_finds_executed_proposal_before_terminal_event(
    terminal_status: str,
) -> None:
    organism = _structural_harness(AblationArm.APPLY)
    original_execute = organism.model.execute_structural_actions

    def execute_with_terminal_event():
        result = original_execute()
        organism.model.blocks[0].moe._structural_event_log.append(
            {"status": terminal_status, "step": 1}
        )
        return result

    organism.model.execute_structural_actions = execute_with_terminal_event
    report: dict = {}

    organism._execute_structural_boundary(report)

    assert report["structural"]["authorized"] is True
    assert organism.model.structural_calls == 1


def test_apply_without_effective_structural_proposal_cannot_mutate() -> None:
    organism = _structural_harness(AblationArm.APPLY, proposed=False)
    report: dict = {}

    organism._execute_structural_boundary(report)

    assert organism.model.structural_calls == 0
    assert report["structural"]["authorized"] is False
    assert report["structural"]["effective_intervention_ids"] == []


def test_structural_authorization_rejects_pending_action_mismatch() -> None:
    organism = _structural_harness(AblationArm.APPLY)
    organism.model.blocks[0].moe._structural_event_log[-1] = {
        "status": "proposed_not_applied",
        "expand_experts": [],
        "prune_experts": [9],
        "neurogenesis": False,
    }
    report: dict = {}

    organism._execute_structural_boundary(report)

    assert organism.model.structural_calls == 0
    assert report["structural"]["authorized"] is False
    assert report["structural"]["error"] == "pending_actions_mismatch"


def test_structural_success_outcome_is_recorded_only_after_remap(
    monkeypatch,
) -> None:
    organism = _structural_harness(AblationArm.APPLY)
    events: list[tuple[str, str | None, tuple[str, ...]]] = []

    class Ledger:
        def record_intent(self, identity, decisions):
            events.append(("intent", None, ()))

        def record_outcome(self, outcome):
            events.append(
                (
                    "outcome",
                    outcome.error,
                    tuple(outcome.accepted_intervention_ids),
                )
            )

    organism.causal_ledger = Ledger()
    monkeypatch.setattr(
        "f51_darwin.organism.lifecycle.build_optimizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("optimizer rebuild failed")
        ),
    )

    with pytest.raises(RuntimeError, match="optimizer rebuild failed"):
        organism._execute_structural_boundary({})

    assert events[0][0] == "intent"
    assert events[1] == (
        "outcome",
        "structural_error:RuntimeError:optimizer rebuild failed",
        (),
    )
    assert len(events) == 2


def test_authorized_structural_boundary_closes_ledger_attempt(tmp_path) -> None:
    organism = _structural_harness(AblationArm.APPLY)
    organism.causal_ledger = CausalLedger(tmp_path / "structural.jsonl")

    organism._execute_structural_boundary({})

    verification = organism.causal_ledger.verify()
    assert verification.valid is True
    assert verification.event_count == 2
    assert verification.orphan_intent_count == 0
