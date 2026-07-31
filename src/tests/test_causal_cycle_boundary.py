from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.organism.control import DarwinOrganism
from f51_darwin.organism.lifecycle import _DarwinLifecycleMixin
from f51_darwin.replay_buffer import ReplayBuffer
from f51_darwin.organism.unified_mesh import (
    UnifiedControlMesh,
    execute_evolution_actions,
    run_sleep_cycle,
)


class _Neuroendocrine:
    def __init__(self, decisions=None, *, should_sleep=True):
        self._decisions = decisions or {
            "prune": [0],
            "quarantine": [1],
            "expand": [2],
            "create": False,
        }
        self._should_sleep = should_sleep

    def state(self):
        return {
            "cortisol": 0.4,
            "bdnf_per_expert": [0.2, 0.6],
            "ach_pressure": 0.3,
        }

    def plasticity_decision(self):
        return self._decisions

    def should_sleep(self):
        return self._should_sleep


class _TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.001, 0.5]))
        self.neuroendocrine = _Neuroendocrine()
        self._expert_usage_buffer = torch.tensor([0.99, 0.01])


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"moe": _TinyMoE()})]
        )
        self.heartbeat = SimpleNamespace(dopamine=0.7)


def test_collect_signals_uses_real_neuroendocrine_schema():
    model = _TinyModel()
    output = SimpleNamespace(
        spider_confidence=torch.tensor([0.8]),
        heartbeat_stats={
            "dopamine": 0.6,
            "memory": {"avg_surprise": 0.25},
        },
        decision_factors={"confidence": 0.9},
    )

    signals = UnifiedControlMesh(model).collect_signals(output)

    assert signals.spider_confidence == pytest.approx(0.8)
    assert signals.neuroendocrine_cortisol == pytest.approx(0.4)
    assert signals.neuroendocrine_bdnf == pytest.approx(0.4)
    assert signals.neuroendocrine_ach_pressure == pytest.approx(0.3)
    assert signals.heartbeat_dopamine == pytest.approx(0.6)
    assert signals.heartbeat_ttm_surprise == pytest.approx(0.25)
    assert signals.decision_confidence == pytest.approx(0.9)
    assert signals.expert_usage_gini == pytest.approx(0.49)


def test_legacy_mesh_apis_only_propose_and_never_mutate_weights():
    model = _TinyModel()
    before = model.blocks[0]["moe"].weight.detach().clone()

    sleep = run_sleep_cycle(model, UnifiedControlMesh(model).collect_signals(None))
    evolution = execute_evolution_actions(
        model, UnifiedControlMesh(model).collect_signals(None)
    )

    assert torch.equal(model.blocks[0]["moe"].weight, before)
    assert sleep.executed is False
    assert sleep.proposed is True
    assert sleep.proposed_prunable_params == 1
    assert evolution.deaths == 0
    assert evolution.quarantines == 0
    assert evolution.expansions == 0
    assert evolution.pruning_decisions == 3
    assert evolution.details


def test_evolution_schema_accepts_legacy_and_rejects_malformed():
    model = _TinyModel()
    model.blocks[0]["moe"].neuroendocrine._decisions = {
        0: "prune",
        1: "quarantine",
        2: "expand",
    }
    legacy = execute_evolution_actions(
        model, UnifiedControlMesh(model).collect_signals()
    )
    assert legacy.pruning_decisions == 3

    model.blocks[0]["moe"].neuroendocrine._decisions = {"prune": "not-a-list"}
    with pytest.raises(ValueError, match="plasticity decision schema"):
        execute_evolution_actions(model, UnifiedControlMesh(model).collect_signals())


def test_lifecycle_records_mesh_failure_as_degraded_organ():
    organism = _DarwinLifecycleMixin.__new__(_DarwinLifecycleMixin)
    organism.model = SimpleNamespace(blocks=[SimpleNamespace(
        moe=SimpleNamespace(
            neuroendocrine=SimpleNamespace(
                state=lambda: (_ for _ in ()).throw(RuntimeError("sensor failed"))
            )
        )
    )])
    report = {}

    organism._evolution_death_loop(report)

    assert report["degraded_organs"] == ["unified_mesh"]
    assert report["organ_errors"]["unified_mesh"] == "sensor failed"


class _TrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32)

    def forward(self, input_ids, labels=None, domain=None):
        logits = self.head(self.embedding(input_ids))
        loss = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
        return SimpleNamespace(
            loss=loss,
            lm_loss=loss,
            mtp_loss=None,
            jepa_loss=None,
            aux_loss=torch.zeros((), device=loss.device),
            ghost_loss=None,
            heartbeat_stats={
                "dopamine": 0.4,
                "memory": {"avg_surprise": 0.2},
            },
            spider_confidence=torch.tensor([0.75]),
            decision_factors={"confidence": 0.8},
        )


def test_training_persists_last_output_and_metric_snapshot():
    organism = DarwinOrganism.__new__(DarwinOrganism)
    organism.cfg = SimpleNamespace(
        block_size=4,
        batch_size=1,
        seed=51,
        replay_warmup_examples=100,
        replay_ratio=0.0,
        replay_add_every=20,
        replay_sample_size=1,
        eval_every=100,
        grad_clip=1.0,
        accum_steps=1,
        holdout_tokens=0,
    )
    organism.cycle = 1
    organism.total_steps = 0
    organism.device = torch.device("cpu")
    organism.amp_dtype = None
    organism.token_ids = np.arange(64, dtype=np.int64) % 32
    organism.holdout_starts = ()
    organism.token_count = len(organism.token_ids)
    organism.replay = ReplayBuffer(capacity=4, seed=51)
    organism.model = _TrainingModel()
    organism.optimizer = torch.optim.SGD(organism.model.parameters(), lr=0.01)
    organism.soul = SimpleNamespace(heartbeat=lambda status: None)

    organism._train_cycle(1, {"forgetting": 0.0})

    assert organism._last_training_output.heartbeat_stats["dopamine"] == 0.4
    assert organism._last_training_output.loss.grad_fn is None
    assert organism._last_training_metrics["fresh"]["count"] == 1
    assert organism._last_training_metrics["replay"]["count"] == 0
    assert organism._last_training_metrics["heldout"] is None


class _CycleHarness(_DarwinLifecycleMixin):
    def __init__(self, *, mesh_enabled, sleep_enabled):
        self.cycle = 0
        self.total_steps = 0
        self.cfg = SimpleNamespace(max_steps_per_cycle=1)
        self.model_config = SimpleNamespace(
            unified_mesh_enabled=mesh_enabled,
            sleep_enabled=sleep_enabled,
        )
        self.model = SimpleNamespace(
            blocks=[],
            execute_structural_actions=self._execute_structural_actions,
            named_parameters=lambda: [],
        )
        self.structural_calls = 0
        self.sleep_calls = 0
        self.evolution_observer_calls = 0

    def _execute_structural_actions(self):
        self.structural_calls += 1
        return []

    def _data_lifecycle(self, report):
        return None

    def _train_cycle(self, steps, report):
        report["loss_end"] = 1.0

    def _evolution_cycle(self, report):
        return None

    def _save_cycle(self, report):
        return None

    def _sleep_cycle(self, report):
        self.sleep_calls += 1

    def _evolution_death_loop(self, report):
        self.evolution_observer_calls += 1


def test_cycle_toggles_mesh_and_sleep_and_has_one_structural_authority():
    disabled = _CycleHarness(mesh_enabled=False, sleep_enabled=False)
    disabled.run_cycle()
    assert disabled.structural_calls == 1
    assert disabled.sleep_calls == 0
    assert disabled.evolution_observer_calls == 0

    mesh_only = _CycleHarness(mesh_enabled=True, sleep_enabled=False)
    mesh_only.run_cycle()
    assert mesh_only.structural_calls == 1
    assert mesh_only.sleep_calls == 0
    assert mesh_only.evolution_observer_calls == 1

    fully_enabled = _CycleHarness(mesh_enabled=True, sleep_enabled=True)
    fully_enabled.run_cycle()
    assert fully_enabled.structural_calls == 1
    assert fully_enabled.sleep_calls == 1
    assert fully_enabled.evolution_observer_calls == 1
