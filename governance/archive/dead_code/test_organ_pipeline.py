from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from f51_darwin.evolution_gate import EvolutionGateMetrics
from f51_darwin.model import DarwinOutput
from f51_darwin.organ_pipeline import OrganPipeline, OrganPipelineConfig


class _TinyOrganModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=8)
        self.token_embedding = nn.Embedding(32, 8)

    def forward(self, input_ids: torch.Tensor) -> DarwinOutput:
        hidden = self.token_embedding(input_ids)
        return DarwinOutput(logits=torch.zeros(*input_ids.shape, 32), hidden_states=hidden)


def _pipeline(tmp_path) -> OrganPipeline:  # noqa: ANN001
    return OrganPipeline(
        _TinyOrganModel(),
        OrganPipelineConfig(
            ghost_interval=1000,
            evolution_interval=1,
            legacy_update_interval=1000,
            legacy_save_path=tmp_path / "legacy",
        ),
    )


def test_organ_pipeline_waits_for_verified_evolution_metrics(tmp_path) -> None:  # noqa: ANN001
    pipeline = _pipeline(tmp_path)

    result = pipeline.cycle(torch.tensor([[4, 5, 6, 7]]), step=1, loss=1.0)

    assert result.evolution_decision == "waiting_for_metrics"
    assert "missing_evolution_metrics" in result.scars
    assert result.evolution_gates == {}
    assert not result.lessons


def test_organ_pipeline_uses_real_evolution_gate_for_promotion(tmp_path) -> None:  # noqa: ANN001
    pipeline = _pipeline(tmp_path)
    metrics = EvolutionGateMetrics(
        baseline_loss=2.0,
        candidate_loss=1.8,
        baseline_replay_loss=2.0,
        candidate_replay_loss=1.9,
        generated_total=2,
        verified_generated=2,
        train_steps=8,
        tokens_seen=64,
    )

    result = pipeline.cycle(
        torch.tensor([[4, 5, 6, 7]]),
        step=1,
        loss=1.0,
        evolution_metrics=metrics,
    )

    assert result.evolution_decision == "promote"
    assert result.evolution_gates["loss_improved"]
    assert result.evolution_gates["replay_retained"]
    assert result.evolution_gates["verified_generation"]
    assert result.evolution_gates["verification_rate"]
    assert result.lessons


def test_organ_pipeline_quarantines_loss_only_improvement(tmp_path) -> None:  # noqa: ANN001
    pipeline = _pipeline(tmp_path)
    metrics = EvolutionGateMetrics(
        baseline_loss=2.0,
        candidate_loss=1.8,
        baseline_replay_loss=2.0,
        candidate_replay_loss=1.9,
        generated_total=2,
        verified_generated=0,
        train_steps=8,
        tokens_seen=64,
    )

    result = pipeline.cycle(
        torch.tensor([[4, 5, 6, 7]]),
        step=1,
        loss=1.0,
        evolution_metrics=metrics,
    )

    assert result.evolution_decision == "quarantine"
    assert not result.evolution_gates["verified_generation"]
    assert result.scars
