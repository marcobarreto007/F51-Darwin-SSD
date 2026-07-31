from pathlib import Path

import pytest
import torch

from f51_darwin.checkpointing import load_checkpoint, load_model_from_checkpoint, save_checkpoint
from f51_darwin.config import DarwinConfig
from f51_darwin.distillation import combine_teacher_logits, distillation_loss
from f51_darwin.expert_pool import ExpertModule, ExpertPool, ModuleState
from f51_darwin.model import F51DarwinModel
from f51_darwin.pruning import ablation_decision, mark_dead, quarantine_if_low_score
from f51_darwin.replay_buffer import ReplayBuffer, forgetting_proxy


def test_config_from_yaml() -> None:
    config = DarwinConfig(
        model_name="F51-Darwin-SSD-Legacy-Smoke",
        vocab_size=128,
        context_length=64,
        d_model=32,
        n_layers=8,
        n_heads=4,
    )
    assert config.model_name == "F51-Darwin-SSD-Legacy-Smoke"
    assert config.n_layers == 8
    assert config.attention_layer_indices == (3, 7)


def test_replay_buffer_mix() -> None:
    buffer = ReplayBuffer(capacity=8)
    buffer.add([1, 2, 3], label="a")
    buffer.add([4, 5, 6], label="b")
    mixed = buffer.mix_with_new([], replay_size=1)
    assert len(mixed) == 1
    assert forgetting_proxy(1.0, 1.2) == pytest.approx(0.2)


def test_expert_pool_lifecycle() -> None:
    pool = ExpertPool()
    pool.add_expert("expert_a", ExpertModule(16), created_at_cycle=1)
    pool.set_score("expert_a", 0.5)
    pool.set_state("expert_a", ModuleState.ACTIVE)
    x = torch.randn(1, 4, 16)
    routed = pool.route_active(x)
    assert routed.shape == x.shape
    assert pool.records["expert_a"].usage_count == 1


def test_pruning_requires_quarantine_before_dead() -> None:
    pool = ExpertPool()
    pool.add_expert("weak", ExpertModule(16), created_at_cycle=1, state=ModuleState.ACTIVE, score=-0.5)
    with pytest.raises(ValueError, match="quarantine"):
        mark_dead(pool, "weak")
    quarantine_if_low_score(pool, "weak", threshold=-0.1)
    mark_dead(pool, "weak")
    assert pool.records["weak"].state == ModuleState.DEAD


def test_ablation_decision() -> None:
    decision = ablation_decision(0.7, 0.695, threshold=0.01)
    assert decision.action == "quarantine"


def test_distillation_helpers() -> None:
    teacher_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    teacher_b = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    combined = combine_teacher_logits([teacher_a, teacher_b])
    assert combined.shape == teacher_a.shape
    student = torch.zeros_like(combined)
    loss = distillation_loss(student, combined, temperature=2.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = DarwinConfig(vocab_size=64, context_length=8, d_model=16, n_layers=4, n_heads=4)
    model = F51DarwinModel(config)
    path = tmp_path / "roundtrip.pt"
    save_checkpoint(path, model, config, metrics={"step": 1})
    loaded, loaded_config, metrics = load_model_from_checkpoint(path)
    assert loaded_config.vocab_size == config.vocab_size
    assert metrics["step"] == 1
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    out = loaded(input_ids)
    assert out.logits.shape == (1, 8, config.vocab_size)


def test_load_checkpoint_rejects_corrupt_zip_header(tmp_path: Path) -> None:
    path = tmp_path / "organism_cycle_001.pt"
    path.write_bytes(b"PK\x03\x04truncated")

    with pytest.raises(ValueError, match="status=corrupt_zip"):
        load_checkpoint(path)
