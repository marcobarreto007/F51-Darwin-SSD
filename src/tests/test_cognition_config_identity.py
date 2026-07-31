from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.organism.checkpoint_root import (
    model_config_identity,
    structural_config_identity,
)
from f51_darwin.state_identity import backbone_identity


def _tiny() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="cognition-config-test",
        vocab_size=32,
        context_length=8,
        inference_context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        mtp_depth=0,
        heartbeat_enabled=False,
        spider_sense_enabled=False,
        ghost_enabled=False,
    )


def test_legacy_config_keeps_cognition_disabled() -> None:
    config = _tiny()
    assert config.cognitive_architecture_version == "disabled"
    assert config.cognitive_shadow_enabled is False
    assert config.cognitive_pulse_enabled is False


def test_disabled_defaults_preserve_pre_cognition_identities() -> None:
    config = _tiny()
    historical = dict(config.__dict__)
    for key in (
        "cognitive_architecture_version",
        "cognitive_shadow_enabled",
        "cognitive_pulse_enabled",
        "cognitive_organ_width",
        "cognitive_residual_max_scale",
    ):
        historical.pop(key)
    assert model_config_identity(config) == model_config_identity(historical)
    assert structural_config_identity(config) == structural_config_identity(
        historical
    )


def test_three_organ_v1_requires_width_512() -> None:
    with pytest.raises(ValueError, match="organ_width=512"):
        replace(
            _tiny(),
            cognitive_architecture_version="three_organs_v1",
            cognitive_organ_width=256,
        )


def test_shadow_toggle_is_operational_but_version_is_structural() -> None:
    base = _tiny()
    enabled = replace(
        base,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    same_structure = replace(enabled, cognitive_shadow_enabled=False)
    assert structural_config_identity(enabled) == structural_config_identity(
        same_structure
    )
    assert structural_config_identity(base) != structural_config_identity(enabled)


def test_brain_identity_ignores_organ_config_and_tensors() -> None:
    state = {
        "token_embedding.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "cognitive_runtime.memory_adapter.gate": torch.tensor(0.0),
    }
    base = _tiny()
    enabled = replace(
        base,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    assert backbone_identity(state, base) == backbone_identity(state, enabled)
