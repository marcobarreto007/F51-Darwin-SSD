from __future__ import annotations

from dataclasses import replace

import torch

from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.state_identity import backbone_identity


def _config(**changes: object) -> DarwinXConfig:
    base = DarwinXConfig(
        model_name="cognition-model-test",
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
        mtp_weight=0.0,
        jepa_weight=0.0,
        ghost_weight=0.0,
        ghost_enabled=False,
        spider_sense_enabled=False,
        heartbeat_enabled=False,
        nitro_enabled=False,
    )
    return replace(base, **changes)


def _metadata() -> CognitiveForwardMetadata:
    return CognitiveForwardMetadata(
        step_id=0,
        checkpoint_id="sha256:" + "a" * 64,
        context_digest="sha256:" + "b" * 64,
    )


def test_disabled_config_has_no_cognitive_state_or_events() -> None:
    model = DarwinXModel(_config()).eval()
    output = model(torch.tensor([[1, 2, 3]]), heartbeat=False)
    assert model.cognitive_runtime is None
    assert output.cognitive_pulse_events == ()
    assert not any(
        key.startswith("cognitive_runtime.") for key in model.state_dict()
    )


def test_shadow_config_loads_same_brain_and_preserves_logits() -> None:
    torch.manual_seed(51)
    base = DarwinXModel(_config()).eval()
    state = base.state_dict()
    batch = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = base(batch, heartbeat=False).logits
    base_identity = backbone_identity(base.state_dict(), base.config)

    cognition = DarwinXModel(
        _config(
            cognitive_architecture_version="three_organs_v1",
            cognitive_shadow_enabled=True,
            cognitive_pulse_enabled=True,
        )
    ).eval()
    missing, unexpected = cognition.load_state_dict(state, strict=False)
    assert unexpected == []
    assert missing
    assert all(key.startswith("cognitive_runtime.") for key in missing)
    with torch.inference_mode():
        output = cognition(
            batch,
            heartbeat=False,
            cognitive_metadata=_metadata(),
        )
    torch.testing.assert_close(output.logits, expected, rtol=0.0, atol=0.0)
    assert len(output.cognitive_pulse_events) == 1
    assert output.cognitive_pulse_events[0]["event"]["mode"] == "shadow"
    assert (
        backbone_identity(cognition.state_dict(), cognition.config)
        == base_identity
    )


def test_shadow_requires_explicit_metadata() -> None:
    model = DarwinXModel(
        _config(
            cognitive_architecture_version="three_organs_v1",
            cognitive_shadow_enabled=True,
            cognitive_pulse_enabled=True,
        )
    )
    try:
        model(torch.tensor([[1, 2, 3]]), heartbeat=False)
    except ValueError as error:
        assert "cognitive_metadata" in str(error)
    else:
        raise AssertionError("shadow forward accepted missing metadata")
