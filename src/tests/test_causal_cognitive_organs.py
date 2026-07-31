from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.losses import (
    CAUSAL_LOSS_SEMANTICS_VERSION,
    LEGACY_LOSS_SEMANTICS_VERSION,
)
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.heartbeat import (
    SurpriseMemorySlot,
    TestTimeMemory as TTM,
    bound_memory_residual,
)
from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
from f51_darwin.organism.checkpoint import validate_v8_causal_contract
from f51_darwin.spider_sense import SpiderSense
from f51_darwin.state_identity import (
    backbone_identity,
    causal_cognitive_state_identity,
    training_contract_identity,
)


def _config(**overrides: object) -> DarwinXConfig:
    values = {
        "vocab_size": 32,
        "context_length": 8,
        "inference_context_length": 16,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "fine_experts": 2,
        "shared_experts": 1,
        "experts_per_token": 1,
        "fine_expert_hidden_dim": 8,
        "shared_expert_hidden_dim": 8,
        "mtp_depth": 1,
        "mtp_weight": 0.0,
        "jepa_weight": 0.0,
        "ghost_enabled": False,
        "aux_loss_adaptive": False,
        "spider_sense_enabled": True,
        "heartbeat_enabled": False,
    }
    values.update(overrides)
    return DarwinXConfig(**values)


def _slot(key: torch.Tensor, value: torch.Tensor) -> SurpriseMemorySlot:
    return SurpriseMemorySlot(
        key=key.detach().clone(),
        value=value.detach().clone(),
        timestamp="2026-07-18T00:00:00+00:00",
        domain="test",
        surprise_score=1.0,
    )


def test_ttm_retrieval_is_batch_safe_and_matches_individual_queries() -> None:
    memory = TTM(d_model=8, capacity=4)
    with torch.no_grad():
        memory.proj_key.weight.zero_()
        memory.proj_key.bias.zero_()
        memory.proj_key.weight[0, 0] = 1.0
        memory.proj_key.weight[1, 1] = 1.0
    memory.slots = [
        _slot(torch.tensor([1.0, 0.0]), torch.arange(8, dtype=torch.float32)),
        _slot(torch.tensor([0.0, 1.0]), -torch.arange(8, dtype=torch.float32)),
    ]
    query = torch.zeros(2, 3, 8)
    query[0, :, 0] = 1.0
    query[1, :, 1] = 1.0

    batched = memory.retrieve(query, top_k=1)
    separate = torch.cat(
        [memory.retrieve(query[index : index + 1], top_k=1) for index in range(2)],
        dim=0,
    )

    assert batched is not None
    assert batched.shape == (2, 1, 8)
    torch.testing.assert_close(batched, separate, rtol=0.0, atol=0.0)
    assert not torch.equal(batched[0], batched[1])


def test_ttm_write_association_uses_separate_key_and_value_sources() -> None:
    memory = TTM(d_model=8, capacity=4)
    with torch.no_grad():
        memory.proj_key.weight.zero_()
        memory.proj_key.bias.zero_()
        memory.proj_key.weight[0, 0] = 1.0
        memory.proj_key.weight[1, 1] = 1.0
    key_source = torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    value_source = torch.arange(8, dtype=torch.float32).unsqueeze(0)

    wrote = memory.write_association(
        key_source,
        value_source,
        jepa_error=1.0,
        domain="fact-a",
    )

    assert wrote is True
    assert len(memory.slots) == 1
    torch.testing.assert_close(
        memory.slots[0].key,
        memory.proj_key(key_source)[0],
    )
    torch.testing.assert_close(memory.slots[0].value, value_source[0])
    assert memory.slots[0].domain == "fact-a"


@pytest.mark.parametrize(
    ("key_source", "value_source", "match"),
    [
        (torch.zeros(8), torch.zeros(1, 8), r"shape \[batch, d_model\]"),
        (torch.zeros(2, 8), torch.zeros(2, 8), r"shape \[1, d_model\]"),
        (
            torch.full((1, 8), float("nan")),
            torch.zeros(1, 8),
            "finite",
        ),
    ],
)
def test_ttm_write_association_rejects_invalid_sources(
    key_source: torch.Tensor,
    value_source: torch.Tensor,
    match: str,
) -> None:
    memory = TTM(d_model=8, capacity=4)

    with pytest.raises(ValueError, match=match):
        memory.write_association(
            key_source,
            value_source,
            jepa_error=1.0,
        )


def test_bound_memory_residual_never_exceeds_reference_norm() -> None:
    value = torch.tensor([[30.0, 40.0, 0.0, 0.0]])
    reference = torch.tensor([[3.0, 4.0, 0.0, 0.0]])

    bounded = bound_memory_residual(value, reference)

    assert bounded.norm().item() == pytest.approx(reference.norm().item())
    torch.testing.assert_close(
        bound_memory_residual(reference * 0.5, reference),
        reference * 0.5,
    )
    with pytest.raises(ValueError, match="finite"):
        bound_memory_residual(
            torch.full_like(value, float("inf")),
            reference,
        )


def test_ttm_last_position_readout_queries_and_changes_only_t_minus_one() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
        ttm_residual_max_scale=0.2,
        ttm_entity_addressing=True,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config).eval()
    batch = torch.tensor([[4, 5, 6, 7]])
    recorded_queries: list[torch.Tensor] = []
    raw_hidden: list[torch.Tensor] = []
    value = torch.arange(config.d_model, dtype=torch.float32).unsqueeze(0)

    def capture_norm(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        raw_hidden.append(output.detach().clone())

    def retrieve(query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args
        assert kwargs == {"top_k": 1, "already_pooled": True}
        recorded_queries.append(query.detach().clone())
        return value.to(device=query.device, dtype=query.dtype)

    model.heartbeat.tt_memory.retrieve = retrieve
    handle = model.norm.register_forward_hook(capture_norm)
    try:
        with torch.inference_mode():
            model.ttm_residual_gate.zero_()
            control = model(batch, heartbeat=False)
            model.ttm_residual_gate.fill_(8.0)
            active = model(batch, heartbeat=False)
    finally:
        handle.remove()

    assert len(recorded_queries) == 2
    assert len(raw_hidden) == 2
    torch.testing.assert_close(
        recorded_queries[-1],
        raw_hidden[-1][:, -1, :],
        rtol=0.0,
        atol=0.0,
    )
    assert active.ttm_memory_retrieved is True
    torch.testing.assert_close(
        active.logits[:, :-1],
        control.logits[:, :-1],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(
        active.logits[:, -1] - control.logits[:, -1]
    ).item() > 0


def test_legacy_v1_adds_no_new_cognitive_state_or_output() -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_config(loss_semantics_version=LEGACY_LOSS_SEMANTICS_VERSION))
    state_keys = set(model.state_dict())
    batch = torch.tensor([[4, 5, 6, 7]])

    output = model(batch, labels=batch)

    assert model.ttm_residual_gate is None
    assert not any("ttm_residual_gate" in key for key in state_keys)
    assert output.spider_confidence is None
    assert output.spider_loss is None
    assert output.effective_spider_loss is None
    assert model.topology_manifest()["version"] == 7


@pytest.mark.parametrize(
    "toggle",
    ["ttm_residual_enabled", "spider_calibration_enabled"],
)
def test_cognitive_toggles_require_loss_v2(toggle: str) -> None:
    with pytest.raises(ValueError, match="loss_semantics_version=2"):
        _config(
            loss_semantics_version=LEGACY_LOSS_SEMANTICS_VERSION,
            **{toggle: True},
        )


def test_ttm_zero_gate_is_exact_identity_and_new_state_is_v8() -> None:
    base_config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
    )
    causal_config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
    )
    torch.manual_seed(51)
    base = DarwinXModel(base_config).eval()
    torch.manual_seed(51)
    causal = DarwinXModel(causal_config).eval()
    assert causal.ttm_residual_gate is not None
    assert causal.ttm_residual_gate.item() == 0.0
    causal.heartbeat.tt_memory.slots.append(
        _slot(
            torch.ones(causal_config.d_model // 4),
            torch.ones(causal_config.d_model),
        )
    )
    batch = torch.tensor([[4, 5, 6, 7]])

    with torch.inference_mode():
        base_logits = base(batch, heartbeat=False).logits
        causal_logits = causal(batch, heartbeat=False).logits

    torch.testing.assert_close(causal_logits, base_logits, rtol=0.0, atol=0.0)
    assert "ttm_residual_gate" in causal.state_dict()
    assert causal.topology_manifest()["version"] == 8


def test_ttm_nonzero_gate_changes_logits_and_is_bounded() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
        ttm_residual_max_scale=0.2,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config).eval()
    batch = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        baseline = model(batch, heartbeat=False)
        key = model.heartbeat.tt_memory.proj_key(
            baseline.hidden_states.mean(dim=1)
        ).squeeze(0)
        model.heartbeat.tt_memory.slots.append(
            _slot(key, torch.ones(config.d_model))
        )
    model.train()
    training_output = model(batch, labels=batch, heartbeat=False)
    training_output.loss.backward()
    assert model.ttm_residual_gate.grad is not None
    assert torch.count_nonzero(model.ttm_residual_gate.grad).item() == 1
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        model.ttm_residual_gate.fill_(100.0)
        changed = model(batch, heartbeat=False)

    assert not torch.equal(changed.logits, baseline.logits)
    assert model.ttm_residual_scale().item() == pytest.approx(
        config.ttm_residual_max_scale
    )
    assert model.ttm_residual_scale().item() > 0.0


def test_legacy_identity_is_stable_but_cognitive_v8_has_new_identity() -> None:
    legacy_config = _config(
        loss_semantics_version=LEGACY_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
    )
    torch.manual_seed(51)
    legacy = DarwinXModel(legacy_config)
    full_config = dataclasses.asdict(legacy_config)
    historical_config = dict(full_config)
    for field in (
        "ttm_residual_enabled",
        "ttm_residual_max_scale",
        "spider_calibration_enabled",
        "spider_calibration_weight",
    ):
        historical_config.pop(field)

    legacy_id = backbone_identity(legacy.state_dict(), full_config)
    historical_id = backbone_identity(legacy.state_dict(), historical_config)
    assert legacy_id == historical_id

    causal_config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
    )
    torch.manual_seed(51)
    causal = DarwinXModel(causal_config)
    causal_id = backbone_identity(
        causal.state_dict(),
        dataclasses.asdict(causal_config),
    )
    assert causal_id == legacy_id

    heartbeat_state = causal.heartbeat_state_dict()
    state_id = causal_cognitive_state_identity(
        causal.state_dict(),
        heartbeat_state,
        dataclasses.asdict(causal_config),
    )
    with torch.no_grad():
        causal.ttm_residual_gate.add_(0.25)
    changed_gate_id = causal_cognitive_state_identity(
        causal.state_dict(),
        heartbeat_state,
        dataclasses.asdict(causal_config),
    )
    assert changed_gate_id != state_id

    changed_heartbeat = copy.deepcopy(heartbeat_state)
    changed_heartbeat["tt_memory_state_dict"]["proj_key.weight"].add_(1.0)
    assert causal_cognitive_state_identity(
        causal.state_dict(),
        changed_heartbeat,
        dataclasses.asdict(causal_config),
    ) != changed_gate_id


def test_v8_cognitive_contract_rejects_tampered_gate_or_heartbeat_memory() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
    )
    model = DarwinXModel(config)
    heartbeat_state = model.heartbeat_state_dict()
    payload = {
        "version": 8,
        "training_contract_id": training_contract_identity(),
        "loss_semantics_version": 2,
        "config": dataclasses.asdict(config),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "heartbeat_state": copy.deepcopy(heartbeat_state),
        "causal_contract": {
            "schema": "darwin-causal-checkpoint-v1",
            "mode": "disabled",
            "state": {"bus_enabled": False, "ablation_arm": None},
            "ledger_head": "0" * 64,
        },
    }
    payload["causal_cognitive_state_id"] = causal_cognitive_state_identity(
        payload["model_state_dict"],
        payload["heartbeat_state"],
        payload["config"],
    )

    validate_v8_causal_contract(
        payload,
        loss_semantics_version=2,
        causal_mode="disabled",
        causal_v8_migration=True,
        ledger_head="0" * 64,
    )

    tampered_gate = copy.deepcopy(payload)
    tampered_gate["model_state_dict"]["ttm_residual_gate"].add_(1.0)
    with pytest.raises(ValueError, match="causal_cognitive_state_id"):
        validate_v8_causal_contract(
            tampered_gate,
            loss_semantics_version=2,
            causal_mode="disabled",
            causal_v8_migration=True,
            ledger_head="0" * 64,
        )

    tampered_memory = copy.deepcopy(payload)
    tampered_memory["heartbeat_state"]["tt_memory_state_dict"][
        "proj_value.weight"
    ].mul_(0.0)
    with pytest.raises(ValueError, match="causal_cognitive_state_id"):
        validate_v8_causal_contract(
            tampered_memory,
            loss_semantics_version=2,
            causal_mode="disabled",
            causal_v8_migration=True,
            ledger_head="0" * 64,
        )


def test_v8_topology_roundtrip_is_strict_and_rejects_legacy_manifest() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config)
    manifest = copy.deepcopy(model.topology_manifest())

    assert DarwinXModel.restore_topology(manifest, model) == 0
    legacy_claim = copy.deepcopy(manifest)
    legacy_claim["version"] = 7
    legacy_claim.pop("causal_cognitive_contract")
    with pytest.raises(ValueError, match="explicit v8"):
        DarwinXModel.restore_topology(legacy_claim, model)


def test_spider_brier_uses_detached_token_correctness_and_backpropagates() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_calibration_enabled=True,
        spider_calibration_weight=0.5,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config)
    batch = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])

    output = model(batch, labels=batch)

    assert output.spider_confidence is not None
    assert output.spider_confidence.shape == batch.shape
    assert output.spider_loss is not None
    assert output.effective_spider_loss is not None
    torch.testing.assert_close(
        output.effective_spider_loss,
        output.spider_loss * config.spider_calibration_weight,
    )
    correctness = SpiderSense.token_correctness(output.logits, batch)
    assert correctness.shape == (batch.shape[0], batch.shape[1] - 1)
    assert correctness.requires_grad is False

    output.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model._spider_sense_module.parameters()
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_spider_brier_masks_ignore_index_exactly_like_lm_loss() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_calibration_enabled=True,
        spider_calibration_weight=0.5,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config).eval()
    tokens = torch.tensor([[4, 5, 6, 7], [7, 6, 5, 4]])
    labels = tokens.clone()
    labels[0, 2] = -100
    labels[1, 1] = -100

    output = model(tokens, labels=labels, heartbeat=False)

    correctness = SpiderSense.token_correctness(output.logits, labels)
    valid = labels[:, 1:].ne(-100)
    expected = (
        output.spider_confidence[:, :-1][valid]
        .sub(correctness[valid].to(output.spider_confidence.dtype))
        .square()
        .mean()
    )
    torch.testing.assert_close(output.spider_loss, expected)


def test_spider_hidden_output_is_batch_independent_per_sample() -> None:
    class _SampleLocalConfidence(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            score = hidden[:, :, 0].mean(dim=1, keepdim=True)
            return torch.sigmoid(score.mul(20.0)).expand(-1, hidden.size(1))

    config = _config(
        loss_semantics_version=LEGACY_LOSS_SEMANTICS_VERSION,
        heartbeat_enabled=False,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config).eval()
    model._spider_sense_module = _SampleLocalConfidence()
    batch = torch.tensor([[4, 5, 6, 7], [17, 18, 19, 20]])

    with torch.inference_mode():
        batched = model(batch, heartbeat=False).hidden_states
        separate = torch.cat(
            [
                model(batch[index : index + 1], heartbeat=False).hidden_states
                for index in range(batch.size(0))
            ],
            dim=0,
        )

    torch.testing.assert_close(batched, separate, rtol=0.0, atol=1e-6)


def test_zero_spider_scale_is_negative_control_without_double_counting() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_calibration_enabled=True,
        spider_calibration_weight=0.0,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config)
    batch = torch.tensor([[4, 5, 6, 7]])

    output = model(batch, labels=batch)

    assert output.spider_loss is not None
    assert output.spider_loss.item() >= 0.0
    assert output.effective_spider_loss is not None
    assert output.effective_spider_loss.item() == 0.0
    expected = (
        output.lm_loss
        + config.mtp_weight * output.mtp_loss
        + config.jepa_weight * output.jepa_loss
        + output.effective_aux_loss
        + output.ghost_loss
        + output.effective_spider_loss
    )
    torch.testing.assert_close(output.loss, expected, rtol=0.0, atol=0.0)
    output.loss.backward()
    assert all(
        parameter.grad is None
        for parameter in model._spider_sense_module.parameters()
    )


def test_cognitive_state_roundtrip_preserves_gate_memory_and_output() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        ttm_residual_enabled=True,
    )
    torch.manual_seed(51)
    source = DarwinXModel(config).eval()
    batch = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        probe = source(batch, heartbeat=False)
        key = source.heartbeat.tt_memory.proj_key(
            probe.hidden_states.mean(dim=1)
        ).squeeze(0)
        source.heartbeat.tt_memory.slots.append(
            _slot(key, torch.arange(config.d_model, dtype=torch.float32))
        )
        source.ttm_residual_gate.fill_(0.4)
    heartbeat_state = copy.deepcopy(source.heartbeat_state_dict())

    torch.manual_seed(99)
    restored = DarwinXModel(config).eval()
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    restored.load_heartbeat_state_dict(heartbeat_state)

    with torch.inference_mode():
        expected = source(batch, heartbeat=False).logits
        actual = restored(batch, heartbeat=False).logits

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert restored.heartbeat.tt_memory.slots
    assert restored.ttm_residual_gate.item() == pytest.approx(0.4)


def test_jepa_surprise_cannot_feed_back_into_same_step_logits() -> None:
    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        heartbeat_surprise_threshold=0.0,
        ttm_residual_enabled=True,
    )
    torch.manual_seed(51)
    active = DarwinXModel(config).eval()
    torch.manual_seed(51)
    negative_control = DarwinXModel(config).eval()
    with torch.no_grad():
        active.ttm_residual_gate.fill_(1.0)
        negative_control.ttm_residual_gate.fill_(1.0)
    batch = torch.tensor([[4, 5, 6, 7]])

    with torch.inference_mode():
        first = active(batch, labels=batch, heartbeat=True)
        control = negative_control(batch, labels=batch, heartbeat=False)

    torch.testing.assert_close(first.logits, control.logits, rtol=0.0, atol=0.0)
    assert first.heartbeat_stats["jepa_surprise"] == pytest.approx(
        first.jepa_loss.item()
    )
    assert len(active.heartbeat.tt_memory.slots) == 1


def test_heartbeat_does_not_retrieve_or_count_slot_written_same_beat() -> None:
    heartbeat = Heartbeat(
        8,
        HeartbeatConfig(
            memory_capacity=4,
            surprise_threshold=0.0,
            think_interval=100,
            explore_interval=100,
            self_reward_interval=100,
            ff_learn_interval=100,
        ),
    )
    hidden = torch.ones(1, 3, 8)

    first = heartbeat.beat(hidden, jepa_error=1.0)

    assert first["memory_used"] is False
    assert len(heartbeat.tt_memory.slots) == 1
    assert heartbeat.tt_memory.slots[0].access_count == 0

    second = heartbeat.beat(hidden, jepa_error=1.0)

    assert second["memory_used"] is True
    assert len(heartbeat.tt_memory.slots) == 2
    assert heartbeat.tt_memory.slots[0].access_count > 0
    assert heartbeat.tt_memory.slots[1].access_count == 0


def test_live_generate_causal_ttm_retrieves_once_without_legacy_bias() -> None:
    class _Tokenizer:
        eos_id = 31

        @staticmethod
        def encode(_prompt: str) -> list[int]:
            return [4, 5, 6]

        @staticmethod
        def decode(ids: list[int], skip_special: bool = True) -> str:
            del skip_special
            return " ".join(map(str, ids))

    config = _config(
        loss_semantics_version=CAUSAL_LOSS_SEMANTICS_VERSION,
        spider_sense_enabled=False,
        heartbeat_enabled=True,
        heartbeat_think_interval=100,
        heartbeat_surprise_threshold=2.0,
        ttm_residual_enabled=True,
    )
    torch.manual_seed(51)
    model = DarwinXModel(config).eval()
    with torch.no_grad():
        model.ttm_residual_gate.fill_(0.5)
    model.heartbeat.thinker.think = lambda _hidden: None
    calls = 0
    memory = torch.arange(config.d_model, dtype=torch.float32).view(1, 1, -1)

    def retrieve(_hidden: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal calls
        del args, kwargs
        calls += 1
        return memory

    model.heartbeat.tt_memory.retrieve = retrieve
    result = model.live_generate(
        "probe",
        _Tokenizer(),
        max_tokens=1,
        temperature=0.0,
        eos_id=999,
    )

    assert calls == 1
    assert result["memories_used"] == 1
