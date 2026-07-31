from __future__ import annotations

import torch

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig


class _TinyTokenizer:
    eos_id = 0

    def encode(self, _text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: list[int], *, skip_special: bool = True) -> str:
        del skip_special
        return " ".join(str(token_id) for token_id in token_ids)


def _tiny_model() -> DarwinXModel:
    config = DarwinXConfig(
        vocab_size=64,
        context_length=16,
        inference_context_length=32,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=4,
        shared_experts=1,
        experts_per_token=2,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=2,
        heartbeat_enabled=True,
        heartbeat_think_interval=1,
        heartbeat_memory_capacity=8,
    )
    return DarwinXModel(config).eval()


def test_quiet_star_preserves_bfloat16_dtype() -> None:
    heartbeat = Heartbeat(32)
    heartbeat.to(dtype=torch.bfloat16)
    context = torch.randn(1, 4, 32, dtype=torch.bfloat16)

    thought = heartbeat.thinker.think(context)

    assert thought.dtype == torch.bfloat16
    assert torch.isfinite(thought).all()


def test_live_generate_counts_exactly_one_heartbeat_per_generated_token() -> None:
    torch.manual_seed(51)
    model = _tiny_model().to(dtype=torch.bfloat16)
    assert model.heartbeat is not None
    beat_before = model.heartbeat.total_beats

    result = model.live_generate(
        "prompt",
        _TinyTokenizer(),
        max_tokens=2,
        temperature=0.0,
        eos_id=999,
    )

    assert len(result["token_ids"]) == 2
    assert result["heartbeat_beats"] == 2
    assert model.heartbeat.total_beats - beat_before == 2


def test_heartbeat_state_round_trip_restores_modules_and_memory_slots() -> None:
    torch.manual_seed(51)
    source = Heartbeat(
        16,
        HeartbeatConfig(memory_capacity=4, ff_layers=2, num_thoughts=2),
    )
    source.total_beats = 7
    source.dopamine = 0.75
    source.memory_writes = 1
    source.thoughts_generated = 3
    source.thinker.scores = [0.2, 0.8]
    source.explorer.register_domain("papers")

    with torch.no_grad():
        source.ff_stack.layers[0].weight.add_(0.25)
        source.tt_memory.proj_key.weight.mul_(0.5)
        source.thinker.thought_start.add_(0.125)

    memory_input = torch.randn(1, 3, 16)
    assert source.tt_memory.write_if_surprised(
        memory_input,
        jepa_error=0.9,
        domain="papers",
    )
    source.tt_memory.slots[0].access_count = 4
    state = source.state_dict()

    restored = Heartbeat(
        16,
        HeartbeatConfig(memory_capacity=4, ff_layers=2, num_thoughts=2),
    )
    restored.load_state_dict(state)

    assert restored.total_beats == 7
    assert restored.dopamine == 0.75
    assert restored.memory_writes == 1
    assert restored.thoughts_generated == 3
    assert restored.thinker.scores == [0.2, 0.8]
    assert "papers" in restored.explorer.explored

    for source_module, restored_module in (
        (source.ff_stack, restored.ff_stack),
        (source.tt_memory, restored.tt_memory),
        (source.thinker, restored.thinker),
    ):
        source_module_state = source_module.state_dict()
        restored_module_state = restored_module.state_dict()
        assert source_module_state.keys() == restored_module_state.keys()
        for key in source_module_state:
            assert torch.equal(source_module_state[key], restored_module_state[key])

    assert len(restored.tt_memory.slots) == 1
    source_slot = source.tt_memory.slots[0]
    restored_slot = restored.tt_memory.slots[0]
    assert torch.equal(source_slot.key, restored_slot.key)
    assert torch.equal(source_slot.value, restored_slot.value)
    assert restored_slot.timestamp == source_slot.timestamp
    assert restored_slot.domain == "papers"
    assert restored_slot.surprise_score == 0.9
    assert restored_slot.access_count == 4

    restored_slot.key.add_(1.0)
    assert not torch.equal(restored_slot.key, state["tt_memory_slots"][0]["key"])

    restored.to(dtype=torch.bfloat16)
    assert restored.tt_memory.slots[0].key.dtype == torch.bfloat16  # gitleaks:allow -- tensor metadata
    assert restored.tt_memory.slots[0].value.dtype == torch.bfloat16


def test_heartbeat_loads_legacy_counter_only_state() -> None:
    heartbeat = Heartbeat(16, HeartbeatConfig(memory_capacity=4))

    heartbeat.load_state_dict(
        {
            "beat": 5,
            "dopamine": 0.25,
            "memory_writes": 2,
            "explorations": 1,
            "thoughts_generated": 4,
            "domains": ["legacy"],
        }
    )

    assert heartbeat.total_beats == 5
    assert heartbeat.dopamine == 0.25
    assert heartbeat.memory_writes == 2
    assert heartbeat.explorations == 1
    assert heartbeat.thoughts_generated == 4
    assert "legacy" in heartbeat.explorer.explored
    assert heartbeat.tt_memory.slots == []
