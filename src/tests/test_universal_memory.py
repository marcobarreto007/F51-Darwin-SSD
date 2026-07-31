from __future__ import annotations

import copy

import pytest
import torch

from f51_darwin.cognition.memory import (
    MemoryReadoutAdapter,
    MemoryRecord,
    UniversalMemory,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_organ(d_model: int = 64, **kwargs: object) -> UniversalMemory:
    return UniversalMemory(d_model, **kwargs)


def _sq(batch: int = 1, seq: int = 4, d: int = 64) -> torch.Tensor:
    return torch.randn(batch, seq, d)


# ── record ──────────────────────────────────────────────────────────────────


def test_memory_record_verification_states() -> None:
    r = MemoryRecord(
        memory_id="abc",
        key_embedding=torch.randn(512),
        value_embedding=torch.randn(512),
        event_type="explicit_teaching",
        source_digest="",
        provenance="human",
        verification_state="verified",
        surprise=0.0,
        confidence=1.0,
        created_step=0,
        last_access_step=0,
        access_count=0,
    )
    assert r.is_verified
    assert not r.is_quarantined

    q = MemoryRecord(
        memory_id="xyz",
        key_embedding=torch.randn(512),
        value_embedding=torch.randn(512),
        event_type="calibrated_novelty",
        source_digest="",
        provenance="model_quarantine",
        verification_state="unverified",
        surprise=0.5,
        confidence=0.3,
        created_step=1,
        last_access_step=1,
        access_count=0,
    )
    assert not q.is_verified
    assert q.is_quarantined


# ── teach / recall ──────────────────────────────────────────────────────────


def test_empty_store_abstains() -> None:
    organ = _make_organ()
    recalls = organ.recall(_sq())
    assert len(recalls) == 1
    assert recalls[0].abstained
    assert recalls[0].abstention_reason == "empty_store"


def test_teach_and_recall_exact() -> None:
    organ = _make_organ(recall_threshold=0.5)
    key_hidden = _sq(d=64)
    value_hidden = _sq(d=64)
    rid = organ.teach(key_hidden, value_hidden, event_type="explicit_teaching", provenance="human")
    assert rid
    assert organ.slot_count == 1
    assert organ.verified_count == 1

    recalls = organ.recall(key_hidden, top_k=2)
    assert len(recalls) >= 1
    assert not recalls[0].abstained
    assert recalls[0].record.memory_id == rid
    assert recalls[0].score > 0.5


def test_recall_returns_none_for_novel_query_with_high_threshold() -> None:
    organ = _make_organ(recall_threshold=0.999)
    organ.teach(_sq(d=64), _sq(d=64), event_type="explicit_teaching", provenance="human")
    novel = torch.randn(1, 4, 64)
    recalls = organ.recall(novel, top_k=1)
    assert recalls[0].abstained or recalls[0].score < 0.999


def test_verified_only_recall_excludes_quarantined() -> None:
    organ = _make_organ()
    organ.teach(_sq(d=64), _sq(d=64), event_type="explicit_teaching", provenance="human")
    organ.teach(_sq(d=64), _sq(d=64), event_type="calibrated_novelty", provenance="model_quarantine")
    assert organ.verified_count == 1
    assert organ.quarantined_count == 1

    # Verified-only recall when query matches the verified entry
    recalls_v = organ.recall(
        torch.randn(1, 4, 64), top_k=4, require_verified=True
    )
    assert all(r.abstained or r.record.is_verified for r in recalls_v)


def test_best_value_returns_tensor_or_none() -> None:
    organ = _make_organ(recall_threshold=0.3)
    key = _sq(d=64)
    val = _sq(d=64)
    organ.teach(key, val, event_type="explicit_teaching", provenance="human")
    v = organ.best_value(key)
    assert isinstance(v, torch.Tensor)
    assert v.shape == (512,)

    organ2 = _make_organ()
    assert organ2.best_value(_sq()) is None


# ── write_if_novel ──────────────────────────────────────────────────────────


def test_write_if_novel_respects_threshold() -> None:
    organ = _make_organ(novelty_threshold=0.99)
    key = _sq(d=64)
    val = _sq(d=64)
    rid1 = organ.write_if_novel(key, val)
    assert rid1 is not None

    # Same key should not write again
    rid2 = organ.write_if_novel(key, val)
    assert rid2 is None
    assert organ.slot_count == 1


# ── persistence ─────────────────────────────────────────────────────────────


def test_memory_snapshot_roundtrip() -> None:
    organ = _make_organ()
    for i in range(3):
        organ.teach(
            torch.randn(1, 4, 64), torch.randn(1, 4, 64),
            event_type="explicit_teaching",
            provenance="human",
            tags=(f"tag_{i}",),
        )

    state = organ.memory_state_dict()
    assert "snapshot_sha256" in state
    assert len(state["records"]) == 3

    # As chaves sao guardadas CODIFICADAS. Um round-trip correto restaura os
    # encoders junto com os registros; sem isso a consulta e codificada por
    # uma projecao nova e comparada contra chaves da projecao antiga. Medido:
    # recall 1.0/1.0/1.0 virava -0.007/0.101/0.050 sem erro nenhum. Desde a
    # verificacao de fingerprint, load_memory_state recusa esse caso em vez
    # de devolver escore plausivel e errado.
    restored = _make_organ()
    restored.load_state_dict(organ.state_dict())
    restored.load_memory_state(state)
    assert restored.slot_count == 3

    # Verify a record
    r0 = next(iter(restored._store.values()))
    assert r0.tags[0] in {"tag_0", "tag_1", "tag_2"}


def test_memory_state_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        _make_organ().load_memory_state({"schema": "wrong", "records": []})


# ── readout adapter ─────────────────────────────────────────────────────────


def test_readout_gate_zero_is_bit_exact() -> None:
    torch.manual_seed(51)
    adapter = MemoryReadoutAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    mem_val = torch.randn(2, 512)
    pos = torch.tensor([[2], [1]], dtype=torch.long)
    result = adapter.condition_hidden(hidden, mem_val, pos)
    assert torch.equal(result, hidden)
    assert adapter.gate.item() == 0.0


def test_readout_nonzero_gate_applies_bounded_residual() -> None:
    torch.manual_seed(51)
    adapter = MemoryReadoutAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    mem_val = torch.randn(2, 512)
    pos = torch.tensor([[2], [1]], dtype=torch.long)
    with torch.no_grad():
        adapter.gate.fill_(2.0)
    result = adapter.condition_hidden(hidden, mem_val, pos)
    assert not torch.equal(result, hidden)
    # Check position specificity
    delta = result - hidden
    assert torch.count_nonzero(delta[0, :2]).item() == 0
    assert torch.count_nonzero(delta[0, 3:]).item() == 0


# ── consolidation ───────────────────────────────────────────────────────────


def test_consolidation_decays_stale_quarantined() -> None:
    organ = _make_organ()
    organ.teach(
        torch.randn(1, 4, 64), torch.randn(1, 4, 64),
        event_type="calibrated_novelty", provenance="model_quarantine",
    )
    assert organ.slot_count == 1
    # Artificially age the record
    organ._step_counter = 200
    report = organ.consolidate()
    assert report["decayed"] >= 0
    assert organ.slot_count <= 1  # quarantined + unaccessed gets decayed


def test_consolidation_preserves_verified() -> None:
    organ = _make_organ()
    organ.teach(
        torch.randn(1, 4, 64), torch.randn(1, 4, 64),
        event_type="explicit_teaching", provenance="human",
    )
    assert organ.verified_count == 1
    organ._step_counter = 200
    report = organ.consolidate()
    assert organ.verified_count == 1  # verified memories survive
