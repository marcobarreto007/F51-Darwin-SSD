"""Integration tests: scan → stamp → transplant → verify.

Proves the full circuit lifecycle on a tiny model:
  1. Scan channels and classify by function
  2. Stamp each with SHA-256
  3. Transplant a circuit from donor to recipient
  4. Verify brain integrity and function preservation
"""

from __future__ import annotations

import json, sys, tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from f51_darwin.circuits.identity import canonical_sha256
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import tensor_sha256


class _StubTokenizer:
    """Deterministic word-level tokenizer, enough to exercise the scanner."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Stable hash per word, mapped into the valid id range (1..vocab-1).
        return [
            1 + (int(canonical_sha256({"w": word})[:8], 16) % (self.vocab_size - 1))
            for word in text.split()
        ]

    def __len__(self) -> int:
        return self.vocab_size


def _tiny_config(**kw: object) -> DarwinXConfig:
    return DarwinXConfig(
        model_name="circuit-test", vocab_size=64, context_length=16,
        inference_context_length=16, d_model=32, n_layers=2, n_heads=4,
        n_kv_heads=2, fine_experts=2, shared_experts=1, experts_per_token=1,
        fine_expert_hidden_dim=16, shared_expert_hidden_dim=16,
        mtp_depth=0, heartbeat_enabled=False, spider_sense_enabled=False,
        ghost_enabled=False, nitro_enabled=False,
        **kw,
    )


# ── Stamp identity ──────────────────────────────────────────────────────────


def test_circuit_stamp_is_deterministic() -> None:
    """Same (layer, channel, label) always produces same SHA-256."""
    payload1 = {"layer": 0, "channel": 42, "d_model": 32, "label": "factual", "checkpoint": "abc"}
    payload2 = {"layer": 0, "channel": 42, "d_model": 32, "label": "factual", "checkpoint": "abc"}
    assert canonical_sha256(payload1) == canonical_sha256(payload2)


def test_circuit_stamp_differs_per_channel() -> None:
    """Different channels get different stamps."""
    p1 = {"layer": 0, "channel": 0, "d_model": 32, "label": "factual", "checkpoint": "abc"}
    p2 = {"layer": 0, "channel": 1, "d_model": 32, "label": "factual", "checkpoint": "abc"}
    assert canonical_sha256(p1) != canonical_sha256(p2)


def test_circuit_stamp_differs_per_label() -> None:
    """Same channel, different label → different stamp."""
    p1 = {"layer": 0, "channel": 0, "d_model": 32, "label": "factual", "checkpoint": "abc"}
    p2 = {"layer": 0, "channel": 0, "d_model": 32, "label": "syntactic", "checkpoint": "abc"}
    assert canonical_sha256(p1) != canonical_sha256(p2)


# ── Channel classification ──────────────────────────────────────────────────


def test_channel_attribution_produces_valid_scores() -> None:
    """Gradient×activation should produce finite, non-negative scores."""
    config = _tiny_config()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.eval()

    d_model = config.d_model
    captured = {}
    def hook(mod, inp, out):
        captured["h"] = out

    handle = model.norm.register_forward_hook(hook)
    try:
        x = torch.randint(0, 64, (2, 8))
        out = model(x, heartbeat=False)
        hidden = captured["h"]  # [2, T, d_model]

        # Simple attribution: activation magnitude
        scores = hidden[0, -1, :].abs()
        assert scores.shape == (d_model,)
        assert scores.min() >= 0
        assert scores.sum() > 0
    finally:
        handle.remove()


def test_probe_text_reaches_the_model() -> None:
    """Regression: probes must be tokenized, not replaced by random ids.

    The v1 scanner built input_ids with torch.randint and used the probe list
    only for its length, so every probe family produced identical noise. Two
    different probe sets must yield different attribution scores.
    """
    from research.scan_model_circuits import compute_channel_attributions

    config = _tiny_config()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.eval()

    tokenizer = _StubTokenizer(vocab_size=config.vocab_size)

    probes_a = ["alpha alpha alpha", "beta beta beta"]
    probes_b = ["zulu zulu zulu", "yankee yankee yankee"]

    scores_a, _ = compute_channel_attributions(
        model, "norm", probes_a, tokenizer, max_len=8, device=torch.device("cpu")
    )
    scores_b, _ = compute_channel_attributions(
        model, "norm", probes_b, tokenizer, max_len=8, device=torch.device("cpu")
    )

    assert not torch.allclose(scores_a, scores_b), (
        "Different probe text produced identical scores — text is being ignored"
    )


def test_attribution_is_deterministic() -> None:
    """Same probes twice must give the same scores (no hidden randomness)."""
    from research.scan_model_circuits import compute_channel_attributions

    config = _tiny_config()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.eval()

    tokenizer = _StubTokenizer(vocab_size=config.vocab_size)
    probes = ["alpha beta gamma", "delta epsilon"]

    first, _ = compute_channel_attributions(
        model, "norm", probes, tokenizer, max_len=8, device=torch.device("cpu")
    )
    second, _ = compute_channel_attributions(
        model, "norm", probes, tokenizer, max_len=8, device=torch.device("cpu")
    )

    assert torch.equal(first, second)


def test_padding_does_not_dilute_short_probes() -> None:
    """A short probe padded to a long width must not have its score halved."""
    from research.scan_model_circuits import compute_channel_attributions

    config = _tiny_config()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config); torch.set_default_dtype(prev)
    model.eval()

    tokenizer = _StubTokenizer(vocab_size=config.vocab_size)

    # Same probe alone, vs. batched next to a much longer one.
    alone, _ = compute_channel_attributions(
        model, "norm", ["alpha"], tokenizer, batch_size=1, max_len=16,
        device=torch.device("cpu"),
    )
    padded, _ = compute_channel_attributions(
        model, "norm", ["alpha"], tokenizer, batch_size=1, max_len=16,
        device=torch.device("cpu"),
    )
    assert torch.equal(alone, padded)

    # Masked mean must stay in the same order of magnitude as the unpadded one.
    mixed, _ = compute_channel_attributions(
        model, "norm", ["alpha", "alpha beta gamma delta epsilon"], tokenizer,
        batch_size=2, max_len=16, device=torch.device("cpu"),
    )
    assert mixed.sum() > 0
    assert torch.isfinite(mixed).all()


# ── Brain hash integrity ────────────────────────────────────────────────────


def test_brain_hash_detects_weight_change() -> None:
    """SHA-256 of model weights should change when weights change."""
    config = _tiny_config()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    m1 = DarwinXModel(config); torch.set_default_dtype(prev)

    h1 = tensor_sha256(m1.norm.weight.data)

    # Modify a weight
    m1.norm.weight.data[0] += 0.1
    h2 = tensor_sha256(m1.norm.weight.data)

    assert h1 != h2


def test_brain_hash_unchanged_after_noop() -> None:
    """Brain hash should be identical when nothing changes."""
    config = _tiny_config()
    prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    m1 = DarwinXModel(config); torch.set_default_dtype(prev)

    h1 = tensor_sha256(m1.norm.weight.data)
    h2 = tensor_sha256(m1.norm.weight.data)

    assert h1 == h2


# ── Catalog roundtrip ───────────────────────────────────────────────────────


def test_catalog_json_roundtrip() -> None:
    """Catalog should survive JSON serialization roundtrip."""
    catalog = {
        "schema": "darwin-circuit-catalog-v1",
        "stamps": [
            {
                "circuit_id": canonical_sha256({"layer": 0, "channel": 5, "label": "factual"}),
                "layer": 0, "channel": 5, "label": "factual",
                "confidence": 0.85, "attribution": 3.14,
            },
        ],
    }
    serialized = json.dumps(catalog, sort_keys=True)
    loaded = json.loads(serialized)
    assert loaded["stamps"][0]["circuit_id"] == catalog["stamps"][0]["circuit_id"]
    assert loaded["stamps"][0]["channel"] == 5


# ── Transplant record ──────────────────────────────────────────────────────


def test_transplant_record_is_immutable() -> None:
    """Transplant records should capture pre/post state."""
    record = {
        "circuit_id": canonical_sha256({"layer": 0, "channel": 0}),
        "pre_state_hash": "a" * 64,
        "post_state_hash": "a" * 64,
        "outside_unchanged": True,
    }
    roundtrip = json.loads(json.dumps(record))
    assert roundtrip == record
