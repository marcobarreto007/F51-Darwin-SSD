from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from f51_darwin.data import CausalLMDataLoader
from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.darwin_x_training import (
    ConcatenatedTokenBins,
    load_token_ids,
    train_step,
    validate_token_source,
)

ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = ROOT / "research" / "train_darwin_x.py"
_SPEC = importlib.util.spec_from_file_location("train_darwin_x", _TRAIN_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_TRAIN_DARWIN_X = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN_DARWIN_X)
save_darwin_x_checkpoint = _TRAIN_DARWIN_X.save_darwin_x_checkpoint


def _small_config(vocab_size: int = 128) -> DarwinXConfig:
    return DarwinXConfig(
        vocab_size=vocab_size,
        context_length=16,
        inference_context_length=64,
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
    )


def test_token_source_validation_accepts_current_int32_family(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    token_path = data / "tokens_chunk_aa"
    np.asarray([3, 4, 5, 127, 8, 9], dtype=np.int32).tofile(token_path)

    token_ids, selected = load_token_ids(tmp_path, token_bin=token_path)
    health = validate_token_source(
        token_ids,
        selected,
        _small_config(vocab_size=128),
        tokenizer_vocab_size=128,
    )

    assert selected == token_path
    assert health.in_vocab
    assert health.token_count == 6
    assert health.sample_max == 127


def test_token_source_validation_rejects_out_of_vocab_ids(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    token_path = data / "tokens_chunk_aa"
    np.asarray([1, 2, 999], dtype=np.int32).tofile(token_path)

    token_ids, selected = load_token_ids(tmp_path, token_bin=token_path)

    with pytest.raises(ValueError, match="outside model vocab"):
        validate_token_source(token_ids, selected, _small_config(vocab_size=128))


def test_load_token_ids_from_single_batch_index(tmp_path: Path) -> None:
    token_path = tmp_path / "batch_001.int32.bin"
    np.asarray([1, 2, 3, 4, 5], dtype=np.int32).tofile(token_path)
    index_path = tmp_path / "index.json"
    index_path.write_text(
        """
{
  "version": 1,
  "vocab_label": "80k",
  "batch_count": 1,
  "total_tokens": 5,
  "total_bytes": 20,
  "batches": [
    {
      "batch": "batch_001",
      "bin": "batch_001.int32.bin",
      "manifest": "batch_001.tokens.json",
      "tokens": 5,
      "bytes": 20,
      "sha256": "unused"
    }
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    token_ids, selected = load_token_ids(tmp_path, token_index=index_path)

    assert selected == index_path
    assert isinstance(token_ids, np.ndarray)
    assert token_ids.tolist() == [1, 2, 3, 4, 5]


def test_load_token_ids_from_multi_batch_index_virtual_sequence(tmp_path: Path) -> None:
    first = tmp_path / "batch_001.int32.bin"
    second = tmp_path / "batch_002.int32.bin"
    np.asarray([1, 2, 3], dtype=np.int32).tofile(first)
    np.asarray([4, 5, 6, 7], dtype=np.int32).tofile(second)
    index_path = tmp_path / "index.json"
    index_path.write_text(
        """
{
  "version": 1,
  "vocab_label": "80k",
  "batch_count": 2,
  "total_tokens": 7,
  "total_bytes": 28,
  "batches": [
    {"batch": "batch_001", "bin": "batch_001.int32.bin", "tokens": 3},
    {"batch": "batch_002", "bin": "batch_002.int32.bin", "tokens": 4}
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    token_ids, selected = load_token_ids(tmp_path, token_index=index_path)

    assert selected == index_path
    assert isinstance(token_ids, ConcatenatedTokenBins)
    assert len(token_ids) == 7
    assert token_ids[0] == 1
    assert token_ids[6] == 7
    assert token_ids[2:6].tolist() == [3, 4, 5, 6]

    loader = CausalLMDataLoader(token_ids, block_size=3, batch_size=2, seed=0)
    batch = loader.next_batch()
    assert tuple(batch.shape) == (2, 3)


def test_load_token_ids_rejects_index_count_mismatch(tmp_path: Path) -> None:
    token_path = tmp_path / "batch_001.int32.bin"
    np.asarray([1, 2, 3], dtype=np.int32).tofile(token_path)
    index_path = tmp_path / "index.json"
    index_path.write_text(
        """
{
  "version": 1,
  "vocab_label": "80k",
  "batch_count": 1,
  "total_tokens": 4,
  "batches": [
    {"batch": "batch_001", "bin": "batch_001.int32.bin", "tokens": 4}
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="token count mismatch"):
        load_token_ids(tmp_path, token_index=index_path)


def test_darwin_x_safe_train_step_is_finite() -> None:
    torch.manual_seed(51)
    model = DarwinXModel(_small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = torch.randint(4, 128, (2, 12))

    metrics = train_step(model, batch, optimizer, grad_clip=1.0)

    assert metrics["loss"] > 0
    assert metrics["grad_norm"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_darwin_x_safe_train_step_rejects_nan_before_optimizer_step() -> None:
    class BadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, batch: torch.Tensor, labels: torch.Tensor | None = None):
            return SimpleNamespace(loss=self.weight.sum() * torch.tensor(float("nan")))

    model = BadModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    before = model.weight.detach().clone()

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        train_step(model, torch.ones(1, 4, dtype=torch.long), optimizer, grad_clip=1.0)

    assert torch.equal(model.weight.detach(), before)


def test_darwin_x_checkpoint_round_trips_heartbeat_state(tmp_path: Path) -> None:
    config = _small_config()
    config = DarwinXConfig(
        **{
            **config.__dict__,
            "heartbeat_enabled": True,
            "heartbeat_memory_capacity": 8,
        }
    )
    model = DarwinXModel(config)
    batch = torch.randint(4, config.vocab_size, (1, 12))
    output = model(batch, labels=batch, domain="checkpoint")
    assert output.heartbeat_stats is not None

    checkpoint = save_darwin_x_checkpoint(
        tmp_path / "step_0000001.pt",
        model,
        config,
        step=1,
        metrics={"loss": float(output.loss.detach())},
        token_source=tmp_path / "tokens.bin",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    resumed = DarwinXModel(config)
    resumed.load_state_dict(payload["model_state_dict"])
    resumed.load_heartbeat_state_dict(payload.get("heartbeat_state"))
    state = resumed.heartbeat_state_dict()

    assert payload["heartbeat_state"]["beat"] == 1
    assert state is not None
    assert state["beat"] == 1
    assert "checkpoint" in state["domains"]
