from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from f51_darwin.darwin_x import DarwinXConfig
from tools.isolate_100m_checkpoint import isolate_checkpoint, sha256_file


def make_config(path: Path) -> DarwinXConfig:
    config = DarwinXConfig(
        model_name="F51-Darwin-X-100M",
        vocab_size=32,
        context_length=32,
        inference_context_length=32,
        d_model=16,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=16,
        shared_expert_hidden_dim=16,
        mtp_depth=1,
    )
    path.write_text(yaml.safe_dump(dict(config.__dict__)), encoding="utf-8")
    return config


def make_checkpoint(path: Path, config: DarwinXConfig) -> Path:
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "version": 8,
            "config": dict(config.__dict__),
            "tokenizer_id": "tok",
            "base_checkpoint_id": "base",
            "training_state": {"cycle": 1, "step": 500},
        },
        path,
    )
    return path


def test_isolation_copies_without_mutating_source(tmp_path: Path) -> None:
    config_path = tmp_path / "100m.yaml"
    config = make_config(config_path)
    source = make_checkpoint(tmp_path / "shared" / "organism_cycle_001.pt", config)
    source_hash = sha256_file(source)
    result = isolate_checkpoint(source, tmp_path / "isolated", config_path)
    assert source.exists()
    assert sha256_file(source) == source_hash
    assert sha256_file(result.checkpoint) == source_hash
    pointer = json.loads(result.pointer.read_text(encoding="utf-8"))
    assert pointer["path"] == "organism_cycle_001.pt"
    assert (result.pointer.parent / "lineage_root.json").exists()


def test_isolation_refuses_existing_destination(tmp_path: Path) -> None:
    config_path = tmp_path / "100m.yaml"
    config = make_config(config_path)
    source = make_checkpoint(tmp_path / "shared" / "organism_cycle_001.pt", config)
    destination = tmp_path / "isolated"
    destination.mkdir()
    existing = destination / "organism_cycle_001.pt"
    existing.write_bytes(b"existing")
    try:
        isolate_checkpoint(source, destination, config_path)
    except FileExistsError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("existing destination was not rejected")
    assert existing.read_bytes() == b"existing"
