from __future__ import annotations

from pathlib import Path

import pytest
import torch

from f51_darwin.darwin_x import DarwinXConfig
from f51_darwin.organism.checkpoint import _write_checkpoint_file
from f51_darwin.organism.checkpoint_root import (
    assert_new_checkpoint_target,
    build_lineage_root_identity,
    model_config_identity,
    preflight_checkpoint_root,
    write_lineage_root_identity,
)


def tiny_config() -> DarwinXConfig:
    return DarwinXConfig(
        model_name="tiny",
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


def test_fresh_start_rejects_non_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "organism_cycle_071.pt").write_bytes(b"gold")
    with pytest.raises(ValueError, match="fresh-start checkpoint root must be empty"):
        preflight_checkpoint_root(
            root=root,
            model_raw=tiny_config(),
            mode="fresh_start",
            resume=None,
        )


def test_fresh_start_accepts_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "isolated"
    report = preflight_checkpoint_root(
        root=root,
        model_raw=tiny_config(),
        mode="fresh_start",
        resume=None,
    )
    assert root.is_dir()
    assert report["config_identity"] == model_config_identity(tiny_config())


def test_existing_cycle_target_is_never_replaced(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    target = root / "organism_cycle_001.pt"
    target.write_bytes(b"scientific evidence")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assert_new_checkpoint_target(root, 1)
    assert target.read_bytes() == b"scientific evidence"


def test_existing_temporary_target_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    temporary = root / "organism_cycle_001.pt.tmp"
    temporary.write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="incomplete checkpoint"):
        assert_new_checkpoint_target(root, 1)


def test_checkpoint_writer_rejects_concurrent_lock(tmp_path: Path) -> None:
    target = tmp_path / "organism_cycle_001.pt"
    lock = target.with_suffix(".pt.lock")
    lock.write_text("pid=other\n", encoding="ascii")
    with pytest.raises(FileExistsError):
        _write_checkpoint_file({"version": 8}, target)
    assert lock.read_text(encoding="ascii") == "pid=other\n"
    assert not target.exists()


def test_resume_requires_matching_root_identity(tmp_path: Path) -> None:
    config = tiny_config()
    checkpoint = tmp_path / "isolated" / "organism_cycle_001.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "version": 7,
            "config": dict(config.__dict__),
            "tokenizer_id": "tok",
            "base_checkpoint_id": "base",
            "training_state": {"cycle": 1, "step": 10},
        },
        checkpoint,
    )
    identity = build_lineage_root_identity(
        model_config=config,
        tokenizer_id="tok",
        creation_mode="resume",
        base_checkpoint_id="base",
    )
    write_lineage_root_identity(checkpoint.parent, identity)
    report = preflight_checkpoint_root(
        root=checkpoint.parent,
        model_raw=config,
        mode="resume",
        resume=checkpoint,
    )
    assert report["model_name"] == "tiny"


def test_resume_rejects_foreign_root_identity(tmp_path: Path) -> None:
    config = tiny_config()
    checkpoint = tmp_path / "isolated" / "organism_cycle_001.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "version": 7,
            "config": dict(config.__dict__),
            "tokenizer_id": "tok",
            "base_checkpoint_id": "base",
        },
        checkpoint,
    )
    identity = build_lineage_root_identity(
        model_config=config,
        tokenizer_id="other",
        creation_mode="resume",
        base_checkpoint_id="base",
    )
    write_lineage_root_identity(checkpoint.parent, identity)
    with pytest.raises(ValueError, match="tokenizer_id"):
        preflight_checkpoint_root(
            root=checkpoint.parent,
            model_raw=config,
            mode="resume",
            resume=checkpoint,
        )
