from __future__ import annotations

import json
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import f51_darwin.data as data_module
from f51_darwin.data_factory import (
    DataFactory,
    DataFactoryPaths,
    approve_quarantine_item,
)
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.dataset_layout import FEAST_TOKEN_RELATIVE, LIVE_TOKEN_RELATIVE
from f51_darwin.tokenizer import F51BPETokenizer
from research import ghost_feeder, ghost_stream
from scripts import ingest_pipeline


def _long_text(label: str = "verified") -> str:
    return ((label + " external evidence with readable scientific prose. ") * 12).strip()


def _patch_tokenization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data_module,
        "tokenize_documents",
        lambda documents, tokenizer: list(range(20)),
    )
    monkeypatch.setattr(
        F51BPETokenizer,
        "load",
        staticmethod(lambda path: object()),
    )


def _fake_organism(root, *, with_brainstem: bool = False):
    organism = SimpleNamespace(
        root=root,
        firewall=DataFirewall(
            FirewallConfig(min_chars=200, quality_pass_score=0.60)
        ),
        cfg=SimpleNamespace(
            corpus_dir="data/corpus",
            tokenizer_dir="tokenizer/f51_bpe_80k",
            max_synthetic_ratio=0.10,
        ),
        tokenizer=None,
        token_ids=None,
        token_path=None,
    )
    if with_brainstem:
        organism.brainstem = SimpleNamespace(
            check_synthetic_ratio=lambda new, total: SimpleNamespace(
                alive=True, metrics={"synthetic_ratio": new / total}
            )
        )
    return organism


def _external_layout(tmp_path):
    project = tmp_path / "F51-Darwin-SSD"
    project.mkdir()
    dataset = project / "workspace"
    workspace = dataset / "02_CORPUS"
    workspace.mkdir(parents=True)
    return project, workspace, dataset


def test_web_source_is_forced_to_imported_quarantine(tmp_path) -> None:
    project, workspace, dataset = _external_layout(tmp_path)
    pipeline = ingest_pipeline.IngestPipeline(project)

    accepted = pipeline.process_text(
        _long_text(),
        source_path="https://example.test/paper",
        source_type=SourceType.REAL,
    )

    assert accepted is False
    quarantine = list((workspace / "generated/quarantine").glob("*.json"))
    assert len(quarantine) == 1
    payload = json.loads(quarantine[0].read_text(encoding="utf-8"))
    assert payload["record"]["source_type"] == SourceType.IMPORTED.value
    assert payload["record"]["status"] == DatasetStatus.QUARANTINE.value
    assert not list((workspace / "approved").glob("*.txt"))
    assert not (dataset / LIVE_TOKEN_RELATIVE).exists()
    assert not (project / "data").exists()


def test_local_real_source_still_requires_explicit_approval(tmp_path) -> None:
    project, workspace, dataset = _external_layout(tmp_path)
    pipeline = ingest_pipeline.IngestPipeline(project)
    feast = np.arange(100, dtype=np.int32)
    feast_path = dataset / FEAST_TOKEN_RELATIVE
    feast_path.parent.mkdir(parents=True)
    feast.tofile(feast_path)
    text = _long_text("local")

    accepted = pipeline.process_text(
        text,
        source_path=str(tmp_path / "trusted_local_source.txt"),
        source_type=SourceType.REAL,
    )

    assert accepted is False
    quarantine = list((workspace / "generated/quarantine").glob("*.json"))
    assert len(quarantine) == 1
    payload = json.loads(quarantine[0].read_text(encoding="utf-8"))
    assert payload["record"]["status"] == DatasetStatus.QUARANTINE.value
    assert not list((workspace / "approved").glob("*.txt"))
    assert not (dataset / LIVE_TOKEN_RELATIVE).exists()
    assert not (project / "data").exists()


def test_atomic_live_write_preserves_previous_target_on_replace_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data/tokens_live.bin"
    target.parent.mkdir(parents=True)
    previous = np.array([7, 8, 9], dtype=np.int32)
    previous.tofile(target)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ingest_pipeline.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        ingest_pipeline.write_int32_atomic(
            target, np.array([1, 2, 3, 4], dtype=np.int32)
        )

    assert np.fromfile(target, dtype=np.int32).tolist() == previous.tolist()
    assert not list(target.parent.glob(".tokens_live.bin.*.tmp"))


def test_ghost_web_candidate_cannot_enter_corpus_without_approval(tmp_path) -> None:
    project, workspace, dataset = _external_layout(tmp_path)
    candidate_dir = workspace / "generated/candidates/ghost_stream"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "web.txt").write_text(_long_text(), encoding="utf-8")
    organism = _fake_organism(project)

    result = ghost_feeder.feed_organism(organism)

    assert result["candidates_found"] == 1
    assert result["approved"] == 0
    assert result["promoted"] == 0
    assert result["reloaded"] is False
    quarantine = list((workspace / "generated/quarantine").glob("*.json"))
    assert len(quarantine) == 1
    payload = json.loads(quarantine[0].read_text(encoding="utf-8"))
    assert payload["record"]["source_type"] == SourceType.IMPORTED.value
    assert not list((workspace / "approved").glob("*.txt"))
    assert not (dataset / LIVE_TOKEN_RELATIVE).exists()
    assert not (project / "data").exists()


def test_explicitly_approved_ghost_item_is_promoted_tokenized_and_reloaded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenization(monkeypatch)
    project, workspace, dataset = _external_layout(tmp_path)
    candidate_dir = workspace / "generated/candidates/ghost_stream"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "web.txt").write_text(_long_text(), encoding="utf-8")
    feast_path = dataset / FEAST_TOKEN_RELATIVE
    feast_path.parent.mkdir(parents=True)
    np.arange(100, dtype=np.int32).tofile(feast_path)
    organism = _fake_organism(project, with_brainstem=True)

    first = ghost_feeder.feed_organism(organism)
    assert first["reloaded"] is False
    quarantine_path = next(
        (workspace / "generated/quarantine").glob("*.json")
    )
    record_id = quarantine_path.stem
    factory = DataFactory(DataFactoryPaths.from_project(project))
    approve_quarantine_item(factory, record_id, reason="operator reviewed source")

    second = ghost_feeder.feed_organism(organism)

    assert second["promoted"] == 1
    assert second["reloaded"] is True
    assert (workspace / "approved" / f"{record_id}.txt").exists()
    assert np.fromfile(dataset / LIVE_TOKEN_RELATIVE, dtype=np.int32).tolist() == (
        list(range(100)) + list(range(10))
    )
    assert not (project / "data").exists()
    organism.token_ids._mmap.close()


def test_wikipedia_source_accepts_and_honors_total_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_stream(lang: str = "pt", limit: int = 300) -> list[dict]:
        calls.append((lang, limit))
        return [{"source": f"wikipedia_{lang}"} for _ in range(limit)]

    monkeypatch.setattr(ghost_stream, "stream_wikipedia", fake_stream)

    docs = ghost_stream.STREAM_SOURCES["wikipedia"](limit=5)

    assert calls == [("pt", 3), ("en", 2)]
    assert len(docs) == 5


def test_ghost_stream_import_does_not_resolve_physical_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import f51_darwin.dataset_layout as layout

    def fail_if_called(*args, **kwargs):
        raise AssertionError("workspace resolution happened during import")

    with monkeypatch.context() as scoped:
        scoped.setattr(layout, "resolve_dataset_root", fail_if_called)
        sys.modules.pop("research.ghost_stream", None)
        imported = importlib.import_module("research.ghost_stream")

    assert callable(imported.resolve_ghost_paths)
    sys.modules.pop("research.ghost_stream", None)
    globals()["ghost_stream"] = importlib.import_module("research.ghost_stream")
