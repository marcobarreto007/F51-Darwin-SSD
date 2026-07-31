from pathlib import Path

import pytest

from f51_darwin.data_factory import (
    DataFactory,
    DataFactoryPaths,
    approve_quarantine_item,
    assert_training_corpus_allowed,
    is_blocked_training_path,
)
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.dataset_states import DatasetStatus, SourceType


def test_blocked_training_paths(tmp_path: Path) -> None:
    root = tmp_path
    blocked = root / "data" / "generated" / "candidates"
    blocked.mkdir(parents=True)
    assert is_blocked_training_path(blocked, root)
    with pytest.raises(PermissionError):
        assert_training_corpus_allowed(blocked, root)


def test_synthetic_never_auto_training(tmp_path: Path) -> None:
    paths = DataFactoryPaths.from_root(tmp_path)
    factory = DataFactory(paths, firewall=DataFirewall(FirewallConfig(min_chars=10)))
    record = factory.register_synthetic(
        text="Synthetic candidate text long enough for quality scoring here.",
        generator_model="F51-Darwin-SSD-30M",
        generator_checkpoint="checkpoints/base/run_x/step_0000001.pt",
        prompt="generate doctrine text",
    )
    assert record.status == DatasetStatus.CANDIDATE
    assert (paths.candidates / f"{record.id}.json").exists()
    assert not (paths.corpus / f"{record.id}.json").exists()

    decision = factory.audit_candidate(record, "Synthetic candidate text long enough for quality scoring here.")
    assert decision.status in {DatasetStatus.QUARANTINE, DatasetStatus.REJECTED}
    assert decision.status != DatasetStatus.APPROVED


def test_duplicate_hash_rejected(tmp_path: Path) -> None:
    paths = DataFactoryPaths.from_root(tmp_path)
    factory = DataFactory(paths, firewall=DataFirewall(FirewallConfig(min_chars=8)))
    text = "Duplicate candidate sample for firewall testing."
    first = factory.register_candidate(
        text=text,
        source_type=SourceType.IMPORTED,
        source_path="import://one",
        generator_model="n/a",
        generator_checkpoint="n/a",
    )
    factory.audit_candidate(first, text)
    second = factory.register_candidate(
        text=text,
        source_type=SourceType.IMPORTED,
        source_path="import://two",
        generator_model="n/a",
        generator_checkpoint="n/a",
    )
    decision = factory.audit_candidate(second, text)
    assert decision.status == DatasetStatus.REJECTED


def test_promotion_only_via_promote_script(tmp_path: Path) -> None:
    paths = DataFactoryPaths.from_root(tmp_path)
    factory = DataFactory(paths, firewall=DataFirewall(FirewallConfig(min_chars=8)))
    text = "Approved after explicit review for clean corpus promotion."
    record = factory.register_candidate(
        text=text,
        source_type=SourceType.SYNTHETIC,
        source_path="synthetic://generated",
        generator_model="F51-Darwin-SSD-30M",
        generator_checkpoint="checkpoints/base/run_x/step_0000001.pt",
    )
    decision = factory.audit_candidate(record, text)
    assert decision.status == DatasetStatus.QUARANTINE
    assert not list(paths.corpus.glob("*.txt"))

    approved = approve_quarantine_item(factory, record.id, reason="manual test approval")
    assert approved.status == DatasetStatus.APPROVED
    exported = factory.promote_approved_to_corpus(build_corpus=True)
    assert exported
    assert list(paths.corpus.glob("*.txt"))


def test_real_source_also_requires_explicit_approval(tmp_path: Path) -> None:
    paths = DataFactoryPaths.from_root(tmp_path)
    factory = DataFactory(paths, firewall=DataFirewall(FirewallConfig(min_chars=8)))
    text = "Verified local real source with sufficient automated quality."
    record = factory.register_candidate(
        text=text,
        source_type=SourceType.REAL,
        source_path="file:///review/pending.txt",
    )
    decision = factory.audit_candidate(record, text)
    assert decision.status == DatasetStatus.QUARANTINE
    assert not list(paths.corpus.glob("*.txt"))
