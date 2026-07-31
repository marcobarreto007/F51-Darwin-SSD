from pathlib import Path

from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.provenance import ProvenanceLedger, content_hash


def test_ledger_append_and_hash_lookup(tmp_path: Path) -> None:
    ledger = ProvenanceLedger(tmp_path / "ledger")
    record = ProvenanceLedger.new_record(
        text="F51 Darwin SSD immune system test document.",
        source_type=SourceType.REAL,
        source_path=str(tmp_path / "sample.txt"),
    )
    ledger.append(record)
    found = ledger.find_by_hash(record.content_hash)
    assert found is not None
    assert found.id == record.id


def test_content_hash_stable() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abcd")
