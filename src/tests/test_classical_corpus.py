import json
from pathlib import Path

import pytest

from f51_darwin.dataset_layout import (
    CLASSICAL_TOKEN_MANIFEST_RELATIVE,
    CLASSICAL_TOKEN_RELATIVE,
    RAW_CLASSICAL_RELATIVE,
    resolve_dataset_path,
    resolve_dataset_root,
)
from research import build_classical_corpus as classical


def test_classical_catalog_has_unique_ids() -> None:
    catalog = classical._validated_catalog()
    book_ids = [book_id for _language, book_id in catalog]

    assert len(book_ids) == len(set(book_ids))
    # The catalog is intentionally append-only; uniqueness is the invariant.
    assert len(book_ids) >= 94
    assert ("en", 1400) in catalog
    assert ("fr", 1400) not in catalog


def test_manifest_is_derived_from_physical_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(classical, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(
        classical,
        "GUTENBERG_IDS_BY_LANGUAGE",
        {"pt": (101,), "en": (202,), "fr": (303,)},
    )
    (tmp_path / "gutenberg_101.txt").write_text("a" * 1200, encoding="utf-8")
    (tmp_path / "gutenberg_202.txt").write_text("b" * 1300, encoding="utf-8")
    # Missing FR file must not be counted merely because it is in the catalog.

    manifest = classical.write_manifest()

    assert manifest["total_books"] == 2
    assert manifest["total_chars"] == 2500
    assert manifest["gutenberg_pt"] == 1
    assert manifest["gutenberg_en"] == 1
    assert manifest["gutenberg_fr"] == 0
    assert manifest["files"] == ["gutenberg_101.txt", "gutenberg_202.txt"]
    assert len(manifest["records"]) == 2
    assert manifest["missing_catalog"] == [{"book_id": 303, "language": "fr"}]
    assert all(record["sha256"] for record in manifest["records"])


def test_live_classical_metadata_matches_physical_artifacts() -> None:
    root = Path(__file__).resolve().parents[2]
    try:
        dataset_root = resolve_dataset_root(root, require=True)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    corpus_dir = dataset_root / RAW_CLASSICAL_RELATIVE
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"classical corpus manifest not present: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_manifest = json.loads(
        (dataset_root / CLASSICAL_TOKEN_MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )
    token_bin = resolve_dataset_path(root, CLASSICAL_TOKEN_RELATIVE, require=True)
    physical_files = sorted(path.name for path in corpus_dir.glob("gutenberg_*.txt"))

    assert manifest["version"] == 2
    assert manifest["total_books"] == len(physical_files)
    assert len(manifest["files"]) == len(set(manifest["files"]))
    assert sorted(manifest["files"]) == physical_files
    assert manifest["total_chars"] == sum(
        (corpus_dir / name).stat().st_size for name in manifest["files"]
    )
    assert token_manifest["source_corpus_sha256"] == manifest["corpus_sha256"]
    assert token_manifest["tokens"] * 4 == token_manifest["size_bytes"]
    assert token_bin.stat().st_size == token_manifest["size_bytes"]
