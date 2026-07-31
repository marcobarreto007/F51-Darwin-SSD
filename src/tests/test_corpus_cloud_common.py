import pytest

from f51_darwin.corpus_cloud_common import safe_batch_name


def test_safe_batch_name_preserves_supported_characters() -> None:
    assert safe_batch_name("batch-01_alpha") == "batch-01_alpha"


def test_safe_batch_name_replaces_unsupported_characters() -> None:
    assert safe_batch_name("batch 01/alpha") == "batch_01_alpha"


def test_safe_batch_name_rejects_empty_names() -> None:
    with pytest.raises(ValueError, match="batch_name cannot be empty"):
        safe_batch_name("")
