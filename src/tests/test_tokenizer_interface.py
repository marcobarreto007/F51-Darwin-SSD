from pathlib import Path

import pytest

from f51_darwin.data import load_text_documents
from f51_darwin.tokenizer import F51BPETokenizer, MIN_BPE_VOCAB_SIZE


def test_train_encode_decode_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    documents = load_text_documents(root / "src" / "tests" / "fixtures" / "corpus")
    tokenizer = F51BPETokenizer.train(
        documents,
        vocab_size=MIN_BPE_VOCAB_SIZE,
        name="F51-Test-BPE",
    )
    text = documents[0]
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id
    decoded = tokenizer.decode(encoded)
    assert decoded
    saved = tokenizer.save(tmp_path / "tokenizer")
    loaded = F51BPETokenizer.load(saved)
    assert loaded.encode(text) == tokenizer.encode(text)


def test_tokenizer_metadata_is_f51_owned() -> None:
    root = Path(__file__).resolve().parents[2]
    documents = load_text_documents(root / "src" / "tests" / "fixtures" / "corpus")
    tokenizer = F51BPETokenizer.train(documents, vocab_size=MIN_BPE_VOCAB_SIZE)
    assert "F51" in tokenizer.metadata.name or "F51" in tokenizer.metadata.lineage


def test_tokenizer_rejects_vocab_smaller_than_byte_floor() -> None:
    with pytest.raises(ValueError, match="at least"):
        F51BPETokenizer.train(["local corpus"], vocab_size=MIN_BPE_VOCAB_SIZE - 1)
