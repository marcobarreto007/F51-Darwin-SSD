from __future__ import annotations

from pathlib import Path

import pytest

from f51_darwin.state_identity import tokenizer_identity
from f51_darwin.transplant_16b.tokenizer import SmolTokenizerAdapter


ROOT = Path(__file__).resolve().parents[2]
SMOL_TOKENIZER = (
    ROOT
    / "workspace"
    / "00_DONORS"
    / "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
    / "snapshots"
    / "31b70e2e869a7173562077fd711b654946d38674"
)


def test_exact_smol_special_ids_and_round_trip() -> None:
    if not SMOL_TOKENIZER.is_dir():
        pytest.skip("local immutable Smol tokenizer specimen is unavailable")

    tokenizer = SmolTokenizerAdapter.load(SMOL_TOKENIZER)

    assert (
        tokenizer.vocab_size,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.unk_id,
    ) == (49_152, 1, 2, 2, 0)
    ids = tokenizer.encode("Darwin aprende.", add_bos=True, add_eos=True)
    assert ids[0] == 1
    assert ids[-1] == 2
    assert "Darwin" in tokenizer.decode(ids)
    assert tokenizer_identity(tokenizer).startswith(
        "smol-tokenizer-contract-v1:"
    )


def test_smol_hashes_are_bound_into_identity() -> None:
    if not SMOL_TOKENIZER.is_dir():
        pytest.skip("local immutable Smol tokenizer specimen is unavailable")

    tokenizer = SmolTokenizerAdapter.load(
        SMOL_TOKENIZER,
        expected_hashes={
            "tokenizer.json": (
                "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
            ),
            "tokenizer_config.json": (
                "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"
            ),
            "special_tokens_map.json": (
                "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52"
            ),
        },
    )

    assert tokenizer.identity_payload()["vocab_size"] == 49_152


def test_missing_smol_tokenizer_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Smol tokenizer"):
        SmolTokenizerAdapter.load(tmp_path)
