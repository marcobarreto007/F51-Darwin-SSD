from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .sources import sha256_file


TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
SMOL_SPECIAL_IDS = {
    "unk": 0,
    "bos": 1,
    "eos": 2,
    "pad": 2,
}


class SmolTokenizerAdapter:
    """Native Darwin tokenizer protocol backed by the exact Smol JSON."""

    def __init__(
        self,
        *,
        tokenizer,
        root: Path,
        hashes: Mapping[str, str],
    ) -> None:
        self._tokenizer = tokenizer
        self.root = root
        self.hashes = dict(hashes)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_hashes: Mapping[str, str] | None = None,
    ) -> "SmolTokenizerAdapter":
        root = Path(directory).resolve()
        missing = [
            filename
            for filename in TOKENIZER_FILENAMES
            if not (root / filename).is_file()
        ]
        if missing:
            raise ValueError(
                "Smol tokenizer directory is incomplete: "
                f"root={root}; missing={missing}"
            )

        hashes = {
            filename: sha256_file(root / filename)
            for filename in TOKENIZER_FILENAMES
        }
        for filename, expected in dict(expected_hashes or {}).items():
            actual = hashes.get(filename)
            if actual is None or actual != str(expected).lower():
                raise ValueError(
                    "Smol tokenizer hash mismatch: "
                    f"file={filename}; expected={expected}; actual={actual}"
                )

        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "tokenizers==0.22.2 is required for the Smol tokenizer"
            ) from exc

        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        config = json.loads(
            (root / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        special_map = json.loads(
            (root / "special_tokens_map.json").read_text(encoding="utf-8")
        )
        special_ids = {
            "unk": tokenizer.token_to_id(
                cls._special_content(special_map["unk_token"])
            ),
            "bos": tokenizer.token_to_id(
                cls._special_content(special_map["bos_token"])
            ),
            "eos": tokenizer.token_to_id(
                cls._special_content(special_map["eos_token"])
            ),
            "pad": tokenizer.token_to_id(
                cls._special_content(special_map["pad_token"])
            ),
        }
        if special_ids != SMOL_SPECIAL_IDS:
            raise ValueError(
                "Smol tokenizer special-token contract mismatch: "
                f"expected={SMOL_SPECIAL_IDS}; actual={special_ids}"
            )
        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if vocab_size != 49_152:
            raise ValueError(
                "Smol tokenizer vocabulary mismatch: "
                f"expected=49152; actual={vocab_size}"
            )
        configured_size = config.get("vocab_size")
        if configured_size is not None and int(configured_size) != vocab_size:
            raise ValueError(
                "Smol tokenizer config vocabulary differs from tokenizer.json"
            )
        return cls(tokenizer=tokenizer, root=root, hashes=hashes)

    @staticmethod
    def _special_content(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return str(value["content"])
        raise ValueError(f"invalid special-token declaration: {value!r}")

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def unk_id(self) -> int:
        return SMOL_SPECIAL_IDS["unk"]

    @property
    def bos_id(self) -> int:
        return SMOL_SPECIAL_IDS["bos"]

    @property
    def eos_id(self) -> int:
        return SMOL_SPECIAL_IDS["eos"]

    @property
    def pad_id(self) -> int:
        return SMOL_SPECIAL_IDS["pad"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = list(self._tokenizer.encode(text).ids)
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=skip_special,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": "smol-tokenizer-contract-v1",
            "tokenizer_json_sha256": self.hashes["tokenizer.json"],
            "tokenizer_config_sha256": self.hashes["tokenizer_config.json"],
            "special_tokens_sha256": self.hashes["special_tokens_map.json"],
            "vocab_size": self.vocab_size,
            "special_ids": dict(SMOL_SPECIAL_IDS),
        }
