from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import nn


DONOR_MODEL_ID = "distilbert/distilgpt2"
DONOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DonorManifest:
    schema_version: int
    model_id: str
    resolved_path: str
    tokenizer_class: str
    model_class: str
    vocab_size: int
    n_layer: int
    n_embd: int
    files: dict[str, str]


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_donor_manifest(manifest: DonorManifest, path: Path) -> None:
    _atomic_json_write(asdict(manifest), path)


def build_donor_manifest(model: nn.Module, tokenizer: Any, donor_path: Path) -> DonorManifest:
    files = {
        str(path.relative_to(donor_path)).replace("\\", "/"): sha256_file(path)
        for path in sorted(donor_path.rglob("*"))
        if path.is_file() and path.name != "donor_manifest.json"
    }
    config = model.config
    return DonorManifest(
        schema_version=DONOR_SCHEMA_VERSION,
        model_id=DONOR_MODEL_ID,
        resolved_path=str(donor_path.resolve()),
        tokenizer_class=type(tokenizer).__name__,
        model_class=type(model).__name__,
        vocab_size=int(config.vocab_size),
        n_layer=int(config.n_layer),
        n_embd=int(config.n_embd),
        files=files,
    )


def verify_donor_manifest(manifest: DonorManifest, donor_path: Path) -> None:
    if manifest.schema_version != DONOR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported donor manifest schema: {manifest.schema_version}"
        )
    if manifest.model_id != DONOR_MODEL_ID:
        raise ValueError(
            f"donor identity mismatch: expected={DONOR_MODEL_ID} actual={manifest.model_id}"
        )
    for relative, expected in manifest.files.items():
        artifact = donor_path / relative
        if not artifact.is_file():
            raise FileNotFoundError(f"donor artifact missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise ValueError(
                f"donor artifact hash mismatch: path={artifact} expected={expected} actual={actual}"
            )


def load_frozen_teacher(
    donor_path: Path,
    *,
    allow_download: bool = False,
    device: str = "cpu",
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install the project transfer extra"
        ) from exc

    donor_path = donor_path.resolve()
    config_path = donor_path / "config.json"
    if not config_path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"donor is not local: {donor_path}; rerun once with allow_download=True"
            )
        donor_path.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(DONOR_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            DONOR_MODEL_ID,
            attn_implementation="eager",
        )
        tokenizer.save_pretrained(donor_path)
        model.save_pretrained(donor_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        donor_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        donor_path,
        local_files_only=True,
        attn_implementation="eager",
    )
    freeze_module(model)
    model.to(device)

    manifest_path = donor_path / "donor_manifest.json"
    if manifest_path.is_file():
        manifest = DonorManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
        verify_donor_manifest(manifest, donor_path)
    else:
        manifest = build_donor_manifest(model, tokenizer, donor_path)
        write_donor_manifest(manifest, manifest_path)
    return model, tokenizer, manifest
