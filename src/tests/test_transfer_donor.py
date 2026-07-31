from __future__ import annotations

import json
from pathlib import Path

import torch

from f51_darwin.transfer.donor import (
    DONOR_MODEL_ID,
    DonorManifest,
    freeze_module,
    sha256_file,
    write_donor_manifest,
)


def test_freeze_module_disables_gradients_and_training() -> None:
    module = torch.nn.Linear(4, 3)
    freeze_module(module)
    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_donor_manifest_round_trip_and_file_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"darwin-transfer")
    manifest = DonorManifest(
        schema_version=1,
        model_id=DONOR_MODEL_ID,
        resolved_path=str(tmp_path.resolve()),
        tokenizer_class="GPT2TokenizerFast",
        model_class="GPT2LMHeadModel",
        vocab_size=50_257,
        n_layer=6,
        n_embd=768,
        files={artifact.name: sha256_file(artifact)},
    )
    output = tmp_path / "donor_manifest.json"
    write_donor_manifest(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_id"] == DONOR_MODEL_ID
    assert payload["files"]["weights.bin"] == sha256_file(artifact)
