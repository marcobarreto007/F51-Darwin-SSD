from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import f51_darwin.circuits.package as circuit_package
from f51_darwin.circuits import CircuitManifest, TapContract, TensorIdentity
from f51_darwin.circuits.package import (
    CircuitPackageError,
    tensor_sha256,
    verify_circuit_package,
    write_circuit_package,
)


def _manifest(tensors: dict[str, torch.Tensor]) -> CircuitManifest:
    identities = tuple(
        TensorIdentity(
            key=key,
            role="parameter",
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            sha256=tensor_sha256(tensor),
        )
        for key, tensor in sorted(tensors.items())
    )
    return CircuitManifest(
        source_checkpoint_sha256="b" * 64,
        source_base_checkpoint_id="100m-cycle-004",
        source_config_identity="f51-config-v1:" + "c" * 64,
        source_topology_identity="f51-topology-v1:" + "d" * 64,
        circuit_kind="predictor",
        constructor_id="predictor-v1",
        tensors=identities,
        tap=TapContract(
            provider="module_path_v1",
            module_path="blocks.1",
            position="post",
            output_selector="tensor",
            rank=3,
            width=4,
            minimum_sequence_length=2,
        ),
        accepted_recipient_families=("darwin-x",),
    )


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "predictor.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "predictor.bias": torch.arange(4, dtype=torch.float32),
    }


def test_package_round_trip_and_runtime_state_tamper_rejected(tmp_path: Path) -> None:
    tensors = _tensors()
    manifest = _manifest(tensors)
    root = tmp_path / "candidate"

    written = write_circuit_package(root, manifest, tensors, {"mode": "candidate"})
    loaded = verify_circuit_package(root, written.sha256)

    assert loaded.manifest == manifest
    assert torch.equal(loaded.tensors["predictor.weight"], tensors["predictor.weight"])
    assert loaded.runtime_state == {"mode": "candidate"}
    assert loaded.identity == written

    runtime_state = root / "runtime_state.json"
    payload = bytearray(runtime_state.read_bytes())
    payload[-2] = ord("x")
    runtime_state.write_bytes(payload)

    with pytest.raises(CircuitPackageError, match="checksum mismatch"):
        verify_circuit_package(root, written.sha256)


def test_writer_rejects_undeclared_extra_tensor(tmp_path: Path) -> None:
    tensors = _tensors()
    manifest = _manifest({"predictor.weight": tensors["predictor.weight"]})

    with pytest.raises(CircuitPackageError, match="tensor keys"):
        write_circuit_package(tmp_path / "candidate", manifest, tensors, {})


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (torch.zeros((2, 8), dtype=torch.float32), "shape mismatch"),
        (torch.zeros((4, 4), dtype=torch.float64), "dtype mismatch"),
    ],
)
def test_writer_rejects_tensor_identity_mismatch(
    tmp_path: Path, replacement: torch.Tensor, match: str
) -> None:
    tensors = _tensors()
    manifest = _manifest(tensors)
    tensors["predictor.weight"] = replacement

    with pytest.raises(CircuitPackageError, match=match):
        write_circuit_package(tmp_path / "candidate", manifest, tensors, {})


def test_writer_rejects_non_finite_tensor(tmp_path: Path) -> None:
    tensors = _tensors()
    tensors["predictor.weight"][0, 0] = float("nan")
    manifest = _manifest({**tensors, "predictor.weight": torch.zeros((4, 4))})

    with pytest.raises(CircuitPackageError, match="non-finite"):
        write_circuit_package(tmp_path / "candidate", manifest, tensors, {})


def test_writer_rejects_existing_non_empty_destination(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "old.txt").write_text("do not replace", encoding="utf-8")
    tensors = _tensors()

    with pytest.raises(CircuitPackageError, match="destination"):
        write_circuit_package(root, _manifest(tensors), tensors, {})


def test_writer_preserves_competing_destination_created_at_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors = _tensors()
    root = tmp_path / "candidate"
    staging = tmp_path / f".{root.name}.staging-{circuit_package.os.getpid()}"
    publish = circuit_package._publish_staging_no_replace

    def competing_publish(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "competitor.txt").write_text("preserve me", encoding="utf-8")
        publish(source, destination)

    monkeypatch.setattr(circuit_package, "_publish_staging_no_replace", competing_publish)

    with pytest.raises(CircuitPackageError, match="destination"):
        write_circuit_package(root, _manifest(tensors), tensors, {})

    assert (root / "competitor.txt").read_text(encoding="utf-8") == "preserve me"
    assert not staging.exists()


def test_verifier_rejects_unexpected_file_and_malformed_expected_sha256(tmp_path: Path) -> None:
    tensors = _tensors()
    root = tmp_path / "candidate"
    written = write_circuit_package(root, _manifest(tensors), tensors, {})

    with pytest.raises(CircuitPackageError, match="expected_sha256"):
        verify_circuit_package(root, "NOT-A-HASH")

    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(CircuitPackageError, match="unexpected files"):
        verify_circuit_package(root, written.sha256)


def test_verifier_rejects_tampered_tensor_despite_recomputed_file_checksum(tmp_path: Path) -> None:
    tensors = _tensors()
    root = tmp_path / "candidate"
    written = write_circuit_package(root, _manifest(tensors), tensors, {})

    # This proves the manifest's content hashes, not merely checksums.json, bind tensors.
    from safetensors.torch import save_file

    altered = _tensors()
    altered["predictor.weight"][0, 0] = 999.0
    save_file(altered, str(root / "weights.safetensors"))
    checksums = {
        "manifest.json": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "weights.safetensors": hashlib.sha256((root / "weights.safetensors").read_bytes()).hexdigest(),
        "runtime_state.json": hashlib.sha256((root / "runtime_state.json").read_bytes()).hexdigest(),
    }
    from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256

    (root / "checksums.json").write_bytes(canonical_json_bytes(checksums))
    identity = canonical_sha256(checksums)
    (root / "package.sha256").write_text(identity, encoding="utf-8")

    with pytest.raises(CircuitPackageError, match="content hash mismatch"):
        verify_circuit_package(root, identity)
