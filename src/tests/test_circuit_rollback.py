from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import nn

import f51_darwin.circuits.rollback as rollback_module
from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256
from f51_darwin.circuits.ledger import CircuitLedger
from f51_darwin.circuits.manifest import CircuitManifest, TapContract, TensorIdentity
from f51_darwin.circuits.package import (
    tensor_sha256,
    write_circuit_package,
)
from f51_darwin.circuits.pointer import publish_active_pointer
from f51_darwin.circuits.rollback import (
    CircuitCheckpoint,
    CircuitRollbackError,
    CircuitSnapshot,
    create_rollback_child,
)


def _package_manifest(module_path: str) -> tuple[CircuitManifest, dict[str, torch.Tensor]]:
    tensors = {"candidate.scale": torch.ones(2, dtype=torch.float32)}
    manifest = CircuitManifest(
        source_checkpoint_sha256="1" * 64,
        source_base_checkpoint_id="toy-parent",
        source_config_identity="f51-config-v1:" + "2" * 64,
        source_topology_identity="f51-topology-v1:" + "3" * 64,
        circuit_kind="residual",
        constructor_id="residual-v1",
        tensors=(
            TensorIdentity(
                key="candidate.scale",
                role="parameter",
                shape=(2,),
                dtype="torch.float32",
                sha256=tensor_sha256(tensors["candidate.scale"]),
            ),
        ),
        tap=TapContract(
            provider="module_path_v1",
            module_path=module_path,
            position="post",
            output_selector="tensor",
            rank=2,
            width=2,
            minimum_sequence_length=1,
        ),
        accepted_recipient_families=("toy",),
    )
    return manifest, tensors


_CIRCUIT_A = _package_manifest("circuit_a")[0].identity
_CIRCUIT_B = _package_manifest("circuit_b")[0].identity


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.circuit_a = nn.Linear(2, 2, bias=True)
        self.circuit_b = nn.Linear(2, 2, bias=True)
        self.backbone.register_buffer("marker", torch.tensor([1.0]))
        self.circuit_a.register_buffer("marker", torch.tensor([2.0]))
        self.circuit_b.register_buffer("marker", torch.tensor([3.0]))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.backbone(values) + self.circuit_a(values) + self.circuit_b(values)


def _model() -> _ToyModel:
    torch.manual_seed(51)
    return _ToyModel().eval()


def _parameter_alias_model() -> _ToyModel:
    model = _model()
    model.circuit_a.weight = model.backbone.weight
    return model


def _buffer_alias_model() -> _ToyModel:
    model = _model()
    model.circuit_a.marker = model.backbone.marker
    return model


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_transaction(
    root: Path,
    transaction_id: str,
    circuit_id: str,
    package_sha256: str,
    tap_sha256: str,
    snapshot_sha256: str,
) -> tuple[str, int, str]:
    root.mkdir()
    (root / "events").mkdir()
    manifest = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "operation": "inject",
        "circuit_id": circuit_id,
        "package_sha256": package_sha256,
        "recipient_parent_sha256": "2" * 64,
        "tap_contract_sha256": tap_sha256,
        "preinstall_snapshot_sha256": snapshot_sha256,
        "lifecycle": "candidate",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "transaction.json").write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    previous = "0" * 64
    lifecycles = ("candidate", "shadow", "adapter_active")
    for sequence, lifecycle in enumerate(lifecycles):
        event: dict[str, object] = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "sequence": sequence,
            "lifecycle": lifecycle,
            "manifest_sha256": manifest_sha256,
            "previous_event_sha256": previous,
        }
        if sequence:
            event["from_lifecycle"] = lifecycles[sequence - 1]
        event["event_sha256"] = canonical_sha256(event)
        (root / "events" / f"{sequence:04d}-{lifecycle}.json").write_bytes(
            canonical_json_bytes(event)
        )
        previous = str(event["event_sha256"])
    return manifest_sha256, len(lifecycles), previous


def _artifact(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return _sha256_file(path)


def _package(root: Path, module_path: str) -> str:
    manifest, tensors = _package_manifest(module_path)
    return write_circuit_package(
        root,
        manifest,
        tensors,
        {"dependencies": ["activation"]},
    ).sha256


def _snapshot_payload(
    *,
    circuit_id: str,
    state_prefix: str,
    state: dict[str, torch.Tensor],
    package_path: Path,
    package_sha256: str,
    tap_path: Path,
    tap_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "circuit_id": circuit_id,
        "state_prefix": state_prefix,
        "circuit_state": state,
        "package_path": str(package_path.resolve()),
        "package_sha256": package_sha256,
        "tap_contract_path": str(tap_path.resolve()),
        "tap_contract_sha256": tap_sha256,
    }


def _registry_entry(
    snapshot_payload: dict[str, object],
    snapshot_sha256: str,
    transaction_root: Path,
    transaction_manifest_sha256: str,
    event_count: int,
    event_tail_sha256: str,
):
    return {
        key: value
        for key, value in snapshot_payload.items()
        if key not in {"schema_version", "circuit_id", "circuit_state"}
    } | {
        "preinstall_snapshot_sha256": snapshot_sha256,
        "transaction_root": str(transaction_root.resolve()),
        "transaction_manifest_sha256": transaction_manifest_sha256,
        "event_count": event_count,
        "event_tail_sha256": event_tail_sha256,
    }


def _build_artifacts(
    tmp_path: Path,
    *,
    probe: Callable[[nn.Module], object] | None = None,
    snapshot_extra_key: bool = False,
    model_factory: Callable[[], _ToyModel] = _model,
    regular_file_package: bool = False,
) -> tuple[
    CircuitCheckpoint,
    CircuitSnapshot,
    Path,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    initial = model_factory()
    preinstall_a = {
        f"circuit_a.{key}": tensor.detach().clone()
        for key, tensor in initial.circuit_a.state_dict().items()
    }
    preinstall_b = {
        f"circuit_b.{key}": tensor.detach().clone()
        for key, tensor in initial.circuit_b.state_dict().items()
    }
    if snapshot_extra_key:
        preinstall_a["circuit_b.weight"] = torch.zeros(2, 2)
    active = model_factory()
    with torch.no_grad():
        for parameter in active.circuit_a.parameters():
            parameter.add_(3.0)
        for parameter in active.circuit_b.parameters():
            parameter.sub_(2.0)
    active_b = {
        key: tensor.detach().clone()
        for key, tensor in active.circuit_b.state_dict().items()
    }

    if regular_file_package:
        package_a = tmp_path / "package-a.bin"
        package_b = tmp_path / "package-b.bin"
        package_a_sha = _artifact(package_a, b"package-a")
        package_b_sha = _artifact(package_b, b"package-b")
    else:
        package_a = tmp_path / "package-a"
        package_b = tmp_path / "package-b"
        package_a_sha = _package(package_a, "circuit_a")
        package_b_sha = _package(package_b, "circuit_b")
    tap_a = tmp_path / "tap-a.json"
    tap_b = tmp_path / "tap-b.json"
    tap_a_sha = _artifact(tap_a, canonical_json_bytes({"module_path": "circuit_a"}))
    tap_b_sha = _artifact(tap_b, canonical_json_bytes({"module_path": "circuit_b"}))
    snapshot_a_path = tmp_path / "snapshot-a.pt"
    snapshot_a_payload = _snapshot_payload(
        circuit_id=_CIRCUIT_A,
        state_prefix="circuit_a.",
        state=preinstall_a,
        package_path=package_a,
        package_sha256=package_a_sha,
        tap_path=tap_a,
        tap_sha256=tap_a_sha,
    )
    torch.save(snapshot_a_payload, snapshot_a_path)
    snapshot_a_sha = _sha256_file(snapshot_a_path)

    snapshot_b_path = tmp_path / "snapshot-b.pt"
    snapshot_b_payload = _snapshot_payload(
        circuit_id=_CIRCUIT_B,
        state_prefix="circuit_b.",
        state=preinstall_b,
        package_path=package_b,
        package_sha256=package_b_sha,
        tap_path=tap_b,
        tap_sha256=tap_b_sha,
    )
    torch.save(snapshot_b_payload, snapshot_b_path)
    snapshot_b_sha = _sha256_file(snapshot_b_path)
    tx_a = tmp_path / "tx-a"
    tx_b = tmp_path / "tx-b"
    tx_a_sha, tx_a_count, tx_a_tail = _write_transaction(
        tx_a,
        "tx-a",
        _CIRCUIT_A,
        package_a_sha,
        tap_a_sha,
        snapshot_a_sha,
    )
    tx_b_sha, tx_b_count, tx_b_tail = _write_transaction(
        tx_b,
        "tx-b",
        _CIRCUIT_B,
        package_b_sha,
        tap_b_sha,
        snapshot_b_sha,
    )
    registry_a = _registry_entry(
        snapshot_a_payload,
        snapshot_a_sha,
        tx_a,
        tx_a_sha,
        tx_a_count,
        tx_a_tail,
    )
    registry_b = _registry_entry(
        snapshot_b_payload,
        snapshot_b_sha,
        tx_b,
        tx_b_sha,
        tx_b_count,
        tx_b_tail,
    )

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CircuitLedger(ledger_path)
    for circuit_id, registry in (
        (_CIRCUIT_A, registry_a),
        (_CIRCUIT_B, registry_b),
    ):
        ledger.append(
            {
                "kind": "circuit_activated",
                "circuit_id": circuit_id,
                "package_sha256": registry["package_sha256"],
                "tap_contract_sha256": registry["tap_contract_sha256"],
                "preinstall_snapshot_sha256": registry[
                    "preinstall_snapshot_sha256"
                ],
                "event_count": registry["event_count"],
                "event_tail_sha256": registry["event_tail_sha256"],
                "transaction_manifest_sha256": registry[
                    "transaction_manifest_sha256"
                ],
            }
        )
    ledger_sha = ledger.verify()
    active_path = tmp_path / "active.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "schema_version": 1,
                "ledger_path": str(ledger_path.resolve()),
                "ledger_sha256": ledger_sha,
                "ledger_count": 2,
                "circuits": {
                    _CIRCUIT_A: registry_a,
                    _CIRCUIT_B: registry_b,
                },
            },
            "model_state": {
                key: tensor.detach().cpu() for key, tensor in active.state_dict().items()
            },
        },
        active_path,
    )
    active_checkpoint = CircuitCheckpoint(
        path=active_path,
        sha256=_sha256_file(active_path),
        ledger_path=ledger_path,
        ledger_sha256=ledger_sha,
        ledger_count=2,
        model_factory=model_factory,
        probe=probe or (lambda model: model(torch.ones(1, 2))),
    )
    return (
        active_checkpoint,
        CircuitSnapshot(snapshot_a_path, snapshot_a_sha),
        snapshot_b_path,
        preinstall_a,
        active_b,
    )


def test_rollback_creates_fresh_immutable_child_and_preserves_other_circuit(
    tmp_path: Path,
) -> None:
    active, snapshot, _, preinstall_a, active_b = _build_artifacts(tmp_path)
    output = tmp_path / "rollback-child.pt"

    child = create_rollback_child(active, snapshot, _CIRCUIT_A, output)

    payload = torch.load(child.path, map_location="cpu", weights_only=True)
    active_payload = torch.load(
        active.path,
        map_location="cpu",
        weights_only=True,
    )
    restored = active.model_factory()
    restored.load_state_dict(payload["model_state"], strict=True)
    for key, expected in preinstall_a.items():
        restored_key = key.removeprefix("circuit_a.")
        assert torch.equal(restored.circuit_a.state_dict()[restored_key], expected)
    for key, expected in active_b.items():
        assert torch.equal(restored.circuit_b.state_dict()[key], expected)
    for key, expected in active_payload["model_state"].items():
        if not key.startswith("circuit_a."):
            assert torch.equal(payload["model_state"][key], expected)
    rollback_metadata = payload["metadata"]
    assert rollback_metadata["parent_checkpoint_sha256"] == active.sha256
    assert rollback_metadata["rolled_back_circuit_id"] == _CIRCUIT_A
    assert rollback_metadata["rollback_snapshot_sha256"] == snapshot.sha256
    assert child.sha256 == _sha256_file(output)
    assert child.ledger_sha256 == active.ledger_sha256
    pointer_root = tmp_path / "pointers"
    pointer_root.mkdir()
    publish_active_pointer(pointer_root, "0" * 64, active)
    pointer = publish_active_pointer(pointer_root, active.sha256, child)
    assert pointer.checkpoint_sha256 == child.sha256
    assert pointer.parent_checkpoint_sha256 == active.sha256
    assert pointer.ledger_sha256 == child.ledger_sha256


def test_rollback_rejects_unknown_or_extra_snapshot_state_keys(tmp_path: Path) -> None:
    active, snapshot, _, _, _ = _build_artifacts(
        tmp_path, snapshot_extra_key=True
    )

    with pytest.raises(CircuitRollbackError, match="keys"):
        create_rollback_child(
            active, snapshot, _CIRCUIT_A, tmp_path / "rejected.pt"
        )


@pytest.mark.parametrize(
    "model_factory",
    [_parameter_alias_model, _buffer_alias_model],
    ids=["tied-parameter", "tied-buffer"],
)
def test_rollback_rejects_cross_prefix_parameter_or_buffer_alias(
    tmp_path: Path,
    model_factory: Callable[[], _ToyModel],
) -> None:
    active, snapshot, _, _, _ = _build_artifacts(
        tmp_path,
        model_factory=model_factory,
    )

    with pytest.raises(CircuitRollbackError, match="alias"):
        create_rollback_child(
            active,
            snapshot,
            _CIRCUIT_A,
            tmp_path / "alias-rejected.pt",
        )
    assert not (tmp_path / "alias-rejected.pt").exists()


def test_rollback_rejects_regular_file_instead_of_canonical_package(
    tmp_path: Path,
) -> None:
    active, snapshot, _, _, _ = _build_artifacts(
        tmp_path,
        regular_file_package=True,
    )

    with pytest.raises(CircuitRollbackError, match="package.*directory"):
        create_rollback_child(
            active,
            snapshot,
            _CIRCUIT_A,
            tmp_path / "file-package-rejected.pt",
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "checkpoint",
        "snapshot",
        "package",
        "tap",
        "ledger",
        "transaction_truncation",
        "transaction_alternate",
    ],
)
def test_rollback_fails_closed_on_identity_tampering(
    tmp_path: Path, tamper: str
) -> None:
    active, snapshot, _, _, _ = _build_artifacts(tmp_path)
    if tamper == "checkpoint":
        active.path.write_bytes(active.path.read_bytes() + b"x")
    elif tamper == "snapshot":
        snapshot.path.write_bytes(snapshot.path.read_bytes() + b"x")
    elif tamper in {"package", "tap", "ledger"}:
        payload = torch.load(snapshot.path, map_location="cpu", weights_only=True)
        path_key = {
            "package": "package_path",
            "tap": "tap_contract_path",
            "ledger": None,
        }[tamper]
        target = (
            Path(payload[path_key])
            if path_key is not None
            else active.ledger_path
        )
        if tamper == "package":
            runtime_state = target / "runtime_state.json"
            runtime_state.write_bytes(runtime_state.read_bytes() + b"x")
        else:
            target.write_bytes(target.read_bytes() + b"x")
    else:
        active_payload = torch.load(
            active.path, map_location="cpu", weights_only=True
        )
        registry = active_payload["metadata"]["circuits"][_CIRCUIT_A]
        transaction_root = Path(registry["transaction_root"])
        terminal = transaction_root / "events" / "0002-adapter_active.json"
        if tamper == "transaction_truncation":
            terminal.unlink()
        else:
            event = json.loads(terminal.read_text(encoding="utf-8"))
            event["lifecycle"] = "quarantine"
            event["event_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in event.items()
                    if key != "event_sha256"
                }
            )
            terminal.unlink()
            (transaction_root / "events" / "0002-quarantine.json").write_bytes(
                canonical_json_bytes(event)
            )

    with pytest.raises(
        CircuitRollbackError,
        match="transaction" if tamper.startswith("transaction") else tamper,
    ):
        create_rollback_child(active, snapshot, _CIRCUIT_A, tmp_path / "rejected.pt")
    assert not (tmp_path / "rejected.pt").exists()


def test_rollback_rejects_nonfinite_fresh_probe_without_publication(tmp_path: Path) -> None:
    active, snapshot, _, _, _ = _build_artifacts(
        tmp_path, probe=lambda model: torch.tensor(float("nan"))
    )
    output = tmp_path / "nonfinite.pt"

    with pytest.raises(CircuitRollbackError, match="non-finite"):
        create_rollback_child(active, snapshot, _CIRCUIT_A, output)
    assert not output.exists()


def test_rollback_refuses_existing_output(tmp_path: Path) -> None:
    active, snapshot, _, _, _ = _build_artifacts(tmp_path)
    output = tmp_path / "existing.pt"
    output.write_bytes(b"keep")

    with pytest.raises(CircuitRollbackError, match="exists"):
        create_rollback_child(active, snapshot, _CIRCUIT_A, output)
    assert output.read_bytes() == b"keep"


def test_rollback_atomic_no_replace_preserves_competing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, snapshot, _, _, _ = _build_artifacts(tmp_path)
    output = tmp_path / "race.pt"
    publish = rollback_module._publish_staging_no_replace

    def competing_publish(source: Path, destination: Path) -> None:
        destination.write_bytes(b"competitor")
        publish(source, destination)

    monkeypatch.setattr(
        rollback_module,
        "_publish_staging_no_replace",
        competing_publish,
    )

    with pytest.raises(CircuitRollbackError, match="appeared|exists"):
        create_rollback_child(active, snapshot, _CIRCUIT_A, output)

    assert output.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".race.pt.*.tmp"))


def test_rollback_publication_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, snapshot, _, _, _ = _build_artifacts(tmp_path)
    observed: list[Path] = []
    monkeypatch.setattr(
        rollback_module,
        "_fsync_directory",
        lambda path: observed.append(Path(path)),
    )

    create_rollback_child(
        active,
        snapshot,
        _CIRCUIT_A,
        tmp_path / "durable.pt",
    )

    assert observed == [tmp_path]


def test_rollback_rejects_symlink_output_ancestor_without_touching_external(
    tmp_path: Path,
) -> None:
    active, snapshot, _, _, _ = _build_artifacts(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(CircuitRollbackError, match="symlink|reparse|ancestor"):
        create_rollback_child(
            active,
            snapshot,
            _CIRCUIT_A,
            redirected / "rollback.pt",
        )

    assert not (external / "rollback.pt").exists()
