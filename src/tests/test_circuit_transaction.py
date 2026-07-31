from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from f51_darwin.circuits.transaction import (
    CircuitTransaction,
    CircuitTransactionError,
    GatedCircuitWrapper,
)
from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256


def _linear(value: float) -> nn.Linear:
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.fill_(value)
    return layer


@pytest.fixture
def inputs() -> torch.Tensor:
    return torch.tensor([[1.0, -2.0, 3.0, -4.0]])


def test_shadow_returns_exact_original_output_and_gate_zero_is_equivalent(
    inputs: torch.Tensor,
) -> None:
    original = _linear(1.0)
    candidate = _linear(-0.5)
    wrapper = GatedCircuitWrapper(original, candidate, mode="replace")

    baseline = original(inputs)
    shadow = wrapper(inputs, circuit_mode="shadow")
    gate_zero = wrapper(inputs, circuit_mode="active")

    assert shadow is not baseline  # Calls are independent; identity is checked below.
    torch.testing.assert_close(shadow, baseline, atol=0, rtol=0)
    torch.testing.assert_close(gate_zero, baseline, atol=1e-7, rtol=0)
    assert wrapper.external_gate.item() == 0.0


def test_shadow_returns_the_original_output_object_from_its_own_forward(
    inputs: torch.Tensor,
) -> None:
    # The wrapper must not reconstruct the object returned by its original call.
    observed: list[torch.Tensor] = []

    class CapturingOriginal(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            result = values.clone()
            observed.append(result)
            return result

    wrapper = GatedCircuitWrapper(CapturingOriginal(), nn.Identity(), mode="replace")
    assert wrapper(inputs, circuit_mode="shadow") is observed[-1]


def test_active_gate_changes_output_with_bounded_finite_residual(inputs: torch.Tensor) -> None:
    original = _linear(1.0)
    candidate = _linear(-0.5)
    wrapper = GatedCircuitWrapper(
        original, candidate, mode="replace", max_residual_ratio=0.25
    )
    baseline = original(inputs)
    with torch.no_grad():
        wrapper.external_gate.fill_(0.5)

    active = wrapper(inputs, circuit_mode="active")

    assert not torch.equal(active, baseline)
    assert bool(torch.isfinite(active).all())
    assert float((active - baseline).abs().max().detach()) <= 0.25 + 1e-6


@pytest.mark.parametrize("mode", ["inject", "replace"])
def test_tuple_selector_zero_preserves_other_tuple_members(
    inputs: torch.Tensor, mode: str
) -> None:
    class TupleOriginal(nn.Module):
        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return values * 2, values.sum(dim=-1)

    class TupleCandidate(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(values) * 9

    wrapper = GatedCircuitWrapper(
        TupleOriginal(), TupleCandidate(), mode=mode, output_selector=0
    )
    with torch.no_grad():
        wrapper.external_gate.fill_(0.5)

    baseline = wrapper.original(inputs)
    active = wrapper(inputs, circuit_mode="active")

    assert isinstance(active, tuple)
    torch.testing.assert_close(active[1], baseline[1], atol=0, rtol=0)
    assert not torch.equal(active[0], baseline[0])


def test_wrapper_rejects_incompatible_or_nonfinite_candidate_outputs(
    inputs: torch.Tensor,
) -> None:
    class WrongShape(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values[:, :3]

    class Nonfinite(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.full_like(values, float("nan"))

    for candidate, match in ((WrongShape(), "shape"), (Nonfinite(), "non-finite")):
        wrapper = GatedCircuitWrapper(nn.Identity(), candidate, mode="inject")
        with pytest.raises(CircuitTransactionError, match=match):
            wrapper(inputs, circuit_mode="active")


def test_shadow_executes_candidate_without_grad_and_returns_original_object(
    inputs: torch.Tensor,
) -> None:
    observed_grad_enabled: list[bool] = []

    class Candidate(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            observed_grad_enabled.append(torch.is_grad_enabled())
            return values

    wrapper = GatedCircuitWrapper(nn.Identity(), Candidate(), mode="replace")
    output = wrapper(inputs.requires_grad_(), circuit_mode="shadow")

    assert output is inputs
    assert observed_grad_enabled == [False]


def _create_transaction(
    root: Path, wrapper: GatedCircuitWrapper | None = None
) -> CircuitTransaction:
    return CircuitTransaction.create(
        root=root,
        transaction_id="transplant-001",
        operation="replace",
        circuit_id="f51-circuit-v1:" + "a" * 64,
        package_sha256="b" * 64,
        recipient_parent_sha256="c" * 64,
        tap_contract_sha256="d" * 64,
        preinstall_snapshot_sha256="e" * 64,
        wrapper=wrapper
        or GatedCircuitWrapper(_linear(1.0), _linear(-0.5), mode="replace"),
    )


def test_transaction_writes_immutable_manifest_and_lifecycle_events(tmp_path: Path) -> None:
    root = tmp_path / "transaction"
    transaction = _create_transaction(root)

    manifest = json.loads((root / "transaction.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "transaction_id": "transplant-001",
        "operation": "replace",
        "circuit_id": "f51-circuit-v1:" + "a" * 64,
        "package_sha256": "b" * 64,
        "recipient_parent_sha256": "c" * 64,
        "tap_contract_sha256": "d" * 64,
        "preinstall_snapshot_sha256": "e" * 64,
        "lifecycle": "candidate",
    }
    assert transaction.lifecycle == "candidate"
    assert (root / "events" / "0000-candidate.json").is_file()

    transaction.transition("shadow")
    assert transaction.lifecycle == "shadow"
    event = json.loads((root / "events" / "0001-shadow.json").read_text(encoding="utf-8"))
    assert event["from_lifecycle"] == "candidate"
    assert event["lifecycle"] == "shadow"
    current_manifest = json.loads((root / "transaction.json").read_text(encoding="utf-8"))
    assert current_manifest["lifecycle"] == "candidate"

    with pytest.raises(CircuitTransactionError, match="lifecycle transition"):
        transaction.transition("active")
    with pytest.raises(CircuitTransactionError, match="transaction root"):
        _create_transaction(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_sha256", "NOT-A-HASH"),
        ("recipient_parent_sha256", "A" * 64),
        ("tap_contract_sha256", "x" * 63),
        ("preinstall_snapshot_sha256", "f" * 63),
        ("circuit_id", "circuit-without-identity"),
    ],
)
def test_transaction_rejects_invalid_identities(
    tmp_path: Path, field: str, value: str
) -> None:
    values = {
        "root": tmp_path / field,
        "transaction_id": "transplant-001",
        "operation": "replace",
        "circuit_id": "f51-circuit-v1:" + "a" * 64,
        "package_sha256": "b" * 64,
        "recipient_parent_sha256": "c" * 64,
        "tap_contract_sha256": "d" * 64,
        "preinstall_snapshot_sha256": "e" * 64,
        "wrapper": GatedCircuitWrapper(_linear(1.0), _linear(-0.5), mode="replace"),
    }
    values[field] = value

    with pytest.raises(CircuitTransactionError, match="identity"):
        CircuitTransaction.create(**values)


@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "nonpersistent"])
def test_shadow_and_gate_zero_reject_and_restore_backbone_buffer_mutation(
    tmp_path: Path, inputs: torch.Tensor, persistent: bool
) -> None:
    class MutatingBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("counter", torch.zeros(()), persistent=persistent)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.counter.add_(1)
            return values

    original = MutatingBackbone()
    wrapper = GatedCircuitWrapper(original, nn.Identity(), mode="replace")
    transaction = _create_transaction(tmp_path / f"mutation-{persistent}", wrapper)

    with pytest.raises(CircuitTransactionError, match="backbone mutation"):
        transaction.verify_shadow([inputs])
    assert original.counter.item() == 0

    with pytest.raises(CircuitTransactionError, match="backbone mutation"):
        transaction.verify_gate_zero([inputs])
    assert original.counter.item() == 0


def test_shadow_and_gate_zero_return_reports_for_clean_backbone(
    tmp_path: Path, inputs: torch.Tensor
) -> None:
    transaction = _create_transaction(tmp_path / "clean")

    shadow = transaction.verify_shadow([inputs])
    gate_zero = transaction.verify_gate_zero([inputs])

    assert shadow.mode == "shadow"
    assert gate_zero.mode == "gate_zero"
    assert shadow.max_abs_error == 0.0
    assert gate_zero.max_abs_error <= 1e-7
    assert shadow.probes_run == gate_zero.probes_run == 1


def test_events_bind_the_exact_created_manifest_bytes(tmp_path: Path) -> None:
    root = tmp_path / "manifest-digest"
    transaction = _create_transaction(root)
    manifest_sha256 = hashlib.sha256((root / "transaction.json").read_bytes()).hexdigest()

    transaction.transition("shadow")

    for event_path in sorted((root / "events").iterdir()):
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        assert payload["manifest_sha256"] == manifest_sha256


def test_live_tail_anchor_rejects_truncated_terminal_event(tmp_path: Path) -> None:
    root = tmp_path / "truncated-tail"
    transaction = _create_transaction(root)
    transaction.transition("shadow")
    (root / "events" / "0001-shadow.json").unlink()
    before = sorted(path.name for path in (root / "events").iterdir())

    with pytest.raises(CircuitTransactionError, match="tail anchor"):
        transaction.transition("shadow")
    assert sorted(path.name for path in (root / "events").iterdir()) == before


def test_live_tail_anchor_rejects_canonical_complete_alternate_history(tmp_path: Path) -> None:
    root = tmp_path / "alternate-tail"
    transaction = _create_transaction(root)
    transaction.transition("shadow")
    transaction.transition("adapter_active")
    transaction.transition("organ_unfrozen")
    previous = json.loads((root / "events" / "0002-adapter_active.json").read_text("utf-8"))
    manifest_sha256 = hashlib.sha256((root / "transaction.json").read_bytes()).hexdigest()
    alternate = {
        "schema_version": 1,
        "transaction_id": transaction.transaction_id,
        "sequence": 3,
        "lifecycle": "quarantine",
        "from_lifecycle": "adapter_active",
        "manifest_sha256": manifest_sha256,
        "previous_event_sha256": previous["event_sha256"],
    }
    alternate["event_sha256"] = canonical_sha256(alternate)
    terminal = root / "events" / "0003-organ_unfrozen.json"
    terminal.unlink()
    replacement = root / "events" / "0003-quarantine.json"
    replacement.write_bytes(canonical_json_bytes(alternate))
    before = sorted(path.name for path in (root / "events").iterdir())

    with pytest.raises(CircuitTransactionError, match="tail anchor"):
        transaction.verify_shadow([torch.ones(1, 4)])
    assert sorted(path.name for path in (root / "events").iterdir()) == before


def test_trusted_tail_is_read_only_and_private_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "tail-memory-tamper"
    transaction = _create_transaction(root)
    before = sorted(path.name for path in (root / "events").iterdir())

    with pytest.raises(AttributeError):
        transaction._trusted_tail = transaction._trusted_tail  # type: ignore[misc]
    object.__setattr__(
        transaction,
        "_trusted_tail",
        replace(transaction._trusted_tail, count=transaction._trusted_tail.count + 1),
    )
    with pytest.raises(CircuitTransactionError, match="tail anchor"):
        transaction.verify_gate_zero([torch.ones(1, 4)])
    assert sorted(path.name for path in (root / "events").iterdir()) == before


def test_in_memory_identity_or_manifest_digest_tampering_cannot_append_event(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory-tamper"
    transaction = _create_transaction(root)
    before = sorted(path.name for path in (root / "events").iterdir())

    with pytest.raises(AttributeError):
        transaction.package_sha256 = "f" * 64  # type: ignore[misc]
    object.__setattr__(
        transaction,
        "_identity",
        replace(transaction._identity, package_sha256="f" * 64),
    )
    with pytest.raises(CircuitTransactionError, match="identity|manifest"):
        transaction.transition("shadow")
    assert sorted(path.name for path in (root / "events").iterdir()) == before

    object.__setattr__(transaction, "_manifest_sha256", "0" * 64)
    with pytest.raises(CircuitTransactionError, match="manifest"):
        transaction.verify_shadow([torch.ones(1, 4)])
    assert sorted(path.name for path in (root / "events").iterdir()) == before


def test_lifecycle_and_sequence_are_disk_derived_and_not_mutable(tmp_path: Path) -> None:
    root = tmp_path / "lifecycle-tamper"
    transaction = _create_transaction(root)
    before = sorted(path.name for path in (root / "events").iterdir())

    with pytest.raises(AttributeError):
        transaction.lifecycle = "shadow"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        object.__setattr__(transaction, "_event_sequence", 99)
    assert sorted(path.name for path in (root / "events").iterdir()) == before

    transaction.transition("shadow")
    assert transaction.lifecycle == "shadow"
    assert sorted(path.name for path in (root / "events").iterdir()) == [
        "0000-candidate.json",
        "0001-shadow.json",
    ]


@pytest.mark.parametrize("tamper", ["missing", "reordered", "extra", "malformed"])
def test_verification_rejects_tampered_append_only_event_history(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / tamper
    transaction = _create_transaction(root)
    candidate = root / "events" / "0000-candidate.json"
    if tamper == "missing":
        candidate.unlink()
    elif tamper == "reordered":
        candidate.rename(root / "events" / "0001-candidate.json")
    elif tamper == "extra":
        (root / "events" / "0001-shadow.json").write_text("{}", encoding="utf-8")
    else:
        candidate.write_text("not-json", encoding="utf-8")
    before = sorted(path.name for path in (root / "events").iterdir())

    with pytest.raises(CircuitTransactionError, match="event"):
        transaction.verify_gate_zero([torch.ones(1, 4)])
    assert sorted(path.name for path in (root / "events").iterdir()) == before


def test_manifest_file_tampering_blocks_verification_before_probe(tmp_path: Path) -> None:
    root = tmp_path / "manifest-tamper"
    transaction = _create_transaction(root)
    payload = json.loads((root / "transaction.json").read_text(encoding="utf-8"))
    payload["package_sha256"] = "f" * 64
    (root / "transaction.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(CircuitTransactionError, match="manifest"):
        transaction.verify_shadow([torch.ones(1, 4)])


@pytest.mark.parametrize("verify_name", ["verify_shadow", "verify_gate_zero"])
@pytest.mark.parametrize("mutation", ["parameter", "storage"])
def test_candidate_backbone_object_and_storage_mutation_is_restored(
    tmp_path: Path, inputs: torch.Tensor, verify_name: str, mutation: str
) -> None:
    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = _linear(1.0)
            self.tied = nn.Linear(4, 4, bias=False)
            self.tied.weight = self.linear.weight

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.linear(values)

    class MutatingCandidate(nn.Module):
        def __init__(self, backbone: Backbone) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            if mutation == "parameter":
                self.backbone.linear.weight = nn.Parameter(
                    self.backbone.linear.weight.detach().clone()
                )
            else:
                self.backbone.linear.weight.data = self.backbone.linear.weight.detach().clone()
            return self.backbone.linear(values)

    backbone = Backbone()
    original_weight = backbone.linear.weight
    external_view = original_weight.detach().view(-1)
    storage_cdata = original_weight.untyped_storage()._cdata
    data_ptr = original_weight.data_ptr()
    wrapper = GatedCircuitWrapper(backbone, MutatingCandidate(backbone), mode="replace")
    transaction = _create_transaction(tmp_path / f"{verify_name}-{mutation}", wrapper)

    with pytest.raises(CircuitTransactionError, match="backbone mutation"):
        getattr(transaction, verify_name)([inputs])

    assert backbone.linear.weight is original_weight
    assert backbone.tied.weight is original_weight
    assert original_weight.untyped_storage()._cdata == storage_cdata
    assert original_weight.data_ptr() == data_ptr
    with torch.no_grad():
        original_weight.view(-1)[0].add_(1.0)
    assert external_view[0].item() == original_weight.view(-1)[0].item()
