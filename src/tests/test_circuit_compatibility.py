from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from f51_darwin.circuits import CircuitManifest, TapContract, TensorIdentity
from f51_darwin.circuits.compatibility import (
    CircuitCompatibilityError,
    RecipientIdentity,
    model_state_sha256,
    package_content_sha256,
    preflight_circuit,
)
from f51_darwin.circuits.package import LoadedCircuitPackage, PackageIdentity, tensor_sha256
from f51_darwin.circuits.taps import TapCapture


class ToyRecipient(nn.Module):
    def __init__(
        self,
        *,
        width: int = 4,
        mutate: bool = False,
        nonfinite: bool = False,
        mutate_nonpersistent: bool = False,
        replace_parameter: bool = False,
        replace_nonpersistent: bool = False,
        replace_parameter_storage: bool = False,
        replace_buffer_storage: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, width, bias=False)])
        self.head = nn.Linear(width, 2, bias=False)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "nonpersistent_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        storage_base = torch.arange(4, dtype=torch.float32)
        self.register_buffer("storage_base", storage_base, persistent=False)
        self.register_buffer("storage_view", storage_base[1:3], persistent=False)
        self.mutate = mutate
        self.nonfinite = nonfinite
        self.mutate_nonpersistent = mutate_nonpersistent
        self.replace_parameter = replace_parameter
        self.replace_nonpersistent = replace_nonpersistent
        self.replace_parameter_storage = replace_parameter_storage
        self.replace_buffer_storage = replace_buffer_storage

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.mutate:
            self.forward_count.add_(1)
        if self.mutate_nonpersistent:
            self.nonpersistent_count.add_(1)
        if self.replace_parameter:
            self.head.weight = nn.Parameter(
                self.head.weight.detach().clone(),
                requires_grad=self.head.weight.requires_grad,
            )
        if self.replace_nonpersistent:
            self._buffers["nonpersistent_count"] = (
                self.nonpersistent_count.detach().clone()
            )
        if self.replace_parameter_storage:
            self.head.weight.data = self.head.weight.detach().clone()
        if self.replace_buffer_storage:
            self.storage_base.data = self.storage_base.detach().clone()
        hidden = self.blocks[0](values)
        logits = self.head(hidden)
        if self.nonfinite:
            logits = logits * torch.tensor(float("nan"))
        return logits


def _package(
    *,
    family: str = "toy",
    tap: TapContract | None = None,
    runtime_state=None,
    candidate_value: float = 1.0,
    constructor_id: str = "residual-v1",
    identity_sha256: str = "d" * 64,
):
    tensor = torch.tensor([candidate_value])
    manifest = CircuitManifest(
        source_checkpoint_sha256="b" * 64,
        source_base_checkpoint_id="toy",
        source_config_identity="f51-config-v1:" + "c" * 64,
        source_topology_identity="f51-topology-v1:" + "d" * 64,
        circuit_kind="residual",
        constructor_id=constructor_id,
        tensors=(
            TensorIdentity(
                key="candidate.scale",
                role="parameter",
                shape=(1,),
                dtype=str(tensor.dtype),
                sha256=tensor_sha256(tensor),
            ),
        ),
        tap=tap
        or TapContract(
            provider="module_path_v1",
            module_path="blocks.0",
            position="post",
            output_selector="tensor",
            rank=2,
            width=4,
            minimum_sequence_length=2,
        ),
        accepted_recipient_families=(family,),
    )
    base_runtime_state = dict(runtime_state or {})
    unsigned = LoadedCircuitPackage(
        manifest=manifest,
        tensors={"candidate.scale": tensor},
        runtime_state=base_runtime_state,
        identity=PackageIdentity(identity_sha256),
    )
    authenticated_runtime_state = {
        **base_runtime_state,
        "package_content_sha256": package_content_sha256(unsigned),
    }
    return LoadedCircuitPackage(
        manifest=manifest,
        tensors={"candidate.scale": tensor},
        runtime_state=authenticated_runtime_state,
        identity=PackageIdentity(identity_sha256),
    )


def _identity(
    model: ToyRecipient,
    package: LoadedCircuitPackage | None = None,
    **changes,
) -> RecipientIdentity:
    approved_package = package or _package()
    identity = RecipientIdentity(
        sha256=model_state_sha256(model),
        family="toy",
        tap_rank=2,
        tap_width=4,
        minimum_sequence_length=2,
        supports_mask=False,
        supports_cache=False,
        expected_package_sha256=approved_package.identity.sha256,
        expected_package_content_sha256=approved_package.runtime_state[
            "package_content_sha256"
        ],
    )
    return replace(identity, **changes)


def test_compatible_preflight_proves_shadow_gate_zero_and_no_mutation() -> None:
    recipient = ToyRecipient()
    report = preflight_circuit(
        _package(),
        recipient,
        _identity(recipient),
        [torch.ones(3, 4)],
    )

    assert report.compatible is True
    assert report.launch is False
    assert report.max_abs_shadow_error <= 1e-5
    assert report.gate_zero_max_abs_logit_error <= 1e-5
    assert report.state_mutations == 0
    assert report.outputs_finite is True


@pytest.mark.parametrize(
    ("identity_changes", "package_changes", "match"),
    [
        ({"sha256": "f" * 64}, {}, "SHA-256 mismatch"),
        ({"family": "other"}, {}, "family"),
        ({"tap_width": 3}, {}, "width"),
        ({"tap_rank": 3}, {}, "rank"),
        ({"minimum_sequence_length": 1}, {}, "sequence"),
    ],
)
def test_preflight_rejects_identity_and_static_interface_mismatches(
    identity_changes: dict, package_changes: dict, match: str
) -> None:
    recipient = ToyRecipient()
    with pytest.raises(CircuitCompatibilityError, match=match):
        preflight_circuit(
            _package(**package_changes),
            recipient,
            _identity(recipient, **identity_changes),
            [torch.ones(3, 4)],
        )


def test_preflight_rejects_unresolved_and_occupied_taps() -> None:
    recipient = ToyRecipient()
    missing = replace(_package().manifest.tap, module_path="blocks.9")
    missing_package = _package(tap=missing)
    with pytest.raises(CircuitCompatibilityError, match="tap"):
        preflight_circuit(
            missing_package,
            recipient,
            _identity(recipient, missing_package),
            [torch.ones(3, 4)],
        )

    with TapCapture(recipient, _package().manifest.tap):
        with pytest.raises(CircuitCompatibilityError, match="occupied"):
            preflight_circuit(_package(), recipient, _identity(recipient), [torch.ones(3, 4)])


@pytest.mark.parametrize("requirement", ["requires_mask", "requires_cache"])
def test_preflight_rejects_unsupported_mask_or_cache(requirement: str) -> None:
    recipient = ToyRecipient()
    package = _package(runtime_state={requirement: True})
    with pytest.raises(CircuitCompatibilityError, match="unsupported"):
        preflight_circuit(
            package,
            recipient,
            _identity(recipient, package),
            [torch.ones(3, 4)],
        )


def test_preflight_rejects_unknown_constructor_and_runtime_nonfinite_candidate() -> None:
    recipient = ToyRecipient()
    unknown = _package(constructor_id="unregistered-v9")
    with pytest.raises(CircuitCompatibilityError, match="constructor"):
        preflight_circuit(
            unknown,
            recipient,
            _identity(recipient, unknown),
            [torch.ones(3, 4)],
        )

    unsupported_dependencies = _package(
        runtime_state={"dependencies": ["activation", "mask"]}
    )
    with pytest.raises(CircuitCompatibilityError, match="dependencies"):
        preflight_circuit(
            unsupported_dependencies,
            recipient,
            _identity(
                recipient,
                unsupported_dependencies,
                supports_mask=True,
            ),
            [torch.ones(3, 4)],
        )

    nonfinite = _package(candidate_value=1e20)
    report = preflight_circuit(
        nonfinite,
        recipient,
        _identity(recipient, nonfinite),
        [torch.full((3, 4), 1e20)],
    )
    assert report.compatible is False
    assert report.outputs_finite is False
    assert report.launch is False


def test_preflight_rejects_untrusted_or_current_content_mismatched_package() -> None:
    recipient = ToyRecipient()
    package = _package()
    arbitrary_identity = replace(package, identity=PackageIdentity("e" * 64))
    with pytest.raises(CircuitCompatibilityError, match="package SHA-256"):
        preflight_circuit(
            arbitrary_identity,
            recipient,
            _identity(recipient),
            [torch.ones(3, 4)],
        )

    forged_unsigned = replace(
        package,
        runtime_state={**package.runtime_state, "dependencies": ["undeclared"]},
    )
    tampered_runtime = replace(
        forged_unsigned,
        runtime_state={
            **forged_unsigned.runtime_state,
            "package_content_sha256": package_content_sha256(forged_unsigned),
        },
    )
    with pytest.raises(CircuitCompatibilityError, match="content proof"):
        preflight_circuit(
            tampered_runtime,
            recipient,
            _identity(recipient),
            [torch.ones(3, 4)],
        )


def test_preflight_fails_closed_on_nonfinite_output_and_backbone_mutation() -> None:
    recipient = ToyRecipient()
    nonfinite_package = _package(candidate_value=float("nan"))
    with pytest.raises(CircuitCompatibilityError, match="tensor identity"):
        preflight_circuit(
            nonfinite_package,
            recipient,
            _identity(recipient, nonfinite_package),
            [torch.ones(3, 4)],
        )

    nonfinite = ToyRecipient(nonfinite=True)
    report = preflight_circuit(
        _package(), nonfinite, _identity(nonfinite), [torch.ones(3, 4)]
    )
    assert report.compatible is False
    assert report.outputs_finite is False
    assert report.launch is False

    mutating = ToyRecipient(mutate=True)
    original = mutating.forward_count.clone()
    report = preflight_circuit(
        _package(), mutating, _identity(mutating), [torch.ones(3, 4)]
    )
    assert report.compatible is False
    assert report.state_mutations > 0
    assert torch.equal(mutating.forward_count, original)

    nonpersistent = ToyRecipient(mutate_nonpersistent=True)
    original_nonpersistent = nonpersistent.nonpersistent_count.clone()
    report = preflight_circuit(
        _package(),
        nonpersistent,
        _identity(nonpersistent),
        [torch.ones(3, 4)],
    )
    assert report.compatible is False
    assert report.state_mutations > 0
    assert torch.equal(nonpersistent.nonpersistent_count, original_nonpersistent)


@pytest.mark.parametrize(
    ("recipient", "attribute_path"),
    [
        (ToyRecipient(replace_parameter=True), "head.weight"),
        (ToyRecipient(replace_nonpersistent=True), "nonpersistent_count"),
    ],
)
def test_preflight_restores_original_registered_objects(
    recipient: ToyRecipient,
    attribute_path: str,
) -> None:
    owner: nn.Module = recipient
    segments = attribute_path.split(".")
    for segment in segments[:-1]:
        owner = getattr(owner, segment)
    original = getattr(owner, segments[-1])

    report = preflight_circuit(
        _package(),
        recipient,
        _identity(recipient),
        [torch.ones(3, 4)],
    )

    restored_owner: nn.Module = recipient
    for segment in segments[:-1]:
        restored_owner = getattr(restored_owner, segment)
    assert report.compatible is False
    assert report.state_mutations > 0
    assert getattr(restored_owner, segments[-1]) is original


def test_preflight_restores_tied_parameter_alias_relationship() -> None:
    recipient = ToyRecipient(replace_parameter=True)
    recipient.tied_head = nn.Linear(4, 2, bias=False)
    recipient.tied_head.weight = recipient.head.weight
    original = recipient.head.weight

    report = preflight_circuit(
        _package(),
        recipient,
        _identity(recipient),
        [torch.ones(3, 4)],
    )

    assert report.compatible is False
    assert report.state_mutations > 0
    assert recipient.head.weight is original
    assert recipient.tied_head.weight is original


def test_clean_preflight_preserves_storage_and_external_view_link() -> None:
    recipient = ToyRecipient()
    original = recipient.head.weight
    external_view = original.view(-1)
    data_ptr = original.data_ptr()
    storage_ptr = original.untyped_storage().data_ptr()
    stride = original.stride()
    storage_offset = original.storage_offset()

    report = preflight_circuit(
        _package(),
        recipient,
        _identity(recipient),
        [torch.ones(3, 4)],
    )

    assert report.compatible is True
    assert original.data_ptr() == data_ptr
    assert original.untyped_storage().data_ptr() == storage_ptr
    assert original.stride() == stride
    assert original.storage_offset() == storage_offset
    with torch.no_grad():
        original.view(-1)[0].add_(1.0)
    assert external_view[0].item() == original.view(-1)[0].item()


def test_preflight_restores_replaced_parameter_storage_and_view_topology() -> None:
    recipient = ToyRecipient(replace_parameter_storage=True)
    original = recipient.head.weight
    external_view = original.view(-1)
    content = original.detach().clone()
    storage = original.untyped_storage()
    data_ptr = original.data_ptr()
    stride = original.stride()
    storage_offset = original.storage_offset()

    report = preflight_circuit(
        _package(),
        recipient,
        _identity(recipient),
        [torch.ones(3, 4)],
    )

    assert report.compatible is False
    assert report.state_mutations > 0
    assert recipient.head.weight is original
    assert original.untyped_storage()._cdata == storage._cdata
    assert original.data_ptr() == data_ptr
    assert original.stride() == stride
    assert original.storage_offset() == storage_offset
    assert torch.equal(original, content)
    with torch.no_grad():
        original.view(-1)[0].add_(1.0)
    assert external_view[0].item() == original.view(-1)[0].item()


def test_preflight_restores_nonpersistent_buffer_storage_alias() -> None:
    recipient = ToyRecipient(replace_buffer_storage=True)
    original_base = recipient.storage_base
    original_view = recipient.storage_view
    base_content = original_base.clone()
    view_content = original_view.clone()
    storage = original_base.untyped_storage()
    assert original_view.untyped_storage()._cdata == storage._cdata

    report = preflight_circuit(
        _package(),
        recipient,
        _identity(recipient),
        [torch.ones(3, 4)],
    )

    assert report.compatible is False
    assert report.state_mutations > 0
    assert recipient.storage_base is original_base
    assert recipient.storage_view is original_view
    assert original_base.untyped_storage()._cdata == storage._cdata
    assert original_view.untyped_storage()._cdata == storage._cdata
    assert torch.equal(original_base, base_content)
    assert torch.equal(original_view, view_content)
