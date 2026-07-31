from __future__ import annotations

from dataclasses import replace

import pytest

from f51_darwin.circuits.manifest import CircuitManifest, TapContract, TensorIdentity


def _manifest() -> CircuitManifest:
    return CircuitManifest(
        source_checkpoint_sha256="b" * 64,
        source_base_checkpoint_id="100m-cycle-004",
        source_config_identity="f51-config-v1:" + "c" * 64,
        source_topology_identity="f51-topology-v1:" + "d" * 64,
        circuit_kind="predictor",
        constructor_id="predictor-v1",
        tensors=(
            TensorIdentity(
                key="predictor.weight",
                role="parameter",
                shape=(4, 4),
                dtype="torch.float32",
                sha256="a" * 64,
            ),
        ),
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
        source_checkpoint_path_hint="workspace/checkpoints/source.pt",
    )


def test_manifest_round_trip_and_identity_excludes_path_hint() -> None:
    manifest = _manifest()

    assert manifest == CircuitManifest.from_dict(manifest.to_dict())
    assert manifest.identity.startswith("f51-circuit-v1:")
    assert manifest.identity == replace(
        manifest,
        source_checkpoint_path_hint="elsewhere/source.pt",
    ).identity


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: TensorIdentity(
                key="predictor.weight",
                role="parameter",
                shape=(4, 4),
                dtype="torch.float32",
                sha256="not-a-sha256",
            ),
            "sha256",
        ),
        (
            lambda: TapContract(
                provider="module_path_v1",
                module_path="",
                position="post",
                output_selector="tensor",
                rank=3,
                width=4,
                minimum_sequence_length=2,
            ),
            "module_path",
        ),
        (
            lambda: TapContract(
                provider="module_path_v1",
                module_path="blocks.1",
                position="post",
                output_selector="tensor",
                rank=3,
                width=-1,
                minimum_sequence_length=2,
            ),
            "width",
        ),
        (
            lambda: TapContract(
                provider="module_path_v1",
                module_path="blocks.1",
                position="between",
                output_selector="tensor",
                rank=3,
                width=4,
                minimum_sequence_length=2,
            ),
            "position",
        ),
    ],
)
def test_manifest_components_reject_invalid_values(factory: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()  # type: ignore[operator]


def test_manifest_from_dict_rejects_unknown_or_missing_fields() -> None:
    payload = _manifest().to_dict()

    with pytest.raises(ValueError, match="unknown"):
        CircuitManifest.from_dict({**payload, "unexpected": True})
    with pytest.raises(ValueError, match="missing"):
        CircuitManifest.from_dict({key: value for key, value in payload.items() if key != "tap"})


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_manifest_rejects_non_integer_schema_version(schema_version: object) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        CircuitManifest(**{**_manifest().__dict__, "schema_version": schema_version})

    with pytest.raises(ValueError, match="schema_version"):
        CircuitManifest.from_dict({**_manifest().to_dict(), "schema_version": schema_version})
