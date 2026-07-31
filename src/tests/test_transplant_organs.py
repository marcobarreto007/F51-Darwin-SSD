from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer
from f51_darwin.jepa_v2 import JEPAHeadV2
from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
from f51_darwin.inter_hemispheric import HemisphereConfig, InterHemisphericSystem
from f51_darwin.transplant.bundle import (
    OrganBundle,
    OrganBundleManifest,
    TensorRecord,
    tensor_sha256,
)
from f51_darwin.transplant.organs import (
    build_original_gaba,
    build_original_heartbeat,
    build_original_ihs,
    build_original_jepa,
)


def _bundle() -> OrganBundle:
    torch.manual_seed(52)
    jepa = JEPAHeadV2(
        d_model=512,
        hidden_dim=768,
        dropout=0.0,
        bottleneck_dim=256,
    )
    gaba = GABAergicLayer(GABAConfig(d_model=512))
    tensors = {
        **{
            f"jepa_predictor.{key}": value.detach().clone()
            for key, value in jepa.state_dict().items()
        },
        **{
            f"blocks.0.gaba.{key}": value.detach().clone()
            for key, value in gaba.state_dict().items()
        },
    }
    records = tuple(
        TensorRecord(
            key=key,
            shape=tuple(tensors[key].shape),
            dtype=str(tensors[key].dtype),
            sha256=tensor_sha256(tensors[key]),
        )
        for key in sorted(tensors)
    )
    return OrganBundle(
        manifest=OrganBundleManifest(
            schema_version=1,
            source_checkpoint="donor.pt",
            source_checkpoint_sha256="a" * 64,
            base_checkpoint_id="core:test",
            organs=("jepa", "gaba.0"),
            tensors=records,
        ),
        tensors=tensors,
        donor_config={
            "d_model": 512,
            "jepa_weight": 0.05,
            "gaba_enabled": True,
        },
        runtime_state={},
    )


def test_build_original_jepa_loads_every_donor_tensor() -> None:
    bundle = _bundle()
    transplant = build_original_jepa(bundle)

    assert isinstance(transplant.module, JEPAHeadV2)
    assert transplant.module.d_model == 512
    assert transplant.source_keys == tuple(
        key
        for key in sorted(bundle.tensors)
        if key.startswith("jepa_predictor.")
    )
    assert transplant.missing_keys == ()
    assert transplant.unexpected_keys == ()
    assert all(not parameter.requires_grad for parameter in transplant.module.parameters())
    for key, value in transplant.module.state_dict().items():
        assert torch.equal(value, bundle.tensors[f"jepa_predictor.{key}"])


def test_build_original_gaba_preserves_parameters_and_buffers() -> None:
    bundle = _bundle()
    transplant = build_original_gaba(bundle, donor_layer=0)

    assert isinstance(transplant.module, GABAergicLayer)
    assert transplant.module.config.d_model == 512
    assert transplant.missing_keys == ()
    assert transplant.unexpected_keys == ()
    assert all(not parameter.requires_grad for parameter in transplant.module.parameters())
    for key, value in transplant.module.state_dict().items():
        assert torch.equal(value, bundle.tensors[f"blocks.0.gaba.{key}"])


def test_build_original_jepa_rejects_incomplete_organ() -> None:
    bundle = _bundle()
    removed = "jepa_predictor.predictor.9.bias"
    tensors = dict(bundle.tensors)
    tensors.pop(removed)
    records = tuple(
        record for record in bundle.manifest.tensors if record.key != removed
    )
    incomplete = replace(
        bundle,
        manifest=replace(bundle.manifest, tensors=records),
        tensors=tensors,
    )

    with pytest.raises(ValueError, match="incomplete jepa organ"):
        build_original_jepa(incomplete)


def test_build_original_gaba_rejects_unrequested_layer() -> None:
    with pytest.raises(ValueError, match="not declared in bundle"):
        build_original_gaba(_bundle(), donor_layer=1)


def _heartbeat_ihs_bundle() -> tuple[OrganBundle, dict, dict]:
    torch.manual_seed(73)
    heartbeat = Heartbeat(
        512,
        HeartbeatConfig(
            think_interval=1,
            explore_interval=10,
            ff_layers=2,
            memory_capacity=8,
            surprise_threshold=0.3,
        ),
    )
    heartbeat.total_beats = 7
    heartbeat.dopamine = 0.625
    heartbeat_state = heartbeat.state_dict()
    ihs = InterHemisphericSystem(
        HemisphereConfig(d_model=512, n_heads=8, dropout=0.0)
    ).eval()
    ihs_state = {
        f"inter_hemispheric.{key}": value.detach().clone()
        for key, value in ihs.state_dict().items()
    }
    heartbeat_tensors = {}
    for state_key, prefix in (
        ("ff_stack_state_dict", "heartbeat.ff_stack."),
        ("tt_memory_state_dict", "heartbeat.tt_memory."),
        ("thinker_state_dict", "heartbeat.thinker."),
    ):
        heartbeat_tensors.update(
            {
                f"{prefix}{key}": value.detach().clone()
                for key, value in heartbeat_state[state_key].items()
            }
        )
    tensors = {**ihs_state, **heartbeat_tensors}
    records = tuple(
        TensorRecord(
            key=key,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            sha256=tensor_sha256(value),
        )
        for key, value in sorted(tensors.items())
    )
    runtime_heartbeat = {
        key: value
        for key, value in heartbeat_state.items()
        if key not in {
            "ff_stack_state_dict",
            "tt_memory_state_dict",
            "thinker_state_dict",
        }
    }
    bundle = OrganBundle(
        manifest=OrganBundleManifest(
            schema_version=1,
            source_checkpoint="synthetic.pt",
            source_checkpoint_sha256="c" * 64,
            base_checkpoint_id="core:test",
            organs=("ihs", "heartbeat"),
            tensors=records,
        ),
        tensors=tensors,
        donor_config={
            "d_model": 512,
            "n_heads": 8,
            "dropout": 0.0,
            "heartbeat_think_interval": 1,
            "heartbeat_explore_interval": 10,
            "heartbeat_ff_layers": 2,
            "heartbeat_memory_capacity": 8,
            "heartbeat_surprise_threshold": 0.3,
        },
        runtime_state={"heartbeat_state": runtime_heartbeat},
    )
    return bundle, heartbeat_state, ihs.state_dict()


def test_build_original_ihs_preserves_all_parameters() -> None:
    bundle, _, expected = _heartbeat_ihs_bundle()

    transplant = build_original_ihs(bundle)

    assert transplant.name == "ihs"
    assert transplant.missing_keys == ()
    assert transplant.unexpected_keys == ()
    assert all(not parameter.requires_grad for parameter in transplant.module.parameters())
    for key, value in transplant.module.state_dict().items():
        assert torch.equal(value, expected[key])


def test_build_original_heartbeat_preserves_modules_and_runtime_state() -> None:
    bundle, expected, _ = _heartbeat_ihs_bundle()

    transplant = build_original_heartbeat(bundle)
    restored = transplant.module.export_state()

    assert transplant.name == "heartbeat"
    assert restored["beat"] == 7
    assert restored["dopamine"] == pytest.approx(0.625)
    for component in (
        "ff_stack_state_dict",
        "tt_memory_state_dict",
        "thinker_state_dict",
    ):
        for key, value in expected[component].items():
            assert torch.equal(restored[component][key], value)
    assert all(not parameter.requires_grad for parameter in transplant.module.parameters())
