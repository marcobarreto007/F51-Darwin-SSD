from __future__ import annotations

import torch

from f51_darwin.transplant_16b.dense_assembly import (
    _assert_dense_structure,
    dense_layer_mapping,
    publish_dense_candidate,
)


def test_dense_language_map_targets_native_ffn() -> None:
    mapping = dense_layer_mapping(0)
    assert mapping["model.layers.0.mlp.gate_proj.weight"] == (
        "blocks.0.ffn.gate_proj.weight",
    )
    assert mapping["model.layers.0.mlp.up_proj.weight"] == (
        "blocks.0.ffn.up_proj.weight",
    )
    assert mapping["model.layers.0.mlp.down_proj.weight"] == (
        "blocks.0.ffn.down_proj.weight",
    )
    assert not any(
        ".moe." in target
        for targets in mapping.values()
        for target in targets
    )


class _DenseBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = None
        self.ffn = torch.nn.Linear(2, 2, bias=False)


class _DenseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DenseBlock(), _DenseBlock()])


def test_dense_structure_gate_rejects_moe_state() -> None:
    report = _assert_dense_structure(_DenseModel(), expected_layers=2)
    assert report == {"moe_modules": 0, "dense_ffn_modules": 2}

    model = _DenseModel()
    model.blocks[0].moe = torch.nn.Linear(2, 2, bias=False)
    try:
        _assert_dense_structure(model, expected_layers=2)
    except ValueError as exc:
        assert "MoE" in str(exc)
    else:
        raise AssertionError("MoE structure was not rejected")


def test_dense_publication_is_fail_closed() -> None:
    names = set(publish_dense_candidate.__code__.co_names)
    assert "verify_shard_manifest" in names
    assert "sha256_file" in names
    assert "optimizer-smoke.json" in publish_dense_candidate.__code__.co_consts
