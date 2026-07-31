"""Tests for Native Layer Transplant Manager."""

import pytest
import torch
import torch.nn as nn
from f51_darwin.organism.layer_transplant import NativeLayerTransplantManager, measure_kl_divergence


class DummyLayer(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyModel(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DummyLayer(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def test_native_layer_transplant_swap() -> None:
    target_model = DummyModel(num_layers=4, dim=16)
    donor_layer = DummyLayer(dim=16)

    # Make donor weight distinct
    with torch.no_grad():
        donor_layer.linear.weight.add_(1.5)

    manager = NativeLayerTransplantManager(target_model)
    dist_before = manager.measure_weight_distance(donor_layer, target_layer_idx=2)
    assert dist_before > 0.0

    result = manager.swap_layer(
        donor_layer=donor_layer,
        target_layer_idx=2,
        donor_layer_idx=2,
        donor_provenance="dummy_donor",
        trainable=False,
    )

    assert result.target_layer_idx == 2
    assert result.weights_transplanted == 2  # weight + bias
    assert len(result.transplant_sha256) == 64

    # Verify weight was copied
    assert torch.allclose(target_model.blocks[2].linear.weight, donor_layer.linear.weight)
    assert not target_model.blocks[2].linear.weight.requires_grad


def test_measure_kl_divergence() -> None:
    model_a = DummyModel(num_layers=2, dim=8)
    model_b = DummyModel(num_layers=2, dim=8)

    # Identical weights -> KL 0
    model_b.load_state_dict(model_a.state_dict())
    x = torch.randn(2, 5, 8)

    kl_identical = measure_kl_divergence(model_a, model_b, x)
    assert abs(kl_identical) < 1e-5

    # Mutate -> KL > 0
    with torch.no_grad():
        model_b.blocks[1].linear.weight.add_(0.5)

    kl_diff = measure_kl_divergence(model_a, model_b, x)
    assert kl_diff >= 0.0
