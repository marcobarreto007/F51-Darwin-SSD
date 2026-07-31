from __future__ import annotations

import torch
from torch import nn

import pytest

from f51_darwin.circuits import TapContract
from f51_darwin.circuits.ablation import discover_residual_channels, run_paired_ablation
from f51_darwin.circuits.taps import CircuitTapError, TapCapture, resolve_module


class ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.head = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.blocks[0].weight.copy_(torch.eye(4))
            self.head.weight.copy_(
                torch.tensor([[2.0, -2.0, 0.0, 0.0], [-2.0, 2.0, 0.0, 0.0]])
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks[0](values))


def _tap(*, width: int = 4) -> TapContract:
    return TapContract(
        provider="module_path_v1",
        module_path="blocks.0",
        position="post",
        output_selector="tensor",
        rank=2,
        width=width,
        minimum_sequence_length=1,
    )


def test_resolve_module_is_strict_and_resolves_registered_module_list_child() -> None:
    model = ToyClassifier()

    assert resolve_module(model, "blocks.0") is model.blocks[0]
    for path in ("blocks.missing", "_private", "blocks._0", "blocks.zero", "blocks.9"):
        with pytest.raises(CircuitTapError):
            resolve_module(model, path)


def test_tap_validates_output_before_running_intervention() -> None:
    model = ToyClassifier()
    intervention_called = False

    def intervention(output: torch.Tensor) -> torch.Tensor:
        nonlocal intervention_called
        intervention_called = True
        return output

    with TapCapture(model, _tap(width=3), intervention=intervention):
        with pytest.raises(CircuitTapError, match="width"):
            model(torch.ones(2, 4))

    assert intervention_called is False


def test_discovery_ranks_gradient_times_activation_on_discovery_split() -> None:
    model = ToyClassifier()
    discovery_batches = [
        (
            torch.tensor(
                [
                    [3.0, 0.0, 20.0, -20.0],
                    [0.0, 3.0, -20.0, 20.0],
                    [4.0, 0.0, 15.0, 15.0],
                    [0.0, 4.0, -15.0, -15.0],
                ]
            ),
            torch.tensor([0, 1, 0, 1]),
        )
    ]

    selector = discover_residual_channels(model, discovery_batches, _tap(), count=2)

    assert selector.layer_path == "blocks.0"
    assert set(selector.channels) == {0, 1}
    assert len(selector.baseline) == 2
    assert selector.discovery_input_sha256 is not None
    assert len(selector.discovery_item_sha256) == 4
    with pytest.raises(ValueError, match="discovery split"):
        run_paired_ablation(lambda: ToyClassifier(), discovery_batches, selector, seed=51)
    discovery_inputs, discovery_labels = discovery_batches[0]
    with pytest.raises(ValueError, match="discovery split"):
        run_paired_ablation(
            lambda: ToyClassifier(),
            [(discovery_inputs[[3, 0]], discovery_labels[[3, 0]])],
            selector,
            seed=51,
        )
    with pytest.raises(ValueError, match="discovery split"):
        run_paired_ablation(
            lambda: ToyClassifier(),
            [(discovery_inputs[[1, 2]], 1 - discovery_labels[[1, 2]])],
            selector,
            seed=51,
        )

    confirmation_inputs = discovery_inputs.clone()
    confirmation_inputs[:, 2:] += 1.0
    confirmation = run_paired_ablation(
        lambda: ToyClassifier(),
        [(confirmation_inputs, discovery_labels)],
        selector,
        seed=51,
    )
    assert not set(selector.discovery_item_sha256).intersection(
        confirmation.confirmation_item_sha256
    )


def test_discovery_rejects_count_larger_than_tap_width() -> None:
    with pytest.raises(ValueError, match="width"):
        discover_residual_channels(
            ToyClassifier(),
            [(torch.ones(2, 4), torch.zeros(2, dtype=torch.long))],
            _tap(),
            count=5,
        )
