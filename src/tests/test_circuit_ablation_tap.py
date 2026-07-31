"""run_paired_ablation com TapContract explícito.

Sem tap a intervenção usa `_RuntimeTap`, um forward hook *post* que exige
saída tensor — o que exclui o residual stream, cujo ponto natural é o *pre*
hook de um bloco que devolve tupla. Estes testes cobrem a passagem do tap.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from f51_darwin.circuits.ablation import (
    CircuitSelector,
    discover_residual_channels,
    run_paired_ablation,
)
from f51_darwin.circuits.manifest import TapContract
from f51_darwin.circuits.taps import CircuitTapError

WIDTH = 8


class TupleBlock(nn.Module):
    """Bloco que devolve tupla, como o GPT2Block — post hook não serve."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden: torch.Tensor):
        return (self.linear(hidden), None)


class TupleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(WIDTH, WIDTH)
        self.block = TupleBlock()
        self.head = nn.Linear(WIDTH, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(inputs)
        hidden, _ = self.block(hidden)
        return self.head(hidden)


def _factory():
    torch.manual_seed(7)
    return TupleModel()


def _batches(count: int = 2, size: int = 8, length: int = 4, seed: int = 11):
    """Splits distintos precisam de sementes distintas — a confirmação não
    pode reusar itens da descoberta, e o runtime recusa se reusar."""
    torch.manual_seed(seed)
    out = []
    for _ in range(count):
        inputs = torch.randn(size, length, WIDTH)
        labels = torch.randint(0, 3, (size, length))
        out.append((inputs, labels, "toy"))
    return out


def _confirmation(count: int = 3):
    return _batches(count=count, seed=99)


def _pre_tap() -> TapContract:
    return TapContract(
        provider="module_path_v1",
        module_path="block",
        position="pre",
        output_selector="tensor",
        rank=3,
        width=WIDTH,
        minimum_sequence_length=2,
    )


def _selector(model, batches, tap, count=3) -> CircuitSelector:
    return discover_residual_channels(model, batches, tap, count)


def test_post_hook_alone_cannot_reach_a_tuple_block() -> None:
    """Sem tap, o caminho antigo recusa o bloco — a lacuna que motivou a mudança."""
    tap = _pre_tap()
    model = _factory()
    selector = _selector(model, _batches(), tap)
    with pytest.raises(CircuitTapError, match="tensor module output"):
        run_paired_ablation(_factory, _confirmation(), selector, seed=5)


def test_explicit_pre_tap_runs_every_arm() -> None:
    tap = _pre_tap()
    model = _factory()
    selector = _selector(model, _batches(), tap)
    result = run_paired_ablation(_factory, _confirmation(), selector, seed=5, tap=tap)
    assert result.tap_width == WIDTH
    assert result.anchors_identical is True
    # Ablação precisa mover a métrica; restauração precisa trazer de volta.
    assert result.ablate.mean_nll != result.clean.mean_nll
    assert result.restore.max_abs_logit_error <= 1e-6


def test_tap_must_match_the_selector_layer() -> None:
    tap = _pre_tap()
    model = _factory()
    selector = _selector(model, _batches(), tap)
    mismatched = TapContract(
        provider="module_path_v1",
        module_path="stem",
        position="pre",
        output_selector="tensor",
        rank=3,
        width=WIDTH,
        minimum_sequence_length=2,
    )
    with pytest.raises(ValueError, match="match the selector layer_path"):
        run_paired_ablation(
            _factory, _confirmation(), selector, seed=5, tap=mismatched
        )


def test_tap_argument_is_type_checked() -> None:
    tap = _pre_tap()
    model = _factory()
    selector = _selector(model, _batches(), tap)
    with pytest.raises(ValueError, match="tap must be a TapContract"):
        run_paired_ablation(
            _factory, _confirmation(), selector, seed=5, tap={"module_path": "block"}
        )


def test_tap_contract_still_validates_shape() -> None:
    wrong_width = TapContract(
        provider="module_path_v1",
        module_path="block",
        position="pre",
        output_selector="tensor",
        rank=3,
        width=WIDTH + 1,
        minimum_sequence_length=2,
    )
    model = _factory()
    with pytest.raises(CircuitTapError, match="width mismatch"):
        discover_residual_channels(model, _batches(), wrong_width, 2)


def test_omitting_the_tap_keeps_the_tensor_only_path_working() -> None:
    """Compatibilidade: modelo de saída tensor continua funcionando sem tap."""

    class PlainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(WIDTH, WIDTH)
            self.head = nn.Linear(WIDTH, 3)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.head(self.stem(inputs))

    def plain_factory():
        torch.manual_seed(3)
        return PlainModel()

    tap = TapContract(
        provider="module_path_v1",
        module_path="stem",
        position="post",
        output_selector="tensor",
        rank=3,
        width=WIDTH,
        minimum_sequence_length=2,
    )
    selector = discover_residual_channels(plain_factory(), _batches(), tap, 3)
    result = run_paired_ablation(plain_factory, _confirmation(), selector, seed=9)
    assert result.tap_width == WIDTH
