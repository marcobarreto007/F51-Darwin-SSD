from __future__ import annotations

import pytest
import torch

from f51_darwin.cognition.adapters import ZeroGatedCognitiveAdapter
from f51_darwin.cognition.contracts import OrganKind, ResidualCondition


def _condition(batch: int = 2) -> ResidualCondition:
    return ResidualCondition(
        condition_id="memory-001",
        organ=OrganKind.MEMORY,
        values=torch.randn(batch, 1, 512),
        positions=torch.tensor([[2], [1]], dtype=torch.long),
    )


def test_gate_zero_is_bit_exact() -> None:
    torch.manual_seed(51)
    adapter = ZeroGatedCognitiveAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    result = adapter.apply_condition(hidden, _condition())
    assert torch.equal(result, hidden)
    assert adapter.gate.item() == 0.0


def test_nonzero_gate_changes_only_selected_positions_and_is_bounded() -> None:
    torch.manual_seed(51)
    adapter = ZeroGatedCognitiveAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(2, 4, 16)
    condition = _condition()
    with torch.no_grad():
        adapter.gate.fill_(100.0)
    result = adapter.apply_condition(hidden, condition)
    delta = result - hidden
    assert torch.count_nonzero(delta[0, :2]).item() == 0
    assert torch.count_nonzero(delta[0, 3:]).item() == 0
    assert torch.count_nonzero(delta[1, :1]).item() == 0
    assert torch.count_nonzero(delta[1, 2:]).item() == 0
    for batch, position in ((0, 2), (1, 1)):
        assert delta[batch, position].norm().item() <= (
            0.15 * hidden[batch, position].norm().item() + 1e-6
        )


def test_adapter_rejects_duplicate_and_out_of_range_positions() -> None:
    adapter = ZeroGatedCognitiveAdapter(d_model=16, max_scale=0.15)
    hidden = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="unique"):
        adapter.apply_condition(
            hidden,
            ResidualCondition(
                condition_id="dup",
                organ=OrganKind.MEMORY,
                values=torch.zeros(1, 2, 512),
                positions=torch.tensor([[1, 1]]),
            ),
        )
