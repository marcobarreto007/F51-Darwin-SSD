"""Invariantes da reducao por selecao.

O criterio de projeto e ficar dentro do grupo de simetria de permutacao:
as operacoes que preservam proveniencia sao exatamente as que preservam
funcao. Estes testes travam esse criterio no codigo.
"""

from __future__ import annotations

import pytest
import torch

from f51_darwin.transplant_16b.selection import (
    Provenance,
    block_influence,
    rank_residual_dims,
    retained_mass,
    select_layers,
    select_residual_dims,
    solve_diagonal_repair,
)


def test_operations_outside_the_symmetry_group_are_rejected() -> None:
    for forbidden in ("project", "average", "fit", "distill"):
        with pytest.raises(ValueError, match="fora do grupo de simetria"):
            Provenance(
                kind=forbidden, axis="layer", source_size=4, kept=(0, 1)
            )


def test_provenance_must_stay_bijective() -> None:
    with pytest.raises(ValueError, match="unique"):
        Provenance(kind="select", axis="neuron", source_size=4, kept=(1, 1))
    with pytest.raises(ValueError, match="inside the source axis"):
        Provenance(kind="select", axis="neuron", source_size=4, kept=(0, 9))


def test_provenance_answers_ancestry_exactly() -> None:
    provenance = Provenance(
        kind="drop", axis="layer", source_size=6, kept=(0, 2, 5)
    )
    assert provenance.target_size == 3
    assert provenance.dropped == (1, 3, 4)
    assert [provenance.source_of(i) for i in range(3)] == [0, 2, 5]
    # a identidade e estavel e serve de chave de ledger
    assert provenance.identity() == Provenance(
        kind="drop", axis="layer", source_size=6, kept=(0, 2, 5)
    ).identity()


def test_apply_slices_and_keeps_the_rows_intact() -> None:
    provenance = Provenance(
        kind="select", axis="residual_dim", source_size=5, kept=(0, 3, 4)
    )
    weight = torch.arange(20, dtype=torch.float32).reshape(4, 5)

    reduced = provenance.apply(weight, dim=1)

    assert reduced.shape == (4, 3)
    # cada coluna do alvo e byte-identica a coluna de origem
    for target_index in range(3):
        torch.testing.assert_close(
            reduced[:, target_index],
            weight[:, provenance.source_of(target_index)],
        )


def test_block_influence_is_zero_for_an_identity_layer() -> None:
    hidden = torch.randn(2, 6, 8)
    scores = block_influence([hidden, hidden.clone()])
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


def test_select_layers_never_drops_a_protected_layer() -> None:
    # a ultima camada domina: 0.72 contra ~0.06 do miolo, medido no SmolLM2
    influence = (0.30, 0.06, 0.05, 0.07, 0.72)

    provenance = select_layers(influence, target_layers=3, protected=(4,))

    assert 4 in provenance.kept
    assert provenance.kind == "drop"
    # descarta as de menor influencia entre as nao protegidas
    assert set(provenance.dropped) == {1, 2}


def test_selection_keeps_massive_activations_that_centering_discards() -> None:
    """Dimensao de magnitude enorme e variancia quase nula deve sobreviver.

    E o perfil das massive activations: viés independente da entrada,
    amarrado aos attention sinks. Uma SVD CENTRADA as ranqueia por ultimo,
    porque centrar remove exatamente a componente constante que elas sao.
    A selecao por contribuicao as mantem.
    """
    torch.manual_seed(3)
    tokens, hidden = 256, 16
    activations = torch.randn(tokens, hidden)
    sink = 7
    activations[:, sink] = 500.0 + torch.randn(tokens) * 0.01
    readout = torch.randn(64, hidden)

    scores = rank_residual_dims(activations, readout)
    provenance = select_residual_dims(scores, target_dim=8)

    assert sink in provenance.kept
    # o criterio que a SVD centrada usaria colocaria a dimensao por ultimo
    centered_rank = activations.std(dim=0).argsort(descending=True).tolist()
    assert centered_rank.index(sink) == hidden - 1
    assert retained_mass(scores, provenance) > 0.5


def test_diagonal_repair_recovers_an_exact_per_channel_gain() -> None:
    torch.manual_seed(5)
    current = torch.randn(128, 12)
    true_gain = torch.rand(12) * 2.0 + 0.5

    gain = solve_diagonal_repair(current, current * true_gain)

    torch.testing.assert_close(gain, true_gain, atol=1e-5, rtol=1e-5)


def test_diagonal_repair_is_neutral_on_dead_channels() -> None:
    current = torch.zeros(32, 4)
    current[:, 1] = torch.randn(32)

    gain = solve_diagonal_repair(current, torch.randn(32, 4))

    # canal morto nao inventa ganho
    assert gain[0] == pytest.approx(1.0)
    assert gain[2] == pytest.approx(1.0)
