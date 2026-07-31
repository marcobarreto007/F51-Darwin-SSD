"""A fronteira permutacao<->byte, e o acoplamento SHA + deriva.

Estes testes existem para fixar a tese, nao so o codigo: as operacoes que
preservam a funcao sob simetria de permutacao sao exatamente as que
preservam o valor escalar, logo um hash criptografico e um certificado
valido de heranca precisamente para elas -- e sua quebra e um alarme
correto para as demais.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from f51_darwin.transplant_16b.ledger import (
    CoverageLedger,
    CoverageRecord,
    DriftRecord,
)
from f51_darwin.transplant_16b.selection import (
    Provenance,
    tensor_digest,
    verify_inheritance,
)


# ── A fronteira ─────────────────────────────────────────────────────────────


def test_selection_preserves_bytes_exactly() -> None:
    """Selecao realoca escalares; nenhum valor novo aparece."""
    donor = torch.randn(8, 5, dtype=torch.float32)
    provenance = Provenance("select", "row", 8, (0, 2, 3, 6))
    target = provenance.apply(donor, dim=0)

    certificate = verify_inheritance(donor, target, provenance, dim=0)
    assert certificate.valid

    # Cada escalar do alvo existe no doador, no endereco que `kept` afirma.
    for target_index in range(target.shape[0]):
        source_index = provenance.source_of(target_index)
        assert torch.equal(target[target_index], donor[source_index])


def test_permutation_preserves_bytes_under_reordering() -> None:
    donor = torch.randn(6, 4, dtype=torch.float32)
    provenance = Provenance("permute", "row", 6, (5, 0, 3, 1, 4, 2))
    target = provenance.apply(donor, dim=0)

    assert verify_inheritance(donor, target, provenance, dim=0).valid
    # O multiconjunto de valores sobrevive a permutacao, ainda que o digest
    # do tensor inteiro nao sobreviva -- e por isso que o objeto hasheado
    # tem de ser o mapa, nao o tensor.
    assert torch.equal(donor.flatten().sort().values,
                       target.flatten().sort().values)
    assert tensor_digest(donor) != tensor_digest(target)


def test_projection_manufactures_values_and_breaks_the_digest() -> None:
    """Fora do grupo, o valor nao existe em lugar nenhum do doador."""
    donor = torch.randn(8, 5, dtype=torch.float32)
    provenance = Provenance("select", "row", 8, (0, 2, 3, 6))
    inherited = provenance.apply(donor, dim=0)

    # Projecao densa: media de duas linhas. O resultado nao e nenhuma linha.
    projected = inherited.clone()
    projected[0] = (donor[0] + donor[1]) / 2

    certificate = verify_inheritance(donor, projected, provenance, dim=0)
    assert not certificate.valid, "media de linhas nao pode certificar heranca"

    donor_rows = {tensor_digest(row) for row in donor}
    assert tensor_digest(projected[0]) not in donor_rows


def test_certificate_refused_outside_the_permutation_group() -> None:
    with pytest.raises(ValueError, match="fora do grupo"):
        Provenance("project", "row", 4, (0, 1))


# ── SHA nao mede ────────────────────────────────────────────────────────────


def test_digest_is_discontinuous_so_it_cannot_measure_distance() -> None:
    """Uma mudanca minuscula e uma enorme produzem digests igualmente distintos.

    E a razao de o registro precisar de um par metrico: o hash sozinho nao
    distingue um escalar reescrito de uma camada reinicializada.
    """
    base = torch.zeros(4, 4, dtype=torch.float32)

    tiny = base.clone()
    tiny[0, 0] = 1e-6
    huge = torch.randn(4, 4, dtype=torch.float32) * 1000.0

    assert tensor_digest(tiny) != tensor_digest(base)
    assert tensor_digest(huge) != tensor_digest(base)
    # Nenhuma nocao de "mais longe" e recuperavel dos digests.
    assert tensor_digest(tiny) != tensor_digest(huge)

    # A metrica, essa sim, ordena.
    assert (tiny - base).norm() < (huge - base).norm()


# ── A regra de acoplamento ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method", ["select", "permute", "exact_copy", "projection", "svd_fit"]
)
def test_weight_record_needs_evidence_regardless_of_its_name(method: str) -> None:
    """Nome de metodo nao e evidencia, para nenhum nome."""
    with pytest.raises(ValueError, match="neither a provenance_sha256"):
        CoverageRecord("donor.w", ("target.w",), method, "complete")


@pytest.mark.parametrize("method", ["select", "exact_copy", "projection"])
def test_naming_a_bijection_forbids_claiming_drift(method: str) -> None:
    provenance = Provenance("select", "row", 4, (0, 1))
    with pytest.raises(ValueError, match="drift must be\n?\\s*exactly 0.0"):
        CoverageRecord(
            "donor.w", ("target.w",), method, "complete", {},
            provenance_sha256=provenance.identity(),
            drift=DriftRecord("kl", 0.267),
        )


def test_a_flattering_method_name_earns_no_provenance_credit() -> None:
    """"exact_copy" sem certificado e registrado como fabricacao."""
    record = CoverageRecord(
        "donor.w", ("target.w",), "exact_copy", "complete", {},
        drift=DriftRecord("l2", 0.031),
    )
    assert not record.inherited


def test_manufacturing_method_must_measure_drift() -> None:
    with pytest.raises(ValueError, match="neither a provenance_sha256"):
        CoverageRecord("donor.w", ("target.w",), "projection", "complete")


def test_manufacturing_method_accepts_a_measured_record() -> None:
    record = CoverageRecord(
        "donor.w", ("target.w",), "projection", "complete", {},
        drift=DriftRecord("kl", 0.267, measured_on="held-out"),
    )
    assert not record.inherited
    assert record.drift is not None and record.drift.value == 0.267


def test_inherited_record_round_trips_through_the_chain(tmp_path: Path) -> None:
    """O certificado sobrevive a serializacao e a validacao da cadeia."""
    donor = torch.randn(8, 5, dtype=torch.float32)
    provenance = Provenance("select", "row", 8, (0, 2, 3, 6))
    target = provenance.apply(donor, dim=0)
    certificate = verify_inheritance(donor, target, provenance, dim=0)
    assert certificate.valid

    path = tmp_path / "coverage.jsonl"
    ledger = CoverageLedger.open(path, plan_id="a" * 64)
    ledger.append(
        CoverageRecord(
            "donor.w", ("target.w",), "select", "complete",
            {"retained_mass": 0.9608},
            provenance_sha256=certificate.provenance_sha256,
            drift=DriftRecord("l2", 0.0, measured_on="inherited"),
        )
    )

    reopened = CoverageLedger.open(path, plan_id="a" * 64)
    restored = reopened.records[0]
    assert restored.inherited
    assert restored.provenance_sha256 == provenance.identity()
    assert restored.drift is not None and restored.drift.value == 0.0


def test_tampering_with_drift_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "coverage.jsonl"
    ledger = CoverageLedger.open(path, plan_id="a" * 64)
    ledger.append(
        CoverageRecord(
            "donor.w", ("target.w",), "projection", "complete", {},
            drift=DriftRecord("kl", 0.267),
        )
    )
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"value":0.267', '"value":0.001'),
                    encoding="utf-8")

    with pytest.raises(ValueError, match="coverage ledger digest"):
        CoverageLedger.open(path, plan_id="a" * 64)


def test_provenance_identity_is_stable_and_unique_per_bijection() -> None:
    """O objeto hasheado e o plano de construcao, e ele e estavel."""
    first = Provenance("select", "residual_dim", 2048, (0, 5, 9))
    same = Provenance("select", "residual_dim", 2048, (0, 5, 9))
    other = Provenance("select", "residual_dim", 2048, (0, 5, 10))

    assert first.identity() == same.identity()
    assert first.identity() != other.identity()
