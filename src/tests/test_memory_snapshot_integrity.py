"""Persistencia da memoria entre sessoes: o que sobrevive e o que avisa.

O organismo guarda chaves CODIFICADAS. Restaurar registros sem restaurar os
encoders nao falha -- devolve escore plausivel e errado. Medido antes destes
testes: recall 1.0/1.0/1.0 virou -0.007/0.101/0.050. E o snapshot ja carregava
um snapshot_sha256 que nada conferia; adulterar um value_embedding em +99
carregava em silencio.
"""

from __future__ import annotations

import copy

import pytest
import torch

from f51_darwin.cognition.memory import (
    PROVENANCE_VALUES,
    VERIFICATION_STATES,
    UniversalMemory,
)

D_MODEL = 128


def _memoria_ensinada(seed: int = 0, n: int = 3):
    torch.manual_seed(seed)
    mem = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    chaves = [torch.randn(1, 4, D_MODEL) for _ in range(n)]
    with torch.no_grad():
        for chave in chaves:
            mem.teach(
                key_hidden=chave,
                value_hidden=torch.randn(1, 4, D_MODEL),
                event_type="explicit_teaching",
                provenance="human",
            )
    return mem, chaves


def _escores(mem: UniversalMemory, chaves) -> list[float]:
    with torch.no_grad():
        return [mem.recall(c)[0].score for c in chaves]


# ── o caminho que funciona ──────────────────────────────────────────────────


def test_memoria_sobrevive_com_encoders_restaurados() -> None:
    mem, chaves = _memoria_ensinada()
    antes = _escores(mem, chaves)
    assert all(s > 0.99 for s in antes)

    snapshot = mem.memory_state_dict()
    pesos = copy.deepcopy(mem.state_dict())

    novo = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    novo.load_state_dict(pesos)
    novo.load_memory_state(snapshot)

    assert novo.slot_count == mem.slot_count
    depois = _escores(novo, chaves)
    for a, b in zip(antes, depois):
        assert abs(a - b) < 1e-4


# ── o caminho que silenciava ────────────────────────────────────────────────


def test_restaurar_sem_encoders_e_recusado() -> None:
    """Sem esta checagem o recall devolvia numero plausivel e errado."""
    mem, _ = _memoria_ensinada()
    snapshot = mem.memory_state_dict()

    outro = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    torch.manual_seed(999)
    for p in outro.parameters():
        with torch.no_grad():
            p.normal_()

    with pytest.raises(ValueError, match="different encoders"):
        outro.load_memory_state(snapshot)


def test_verify_false_permite_importar_deliberadamente() -> None:
    mem, _ = _memoria_ensinada()
    snapshot = mem.memory_state_dict()

    outro = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    torch.manual_seed(999)
    for p in outro.parameters():
        with torch.no_grad():
            p.normal_()

    outro.load_memory_state(snapshot, verify=False)
    assert outro.slot_count == mem.slot_count


# ── o hash que ninguem conferia ─────────────────────────────────────────────


def test_snapshot_adulterado_e_rejeitado() -> None:
    mem, _ = _memoria_ensinada()
    pesos = copy.deepcopy(mem.state_dict())
    snapshot = mem.memory_state_dict()
    snapshot["records"][0]["value_embedding"][0] += 99.0

    novo = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    novo.load_state_dict(pesos)
    with pytest.raises(ValueError, match="digest mismatch"):
        novo.load_memory_state(snapshot)


def test_snapshot_sem_digest_e_rejeitado() -> None:
    mem, _ = _memoria_ensinada()
    pesos = copy.deepcopy(mem.state_dict())
    snapshot = mem.memory_state_dict()
    snapshot.pop("snapshot_sha256")

    novo = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    novo.load_state_dict(pesos)
    with pytest.raises(ValueError, match="no snapshot_sha256"):
        novo.load_memory_state(snapshot)


# ── o fingerprint cobre os pesos inteiros ───────────────────────────────────


def test_fingerprint_muda_com_peso_fora_da_amostra_inicial() -> None:
    """Uma versao anterior hasheava so os 64 primeiros valores de cada tensor.

    Dois encoders diferindo apenas depois desse ponto colidiam, e o snapshot
    de um carregava no outro sem alarme.
    """
    mem, _ = _memoria_ensinada()
    antes = mem._encoder_fingerprint()

    with torch.no_grad():
        peso = mem.key_encoder[0].weight if isinstance(
            mem.key_encoder, torch.nn.Sequential) else mem.key_encoder.weight
        plano = peso.view(-1)
        assert plano.numel() > 64, "tensor curto demais para o teste valer"
        plano[-1] += 1.0

    assert mem._encoder_fingerprint() != antes


def test_snapshot_de_encoder_alterado_no_fim_e_recusado() -> None:
    mem, _ = _memoria_ensinada()
    snapshot = mem.memory_state_dict()

    with torch.no_grad():
        peso = mem.key_encoder[0].weight if isinstance(
            mem.key_encoder, torch.nn.Sequential) else mem.key_encoder.weight
        peso.view(-1)[-1] += 1.0

    with pytest.raises(ValueError, match="different encoders"):
        mem.load_memory_state(snapshot)


# ── taxonomia de proveniencia imposta ───────────────────────────────────────


def test_provenance_fora_da_taxonomia_e_recusada_na_escrita() -> None:
    mem = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    with pytest.raises(ValueError, match="provenance"):
        with torch.no_grad():
            mem.teach(
                key_hidden=torch.randn(1, 4, D_MODEL),
                value_hidden=torch.randn(1, 4, D_MODEL),
                event_type="explicit_teaching",
                provenance="inventado",
            )


@pytest.mark.parametrize("valor", sorted(PROVENANCE_VALUES))
def test_taxonomia_completa_e_aceita(valor: str) -> None:
    """Os tres valores documentados tem de funcionar, model_verified inclusive."""
    mem = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    with torch.no_grad():
        mem.teach(
            key_hidden=torch.randn(1, 4, D_MODEL),
            value_hidden=torch.randn(1, 4, D_MODEL),
            event_type="explicit_teaching",
            provenance=valor,
        )
    assert mem.slot_count == 1


def test_snapshot_com_provenance_invalida_e_recusado() -> None:
    """Carregar reconstroi registros fora do funil de escrita."""
    mem, _ = _memoria_ensinada()
    pesos = copy.deepcopy(mem.state_dict())
    snapshot = mem.memory_state_dict()
    snapshot["records"][0]["provenance"] = "injetado"
    snapshot["snapshot_sha256"] = None
    snapshot.pop("snapshot_sha256")

    novo = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
    novo.load_state_dict(pesos)
    with pytest.raises(ValueError):
        novo.load_memory_state(snapshot)


# ── rollback do encoder verificado por hash ─────────────────────────────────


def test_encoder_snapshot_restaura_e_verifica() -> None:
    mem, chaves = _memoria_ensinada()
    antes = _escores(mem, chaves)
    guardado = mem.encoder_snapshot()

    with torch.no_grad():
        for p in mem.parameters():
            p.normal_()
    assert _escores(mem, chaves) != antes

    mem.restore_encoder(guardado)
    depois = _escores(mem, chaves)
    for a, b in zip(antes, depois):
        assert abs(a - b) < 1e-4


def test_restore_encoder_recusa_snapshot_incoerente() -> None:
    mem, _ = _memoria_ensinada()
    guardado = mem.encoder_snapshot()
    guardado["fingerprint"] = "f" * 64

    with pytest.raises(ValueError, match="did not reproduce"):
        mem.restore_encoder(guardado)


def test_restore_encoder_recusa_snapshot_vazio() -> None:
    mem, _ = _memoria_ensinada()
    with pytest.raises(ValueError, match="no weights"):
        mem.restore_encoder({"fingerprint": "0" * 64, "weights": {}})


def test_digest_cobre_todo_o_conteudo() -> None:
    """Mudar qualquer campo de qualquer registro tem de quebrar o digest."""
    mem, _ = _memoria_ensinada()
    pesos = copy.deepcopy(mem.state_dict())
    base = mem.memory_state_dict()

    # Cada valor tem de DIFERIR do que teach() gravou: provenance="human",
    # verification_state="verified", event_type="explicit_teaching",
    # access_count=0. Trocar um campo por ele mesmo nao muda o digest, e nao
    # deve mesmo -- foi assim que a primeira versao deste teste falhou.
    for campo, valor in (
        ("provenance", "model_quarantine"),
        ("verification_state", "unverified"),
        ("event_type", "outro"),
        ("access_count", 4242),
    ):
        adulterado = copy.deepcopy(base)
        adulterado["records"][0][campo] = valor
        novo = UniversalMemory(d_model=D_MODEL, max_scale=1.5)
        novo.load_state_dict(pesos)
        with pytest.raises(ValueError, match="digest mismatch"):
            novo.load_memory_state(adulterado)
