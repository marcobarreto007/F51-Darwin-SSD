"""Datasets determinísticos de habilidade para descoberta e confirmação causal.

Uma habilidade só serve de espécime para o microscópio de circuitos se
render uma métrica **pareada por item**: cada item precisa de um número
próprio, para que `ABLATE - CLEAN` seja um efeito emparelhado e os portões
estatísticos de `build_causal_verdict` tenham o que testar.

Os splits de descoberta e confirmação precisam ser **disjuntos**:
`run_paired_ablation` recusa evidência cujo digest de item reapareça no
split de descoberta. Aqui isso é estrutural — o gerador é semeado pela
identidade do contrato, que inclui o nome do split, então dois splits nunca
percorrem o mesmo fluxo de aleatoriedade.

Nenhum construtor vem do artefato: builders e pools são escolhidos de um
registro local, como o design exige.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from f51_darwin.circuits.identity import canonical_sha256


class SkillContractError(ValueError):
    """O contrato de habilidade viola seu schema ou pede algo não registrado."""


# ─────────────────────────── pools de token ───────────────────────────


def _pool_gpt2_alpha_space_v1(vocab: Mapping[str, int]) -> tuple[int, ...]:
    """Tokens alfabéticos com espaço à esquerda, comprimento >= 4.

    Restringir a palavras comuns evita que o efeito medido venha de tokens
    de byte raros, cuja NLL é alta por frequência e não por habilidade.
    """
    return tuple(
        sorted(
            index
            for token, index in vocab.items()
            if token.startswith("Ġ")
            and token[1:].isalpha()
            and len(token) >= 4
        )
    )


_POOLS = {"gpt2_alpha_space_v1": _pool_gpt2_alpha_space_v1}


# ─────────────────────────────── contrato ───────────────────────────────


@dataclass(frozen=True)
class SkillContract:
    """Descrição imutável e hashável de um split de habilidade."""

    schema_version: int
    skill_id: str
    builder: str
    split: str
    domain: str
    seed: int
    items: int
    batch_size: int
    copy_length: int
    token_pool: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "skill_id",
            "builder",
            "split",
            "domain",
            "seed",
            "items",
            "batch_size",
            "copy_length",
            "token_pool",
        }
    )
    _SPLITS: ClassVar[frozenset[str]] = frozenset({"discovery", "confirmation"})

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise SkillContractError("schema_version must be 1")
        for name in ("skill_id", "builder", "domain", "token_pool"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SkillContractError(f"{name} must be a non-empty string")
        if self.split not in self._SPLITS:
            raise SkillContractError(f"split must be one of {sorted(self._SPLITS)}")
        for name in ("seed", "items", "batch_size", "copy_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SkillContractError(f"{name} must be an integer >= 1")
        if self.copy_length < 2:
            raise SkillContractError("copy_length must be >= 2 to score any position")
        if self.builder not in _BUILDERS:
            raise SkillContractError(f"unregistered builder: {self.builder!r}")
        if self.token_pool not in _POOLS:
            raise SkillContractError(f"unregistered token pool: {self.token_pool!r}")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(self._FIELDS)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillContract:
        if not isinstance(payload, Mapping):
            raise SkillContractError("skill contract must be a mapping")
        observed = set(payload)
        unknown = observed - cls._FIELDS
        missing = cls._FIELDS - observed
        if unknown:
            raise SkillContractError(f"unknown fields: {sorted(unknown)}")
        if missing:
            raise SkillContractError(f"missing fields: {sorted(missing)}")
        return cls(**{name: payload[name] for name in cls._FIELDS})

    @property
    def identity(self) -> str:
        return f"f51-skill-v1:{canonical_sha256(self.to_dict())}"

    def stream_seed(self) -> int:
        """Semente derivada da identidade — splits distintos nunca coincidem."""
        digest = hashlib.sha256(self.identity.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)


# ─────────────────────────────── builders ───────────────────────────────


def _build_induction_repeat_v1(
    contract: SkillContract, pool: Sequence[int]
) -> list[dict[str, Any]]:
    """Cópia em contexto: a sequência ``[S S]``.

    A segunda cópia só é previsível copiando do próprio contexto, porque
    ``S`` é amostrado uniformemente do pool e não carrega estatística de
    linguagem que ajude.

    São pontuadas as posições absolutas ``L+1 .. 2L-1``. A posição ``L`` fica
    de fora de propósito: o token anterior a ela é ``S[L-1]``, que na primeira
    cópia não foi seguido de nada — não há o que copiar, e incluí-la
    contaminaria o efeito com um item impossível.
    """
    length = contract.copy_length
    pool_tensor = torch.tensor(pool, dtype=torch.long)
    generator = torch.Generator().manual_seed(contract.stream_seed())

    batches: list[dict[str, Any]] = []
    remaining = contract.items
    while remaining > 0:
        size = min(contract.batch_size, remaining)
        core = pool_tensor[
            torch.randint(len(pool), (size, length), generator=generator)
        ]
        input_ids = torch.cat([core, core], dim=1)

        # logits[:, t] prevê input_ids[:, t+1]; para pontuar a posição p
        # usa-se t = p-1, logo t percorre L .. 2L-2.
        labels = torch.full_like(input_ids, -100)
        labels[:, length : 2 * length - 1] = input_ids[:, length + 1 : 2 * length]

        batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "domains": contract.domain,
            }
        )
        remaining -= size
    return batches


def _build_induction_repeat_offset_v1(
    contract: SkillContract, pool: Sequence[int]
) -> list[dict[str, Any]]:
    """Cópia em contexto com a fronteira em posição variável por item.

    Motivação medida: com ``induction_repeat_v1`` todos os itens têm a mesma
    estrutura, então o arm SHUFFLE — que permuta ativações **entre itens** —
    preserva o perfil posicional e vira uma intervenção fraca por construção.
    Na primeira execução real ele reproduziu só 4,7% do efeito de ablação e
    reprovou o portão, apesar de restore=1.00 e razão alvo/controle de 49,8×.

    Aqui cada item recebe um prefixo aleatório de comprimento ``o`` sorteado
    em ``[0, L]``, então a fronteira da repetição cai numa posição diferente
    em cada item. Permutar entre itens passa a misturar fases distintas, e o
    controle volta a ser informativo.

    Formato: ``[prefixo(o), S, S, enchimento(L-o)]``, comprimento fixo ``3L``.
    """
    length = contract.copy_length
    total = 3 * length
    pool_tensor = torch.tensor(pool, dtype=torch.long)
    generator = torch.Generator().manual_seed(contract.stream_seed())

    batches: list[dict[str, Any]] = []
    remaining = contract.items
    while remaining > 0:
        size = min(contract.batch_size, remaining)
        input_ids = pool_tensor[
            torch.randint(len(pool), (size, total), generator=generator)
        ]
        labels = torch.full_like(input_ids, -100)
        offsets = torch.randint(0, length + 1, (size,), generator=generator)
        for row in range(size):
            offset = int(offsets[row])
            core = input_ids[row, offset : offset + length].clone()
            input_ids[row, offset + length : offset + 2 * length] = core
            # Pontua as posições o+L+1 .. o+2L-1, via t = p-1.
            start = offset + length
            stop = offset + 2 * length - 1
            labels[row, start:stop] = input_ids[row, start + 1 : stop + 1]

        batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "domains": contract.domain,
            }
        )
        remaining -= size
    return batches


_BUILDERS = {
    "induction_repeat_v1": _build_induction_repeat_v1,
    "induction_repeat_offset_v1": _build_induction_repeat_offset_v1,
}


# ──────────────────────────────── fachada ────────────────────────────────


def build_skill_batches(
    contract: SkillContract, vocab: Mapping[str, int]
) -> list[dict[str, Any]]:
    """Materializa os batches do contrato no formato de `run_paired_ablation`."""
    if not isinstance(contract, SkillContract):
        raise SkillContractError("contract must be a validated SkillContract")
    if not isinstance(vocab, Mapping) or not vocab:
        raise SkillContractError("vocab must be a non-empty mapping")
    pool = _POOLS[contract.token_pool](vocab)
    if len(pool) < 2:
        raise SkillContractError(f"token pool {contract.token_pool!r} is too small")
    batches = _BUILDERS[contract.builder](contract, pool)
    if not batches:
        raise SkillContractError("builder produced no batches")
    return batches


def token_pool_identity(token_pool: str, vocab: Mapping[str, int]) -> str:
    """Identidade do pool — entra na proveniência do experimento."""
    if token_pool not in _POOLS:
        raise SkillContractError(f"unregistered token pool: {token_pool!r}")
    pool = _POOLS[token_pool](vocab)
    return canonical_sha256({"token_pool": token_pool, "indices": list(pool)})


def registered_builders() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def registered_token_pools() -> tuple[str, ...]:
    return tuple(sorted(_POOLS))


__all__ = [
    "SkillContract",
    "SkillContractError",
    "build_skill_batches",
    "registered_builders",
    "registered_token_pools",
    "token_pool_identity",
]
