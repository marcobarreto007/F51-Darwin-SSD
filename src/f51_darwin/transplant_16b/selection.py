"""Reducao por selecao, dentro do grupo de simetria de permutacao.

Toda operacao aqui preserva uma bijecao entre unidade do doador e unidade
do alvo. Isso nao e uma preferencia de estilo: as unidades ocultas de um
bloco cuja nao-linearidade e aplicada coordenada-a-coordenada admitem
simetria de PERMUTACAO, nao ortogonal. Dentro desse grupo a funcao e
invariante e o mapa e bijetivo; fora dele perdem-se os dois ao mesmo
tempo -- projecao densa, media de camadas e fit por gradiente falham nos
dois criterios simultaneamente.

Medido no SmolLM2-1.7B (2026-07-29):
  - selecao 2048 -> 1920 dimensoes retem 96.08% da massa de contribuicao;
    so 13 dimensoes tem contribuicao > 10x a mediana (massive activations).
  - descarte de camadas por Block Influence: 24->22 custa KL 0.267,
    24->20 custa 0.739, 24->16 custa 4.354 (top-1 cai a 26.1%).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch.nn import functional as F


_ALLOWED_KINDS = frozenset({"select", "permute", "scatter", "drop"})


@dataclass(frozen=True)
class Provenance:
    """Bijecao doador -> alvo ao longo de um eixo.

    ``kept[i]`` e o indice no doador que ocupa a posicao ``i`` no alvo, e
    e a unica coisa necessaria para responder "de onde veio este
    parametro" sem estimativa.
    """

    kind: str
    axis: str
    source_size: int
    kept: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"operacao {self.kind!r} esta fora do grupo de simetria de "
                f"permutacao; permitidas: {sorted(_ALLOWED_KINDS)}"
            )
        if not self.axis:
            raise ValueError("axis must be a non-empty string")
        if self.source_size < 1:
            raise ValueError("source_size must be positive")
        if len(set(self.kept)) != len(self.kept):
            raise ValueError("kept indices must be unique (bijection)")
        if any(not 0 <= index < self.source_size for index in self.kept):
            raise ValueError("kept indices must lie inside the source axis")

    @property
    def target_size(self) -> int:
        return len(self.kept)

    @property
    def dropped(self) -> tuple[int, ...]:
        return tuple(sorted(set(range(self.source_size)) - set(self.kept)))

    def source_of(self, target_index: int) -> int:
        """Indice no doador que originou ``target_index``."""
        return self.kept[target_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "axis": self.axis,
            "source_size": self.source_size,
            "target_size": self.target_size,
            "kept": list(self.kept),
            "dropped": list(self.dropped),
        }

    def identity(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def apply(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Fatia ``tensor`` ao longo de ``dim`` preservando a ordem do alvo."""
        if tensor.shape[dim] != self.source_size:
            raise ValueError(
                f"tensor tem {tensor.shape[dim]} no eixo {dim}, "
                f"provenance espera {self.source_size}"
            )
        index = torch.tensor(self.kept, dtype=torch.long, device=tensor.device)
        return tensor.index_select(dim, index)


def tensor_digest(tensor: torch.Tensor) -> str:
    """SHA-256 dos bytes de ``tensor``, com dtype e shape."""
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(
        memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class InheritanceCertificate:
    """Prova de que ``target`` e byte-identico ao gather de ``donor``.

    Este e o trabalho para o qual SHA-256 serve, e o unico. Ele nao mede
    distancia -- e descontinuo, entao nao tem estrutura metrica -- mas
    responde exatamente a pergunta que o criterio de proveniencia faz:
    este parametro foi HERDADO do doador ou FABRICADO?

    A resposta so e decidivel porque as operacoes permitidas pelo grupo de
    simetria de permutacao sao precisamente as que preservam o valor
    escalar. Fora do grupo o valor passa a nao existir em lugar nenhum do
    doador, o digest quebra, e quebra corretamente.
    """

    provenance_sha256: str
    donor_sha256: str
    expected_sha256: str
    actual_sha256: str
    axis: str
    dim: int

    @property
    def valid(self) -> bool:
        return self.expected_sha256 == self.actual_sha256


def verify_inheritance(
    donor: torch.Tensor,
    target: torch.Tensor,
    provenance: Provenance,
    dim: int,
) -> InheritanceCertificate:
    """Certifica que ``target`` saiu de ``donor`` por ``provenance``, sem estimativa.

    Recomputa o gather e compara digests. Um certificado valido afirma algo
    forte e verificavel por terceiros: cada escalar do alvo tem um endereco
    no doador, dado por ``provenance.kept``, e os bytes batem.
    """
    if provenance.kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"operacao {provenance.kind!r} nao preserva bytes; nao existe "
            f"certificado de heranca fora do grupo de permutacao"
        )
    expected = provenance.apply(donor, dim)
    if expected.shape != target.shape:
        raise ValueError(
            f"gather produz {tuple(expected.shape)}, alvo tem "
            f"{tuple(target.shape)}"
        )
    return InheritanceCertificate(
        provenance_sha256=provenance.identity(),
        donor_sha256=tensor_digest(donor),
        expected_sha256=tensor_digest(expected),
        actual_sha256=tensor_digest(target),
        axis=provenance.axis,
        dim=dim,
    )


def block_influence(
    hidden_states: list[torch.Tensor],
    mask: torch.Tensor | None = None,
) -> tuple[float, ...]:
    """Block Influence por camada: ``1 - cos(entrada, saida)``.

    Metrica do ShortGPT. Valor baixo significa que a camada quase nao move
    o residual, isto e, e redundante e pode ser descartada inteira --
    descartar preserva a bijecao das que ficam, ao contrario de fundir.
    """
    if len(hidden_states) < 2:
        raise ValueError("block_influence requires at least two hidden states")
    scores = []
    for index in range(len(hidden_states) - 1):
        before = hidden_states[index].reshape(-1, hidden_states[index].shape[-1])
        after = hidden_states[index + 1].reshape(-1, hidden_states[index + 1].shape[-1])
        if mask is not None:
            selector = mask.reshape(-1).bool()
            before, after = before[selector], after[selector]
        similarity = F.cosine_similarity(before.float(), after.float(), dim=-1)
        scores.append(float((1.0 - similarity).mean()))
    return tuple(scores)


def select_layers(
    influence: tuple[float, ...],
    target_layers: int,
    *,
    protected: tuple[int, ...] = (),
) -> Provenance:
    """Mantem as ``target_layers`` camadas de maior Block Influence.

    ``protected`` nunca e descartada. No SmolLM2 a ultima camada mede
    influencia 0.727 contra ~0.06 do miolo -- uma ordem de grandeza -- e
    perde-la destroi o modelo, entao ela entra protegida por padrao no
    call site.
    """
    total = len(influence)
    if not 1 <= target_layers <= total:
        raise ValueError("target_layers must lie in [1, len(influence)]")
    for index in protected:
        if not 0 <= index < total:
            raise ValueError("protected index outside the layer range")
    if len(set(protected)) > target_layers:
        raise ValueError("protected layers exceed the target depth")

    forced = set(protected)
    candidates = sorted(
        (index for index in range(total) if index not in forced),
        key=lambda index: (-influence[index], index),
    )
    kept = forced | set(candidates[: target_layers - len(forced)])
    return Provenance(
        kind="drop",
        axis="layer",
        source_size=total,
        kept=tuple(sorted(kept)),
    )


def rank_residual_dims(
    activations: torch.Tensor,
    readout: torch.Tensor,
) -> torch.Tensor:
    """Contribuicao por dimensao do residual para o logit.

    ``activations`` sao estados do residual ``[tokens, hidden]`` e
    ``readout`` e a matriz que os le (embedding amarrado ao lm_head). A
    contribuicao da dimensao ``d`` para o logit e proporcional a
    ``|h[d]| * |E[:, d]|``, entao o produto das magnitudes medias e o
    escore natural -- e ele preserva as massive activations, que uma SVD
    centrada descartaria por terem variancia baixa apesar de magnitude
    enorme.
    """
    if activations.ndim != 2 or readout.ndim != 2:
        raise ValueError("activations and readout must be rank-2")
    if activations.shape[-1] != readout.shape[-1]:
        raise ValueError("activation and readout hidden dimensions differ")
    return (
        activations.detach().float().abs().mean(0)
        * readout.detach().float().abs().mean(0)
    )


def select_residual_dims(scores: torch.Tensor, target_dim: int) -> Provenance:
    """Mantem as ``target_dim`` dimensoes de maior contribuicao, em ordem."""
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    source = int(scores.shape[0])
    if not 1 <= target_dim <= source:
        raise ValueError("target_dim must lie in [1, len(scores)]")
    ranked = torch.topk(scores.float(), k=target_dim, largest=True).indices
    return Provenance(
        kind="select",
        axis="residual_dim",
        source_size=source,
        kept=tuple(sorted(int(index) for index in ranked.tolist())),
    )


def retained_mass(scores: torch.Tensor, provenance: Provenance) -> float:
    """Fracao da massa de contribuicao que sobrevive a selecao."""
    values = scores.detach().float()
    total = values.sum()
    if total <= 0:
        raise ValueError("contribution scores must sum to a positive value")
    index = torch.tensor(provenance.kept, dtype=torch.long, device=values.device)
    return float(values[index].sum() / total)


def solve_diagonal_repair(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    clamp: tuple[float, float] = (0.1, 10.0),
) -> torch.Tensor:
    """Ganho diagonal por canal que aproxima ``current`` de ``target``.

    Reparo LEGITIMO sob a regra de simetria: uma escala por canal e
    diagonal, entao nao mistura unidades e cada peso continua apontando
    para a mesma origem. Nao e destilacao nem fit de tensor denso.

    Minimo quadrado fechado por canal:
        ``d_c = sum_t current[t,c] * target[t,c] / sum_t current[t,c]^2``
    """
    if current.shape != target.shape or current.ndim != 2:
        raise ValueError("current and target must be matching rank-2 tensors")
    numerator = (current.float() * target.float()).sum(0)
    denominator = current.float().square().sum(0)
    gain = numerator / denominator.clamp_min(1e-12)
    gain = torch.where(denominator > 1e-12, gain, torch.ones_like(gain))
    return gain.clamp(*clamp)


__all__ = [
    "InheritanceCertificate",
    "Provenance",
    "block_influence",
    "tensor_digest",
    "verify_inheritance",
    "rank_residual_dims",
    "retained_mass",
    "select_layers",
    "select_residual_dims",
    "solve_diagonal_repair",
]
