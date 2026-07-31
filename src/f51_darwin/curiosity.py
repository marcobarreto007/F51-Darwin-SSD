"""
F51 Curiosity Engine — O organismo EXPLORA, não só otimiza.

Princípio: O gradiente é covarde. Só desce. A curiosidade é corajosa.
Ela SOBE morros errados porque VALORIZA o desconhecido.

Reward total:  R = R_loss + α·R_novelty + β·R_curiosity + γ·R_jepa_surprise

Onde:
  R_loss:        loss caiu → recompensa (já existe)
  R_novelty:     descobriu padrão NOVO → recompensa exploratória
  R_curiosity:   reduziu incerteza em área desconhecida → recompensa epistêmica
  R_jepa_surprise: JEPA errou MUITO → tem algo interessante aqui → EXPLORA MAIS

Integrado em:
  - model.py: forward retorna hidden_states + jepa_error pro curiosity
  - training.py: loss inclui reward exploratório
  - soul.py: DopamineEngine ganha curiosity_drive
  - organism_247.py: ciclo inclui fase de exploração ativa

Uso:
  from f51_darwin.curiosity import CuriosityDrive
  curiosity = CuriosityDrive()
  reward = curiosity.evaluate(hidden_states, jepa_error, domain)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Any

import torch
import torch.nn.functional as F


_CURIOSITY_STATE_VERSION = 1
_OBSERVATION_PREFIX = "curiosity-observation-v1:"
_STATE_PREFIX = "curiosity-state-v1:"
_SURPRISE_CAP = 20.0


@dataclass(frozen=True, eq=False)
class CuriosityObservation:
    """Detached per-sample evidence produced without mutating memory."""

    sample_ids: tuple[str, ...]
    content_digests: tuple[str, ...]
    representations: torch.Tensor
    novelty: tuple[float, ...]
    surprise: tuple[float, ...]
    priority: tuple[float, ...]
    observation_id: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CuriosityObservation):
            return NotImplemented
        return (
            self.sample_ids == other.sample_ids
            and self.content_digests == other.content_digests
            and torch.equal(self.representations, other.representations)
            and self.novelty == other.novelty
            and self.surprise == other.surprise
            and self.priority == other.priority
            and self.observation_id == other.observation_id
        )


class CuriosityMemory:
    """Checkpointed sample-local novelty memory.

    ``observe`` is pure: it compares detached per-sample representations with
    the bank frozen at the previous successful commit. ``commit`` is the only
    mutation boundary and rejects stale, duplicated, or tampered observations.
    """

    def __init__(
        self,
        d_model: int,
        bank_capacity: int = 200,
        noisy_repeat_limit: int = 3,
    ) -> None:
        for name, value in (
            ("d_model", d_model),
            ("bank_capacity", bank_capacity),
            ("noisy_repeat_limit", noisy_repeat_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.d_model = d_model
        self.bank_capacity = bank_capacity
        self.noisy_repeat_limit = noisy_repeat_limit
        self._bank = torch.empty((0, d_model), dtype=torch.float32)
        self._content_counts: dict[str, int] = {}
        self._unimproved_repeats: dict[str, int] = {}
        self._committed_observation_ids: set[str] = set()
        self._cursor = 0

    @property
    def bank_size(self) -> int:
        return int(self._bank.shape[0])

    def observe(
        self,
        hidden: torch.Tensor,
        jepa_error: torch.Tensor | Sequence[float] | float,
        sample_ids: Sequence[str],
        *,
        content_digests: Sequence[str] | None = None,
    ) -> CuriosityObservation:
        representations = self._represent(hidden)
        batch_size = int(representations.shape[0])
        identifiers = _string_tuple(
            sample_ids,
            expected=batch_size,
            label="sample_ids",
            unique=True,
        )
        surprise = self._surprise_values(jepa_error, batch_size)
        digests = (
            _string_tuple(
                content_digests,
                expected=batch_size,
                label="content_digests",
                unique=False,
            )
            if content_digests is not None
            else tuple(_representation_digest(row) for row in representations)
        )

        if self.bank_size == 0:
            novelty = (1.0,) * batch_size
        else:
            similarities = representations @ self._bank.T
            novelty = tuple(
                float((1.0 - value).clamp(min=0.0, max=2.0).item())
                for value in similarities.max(dim=1).values
            )

        priorities = tuple(
            self._priority_for(
                novelty[index],
                surprise[index],
                digests[index],
            )
            for index in range(batch_size)
        )
        payload = {
            "memory_digest": self._state_digest(),
            "sample_ids": identifiers,
            "content_digests": digests,
            "representations": representations.tolist(),
            "novelty": novelty,
            "surprise": surprise,
            "priority": priorities,
        }
        observation_id = _OBSERVATION_PREFIX + sha256(
            _canonical(payload)
        ).hexdigest()
        return CuriosityObservation(
            sample_ids=identifiers,
            content_digests=digests,
            representations=representations.clone(),
            novelty=novelty,
            surprise=surprise,
            priority=priorities,
            observation_id=observation_id,
        )

    def commit(
        self,
        observation: CuriosityObservation,
        *,
        loss_improved: bool | Sequence[bool] = False,
    ) -> None:
        """Commit evidence after its source outcome completed successfully."""
        if not isinstance(observation, CuriosityObservation):
            raise TypeError("observation must be a CuriosityObservation")
        if observation.observation_id in self._committed_observation_ids:
            raise ValueError("duplicate curiosity observation")
        self._validate_observation(observation)

        expected = self._expected_observation_id(observation)
        if observation.observation_id != expected:
            raise ValueError("curiosity observation digest indicates tamper")
        improvements = _bool_tuple(
            loss_improved,
            expected=len(observation.sample_ids),
        )

        updated_bank = torch.cat(
            (self._bank, observation.representations.detach().clone()),
            dim=0,
        )
        self._bank = updated_bank[-self.bank_capacity :].contiguous()
        for digest, improved in zip(
            observation.content_digests,
            improvements,
            strict=True,
        ):
            self._content_counts[digest] = (
                self._content_counts.get(digest, 0) + 1
            )
            self._unimproved_repeats[digest] = (
                0
                if improved
                else self._unimproved_repeats.get(digest, 0) + 1
            )
        self._committed_observation_ids.add(observation.observation_id)
        self._cursor += len(observation.sample_ids)

    def state_dict(self) -> dict[str, object]:
        payload = self._state_payload()
        return payload | {"digest": self._state_digest(payload)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("CuriosityMemory state must be a mapping")
        expected_keys = {
            "version",
            "d_model",
            "bank_capacity",
            "noisy_repeat_limit",
            "surprise_cap",
            "bank",
            "content_counts",
            "unimproved_repeats",
            "committed_observation_ids",
            "cursor",
            "digest",
        }
        if set(state) != expected_keys:
            raise ValueError("CuriosityMemory state schema mismatch")
        payload = {key: copy_value for key, copy_value in state.items() if key != "digest"}
        expected_digest = self._state_digest(payload)
        if state.get("digest") != expected_digest:
            raise ValueError("CuriosityMemory state digest mismatch")
        if int(payload["version"]) != _CURIOSITY_STATE_VERSION:
            raise ValueError("unsupported CuriosityMemory state version")
        for name, expected in (
            ("d_model", self.d_model),
            ("bank_capacity", self.bank_capacity),
            ("noisy_repeat_limit", self.noisy_repeat_limit),
        ):
            if int(payload[name]) != expected:
                raise ValueError(f"CuriosityMemory {name} mismatch")
        if float(payload["surprise_cap"]) != _SURPRISE_CAP:
            raise ValueError("CuriosityMemory surprise_cap mismatch")

        bank = torch.tensor(payload["bank"], dtype=torch.float32)
        if bank.numel() == 0:
            bank = torch.empty((0, self.d_model), dtype=torch.float32)
        if bank.ndim != 2 or bank.shape[1] != self.d_model:
            raise ValueError("CuriosityMemory bank shape mismatch")
        if bank.shape[0] > self.bank_capacity:
            raise ValueError("CuriosityMemory bank exceeds capacity")
        if not torch.isfinite(bank).all():
            raise ValueError("CuriosityMemory bank must be finite")
        norms = bank.norm(dim=1)
        if bank.shape[0] and not torch.all(
            torch.isclose(norms, torch.ones_like(norms), atol=1e-5)
            | torch.isclose(norms, torch.zeros_like(norms), atol=1e-7)
        ):
            raise ValueError("CuriosityMemory bank must be normalized")

        counts = _count_mapping(payload["content_counts"], "content_counts")
        repeats = _count_mapping(
            payload["unimproved_repeats"],
            "unimproved_repeats",
        )
        if not set(repeats).issubset(counts):
            raise ValueError("CuriosityMemory repeat keys must be counted")
        committed = _string_tuple(
            payload["committed_observation_ids"],
            expected=None,
            label="committed_observation_ids",
            unique=True,
        )
        cursor = payload["cursor"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("CuriosityMemory cursor must be non-negative")

        self._bank = bank.clone().contiguous()
        self._content_counts = counts
        self._unimproved_repeats = repeats
        self._committed_observation_ids = set(committed)
        self._cursor = cursor

    def _represent(self, hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if hidden.ndim != 3 or hidden.shape[2] != self.d_model:
            raise ValueError(
                "hidden must have shape [batch, sequence, d_model]"
            )
        if hidden.shape[0] == 0 or hidden.shape[1] == 0:
            raise ValueError("hidden batch and sequence must be non-empty")
        detached = hidden.detach().float()
        if not torch.isfinite(detached).all():
            raise ValueError("hidden representations must be finite")
        represented = F.normalize(
            detached.mean(dim=1),
            p=2.0,
            dim=1,
            eps=1e-12,
        )
        return represented.to(device="cpu").contiguous().clone()

    def _surprise_values(
        self,
        jepa_error: torch.Tensor | Sequence[float] | float,
        batch_size: int,
    ) -> tuple[float, ...]:
        if isinstance(jepa_error, torch.Tensor):
            values = jepa_error.detach().float().to(device="cpu").reshape(-1)
        elif isinstance(jepa_error, bool):
            raise TypeError("jepa_error must be numeric")
        elif isinstance(jepa_error, (int, float)):
            values = torch.tensor([float(jepa_error)], dtype=torch.float32)
        else:
            try:
                values = torch.tensor(
                    list(jepa_error),
                    dtype=torch.float32,
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise TypeError("jepa_error must be numeric") from exc
        if values.numel() != batch_size:
            raise ValueError("jepa_error must contain one value per sample")
        if not torch.isfinite(values).all():
            raise ValueError("jepa_error must contain only finite values")
        values = values.clamp(min=0.0, max=_SURPRISE_CAP)
        return tuple(float(value) for value in values.tolist())

    def _priority_for(
        self,
        novelty: float,
        surprise: float,
        content_digest: str,
    ) -> float:
        base = min(
            2.0,
            max(0.5, 0.5 + 0.75 * novelty + 0.25 * math.tanh(surprise)),
        )
        repeats_after_limit = max(
            0,
            self._unimproved_repeats.get(content_digest, 0)
            - self.noisy_repeat_limit
            + 1,
        )
        factor = 0.5**repeats_after_limit
        return min(2.0, max(0.5, 0.5 + (base - 0.5) * factor))

    def _validate_observation(
        self,
        observation: CuriosityObservation,
    ) -> None:
        batch_size = len(observation.sample_ids)
        _string_tuple(
            observation.sample_ids,
            expected=batch_size,
            label="sample_ids",
            unique=True,
        )
        _string_tuple(
            observation.content_digests,
            expected=batch_size,
            label="content_digests",
            unique=False,
        )
        representation = observation.representations
        if (
            not isinstance(representation, torch.Tensor)
            or representation.device.type != "cpu"
            or representation.dtype != torch.float32
            or representation.requires_grad
            or representation.shape != (batch_size, self.d_model)
            or not torch.isfinite(representation).all()
        ):
            raise ValueError(
                "curiosity observation representations are invalid or non-finite"
            )
        for name, values in (
            ("novelty", observation.novelty),
            ("surprise", observation.surprise),
            ("priority", observation.priority),
        ):
            if len(values) != batch_size:
                raise ValueError(f"curiosity observation {name} length mismatch")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(
                    f"curiosity observation {name} must be finite"
                )
        if any(not 0.5 <= value <= 2.0 for value in observation.priority):
            raise ValueError("curiosity observation priority out of bounds")
        if (
            not isinstance(observation.observation_id, str)
            or not observation.observation_id.startswith(_OBSERVATION_PREFIX)
        ):
            raise ValueError("curiosity observation_id is invalid")

    def _expected_observation_id(
        self,
        observation: CuriosityObservation,
    ) -> str:
        payload = {
            "memory_digest": self._state_digest(),
            "sample_ids": observation.sample_ids,
            "content_digests": observation.content_digests,
            "representations": observation.representations.tolist(),
            "novelty": observation.novelty,
            "surprise": observation.surprise,
            "priority": observation.priority,
        }
        return _OBSERVATION_PREFIX + sha256(_canonical(payload)).hexdigest()

    def _state_payload(self) -> dict[str, object]:
        return {
            "version": _CURIOSITY_STATE_VERSION,
            "d_model": self.d_model,
            "bank_capacity": self.bank_capacity,
            "noisy_repeat_limit": self.noisy_repeat_limit,
            "surprise_cap": _SURPRISE_CAP,
            "bank": self._bank.tolist(),
            "content_counts": dict(sorted(self._content_counts.items())),
            "unimproved_repeats": dict(
                sorted(self._unimproved_repeats.items())
            ),
            "committed_observation_ids": sorted(
                self._committed_observation_ids
            ),
            "cursor": self._cursor,
        }

    def _state_digest(
        self,
        payload: Mapping[str, object] | None = None,
    ) -> str:
        body = self._state_payload() if payload is None else dict(payload)
        return _STATE_PREFIX + sha256(_canonical(body)).hexdigest()


def _representation_digest(representation: torch.Tensor) -> str:
    payload = (
        str(representation.dtype).encode("ascii")
        + str(tuple(representation.shape)).encode("ascii")
        + representation.contiguous().numpy().tobytes()
    )
    return "curiosity-content-v1:" + sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _string_tuple(
    values: Sequence[str] | object,
    *,
    expected: int | None,
    label: str,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(values)
    if expected is not None and len(result) != expected:
        raise ValueError(f"{label} must contain one value per sample")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain non-empty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def _bool_tuple(
    values: bool | Sequence[bool],
    *,
    expected: int,
) -> tuple[bool, ...]:
    if isinstance(values, bool):
        return (values,) * expected
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("loss_improved must be a bool or sequence of bools")
    result = tuple(values)
    if len(result) != expected or any(
        not isinstance(value, bool) for value in result
    ):
        raise ValueError("loss_improved must contain one bool per sample")
    return result


def _count_mapping(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"CuriosityMemory {label} must be a mapping")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"CuriosityMemory {label} keys must be strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"CuriosityMemory {label} counts must be non-negative integers"
            )
        result[key] = count
    return result


@dataclass
class CuriosityState:
    """Estado da curiosidade do organismo."""
    domains_explored: dict[str, int] = field(default_factory=dict)  # domínio → visitas
    novelty_scores: dict[str, float] = field(default_factory=dict)  # domínio → quão novo
    surprise_history: list[float] = field(default_factory=list)  # últimas surpresas
    exploration_budget: float = 0.3  # 30% do tempo = explorar
    total_steps: int = 0
    exploration_steps: int = 0
    discoveries: int = 0  # coisas genuinamente novas
    current_domain: str = "default"

    def to_dict(self) -> dict:
        return {
            'domains': dict(self.domains_explored),
            'novelty': {k: round(v, 4) for k, v in self.novelty_scores.items()},
            'avg_surprise': round(sum(self.surprise_history[-100:]) / max(1, len(self.surprise_history[-100:])), 4),
            'exploration_budget': self.exploration_budget,
            'exploration_ratio': round(self.exploration_steps / max(1, self.total_steps), 3),
            'discoveries': self.discoveries,
            'current_domain': self.current_domain,
        }


class CuriosityDrive:
    """Motor de curiosidade — o organismo ESCOLHE o que explorar.

    Inspirado em:
      - Schmidhuber (curiosity = prediction error reduction)
      - Pathak (intrinsic reward = model prediction error)
      - Stanley/Lehman (novelty search = reward novelty, not fitness)
      - Ecoffet (Go-Explore: first return, then explore)
    """

    def __init__(self, exploration_budget: float = 0.30):
        self.state = CuriosityState(exploration_budget=exploration_budget)
        self._domain_embeddings: dict[str, list[float]] = defaultdict(list)
        self._known_patterns: list[torch.Tensor] = []  # o que já foi visto
        self._pattern_threshold: float = 0.15  # similaridade abaixo disso = NOVO

    def choose_domain(self, available_domains: list[str]) -> str:
        """Escolhe QUAL domínio explorar.

        Estratégia Go-Explore:
          70% → explora domínio mais promissor (mais surpresa)
          30% → explora domínio aleatório (pra não viciar)
        """
        import random

        self.state.total_steps += 1

        if not available_domains:
            return "default"

        # Decide: explorar ou explorar mais fundo?
        if random.random() < self.state.exploration_budget:
            # EXPLORAÇÃO: visitar domínio pouco visitado ou de alta surpresa
            scores = {}
            for d in available_domains:
                visits = self.state.domains_explored.get(d, 0)
                novelty = self.state.novelty_scores.get(d, 0.5)
                # Domínios pouco visitados E de alta novelty têm prioridade
                scores[d] = novelty / (1.0 + visits * 0.01)

            if scores:
                self.state.exploration_steps += 1
                chosen = max(scores, key=scores.get)
                self.state.current_domain = chosen
                return chosen

        # EXPLOITAÇÃO: ficar no domínio atual ou no mais visitado
        if self.state.current_domain in available_domains:
            return self.state.current_domain
        return available_domains[0]

    def evaluate_novelty(
        self,
        hidden_states: torch.Tensor,
        domain: str = "default",
    ) -> float:
        """Avalia quão NOVO é um padrão nos hidden states.

        Compara com padrões já conhecidos via similaridade de cosseno.
        Quanto MENOS similar → MAIS novo → MAIS recompensa.
        """
        if hidden_states.numel() == 0:
            return 0.0

        # Representação compacta do padrão atual
        current = hidden_states.mean(dim=(0, 1)).detach()  # [D]
        current = current / (current.norm() + 1e-8)  # normaliza

        if not self._known_patterns:
            self._known_patterns.append(current)
            self.state.discoveries += 1
            return 1.0  # primeira coisa = totalmente novo!

        # Compara com padrões conhecidos
        max_similarity = 0.0
        for known in self._known_patterns[-50:]:  # últimos 50 padrões
            sim = torch.cosine_similarity(current.unsqueeze(0), known.unsqueeze(0)).item()
            max_similarity = max(max_similarity, sim)

        novelty = 1.0 - max_similarity  # 0 = igual, 1 = totalmente novo

        # Armazena se for suficientemente novo
        if novelty > self._pattern_threshold:
            self._known_patterns.append(current)
            if len(self._known_patterns) > 200:
                self._known_patterns = self._known_patterns[-100:]
            if novelty > 0.5:  # bem novo = descoberta
                self.state.discoveries += 1

        # Atualiza scores
        self.state.domains_explored[domain] = self.state.domains_explored.get(domain, 0) + 1
        old_novelty = self.state.novelty_scores.get(domain, 0.5)
        self.state.novelty_scores[domain] = 0.7 * old_novelty + 0.3 * novelty

        return novelty

    def evaluate_curiosity_reward(
        self,
        hidden_states: torch.Tensor,
        jepa_prediction_error: float = 0.0,
        domain: str = "default",
        loss_improved: bool = False,
    ) -> dict:
        """Calcula a recompensa total de curiosidade.

        Returns:
            dict com reward components e métricas
        """
        novelty = self.evaluate_novelty(hidden_states, domain)

        # JEPA surprise: errou MUITO na previsão = tem coisa interessante
        jepa_surprise = min(1.0, jepa_prediction_error / 10.0)

        # Curiosity reward: combina novelty + surpresa
        curiosity_reward = 0.6 * novelty + 0.4 * jepa_surprise

        self.state.surprise_history.append(jepa_surprise)
        if len(self.state.surprise_history) > 500:
            self.state.surprise_history = self.state.surprise_history[-200:]

        return {
            'novelty': round(novelty, 4),
            'jepa_surprise': round(jepa_surprise, 4),
            'curiosity_reward': round(curiosity_reward, 4),
            'domain': domain,
            'discoveries': self.state.discoveries,
            'exploration_ratio': round(
                self.state.exploration_steps / max(1, self.state.total_steps), 3
            ),
        }

    def should_explore(self) -> bool:
        """O organismo deve explorar AGORA?

        Baseado no Go-Explore: retorna a domínios promissores.
        Mas também injeta exploração aleatória.
        """
        import random
        return random.random() < self.state.exploration_budget

    def best_domain_to_explore(self) -> str:
        """Qual domínio tem maior potencial de descoberta?"""
        if not self.state.novelty_scores:
            return "default"
        return max(self.state.novelty_scores, key=self.state.novelty_scores.get)


# ═══════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════

def test_curiosity_basic():
    c = CuriosityDrive()
    d = c.choose_domain(["math", "code", "finance"])
    assert d in ["math", "code", "finance"]
    print("  choose_domain: OK")

def test_novelty_detection():
    c = CuriosityDrive()
    x1 = torch.randn(4, 64, 384)
    x2 = torch.randn(4, 64, 384)  # diferente
    n1 = c.evaluate_novelty(x1, "math")
    n2 = c.evaluate_novelty(x2, "math")
    assert n1 > 0.9, f"First pattern should be novel, got {n1}"
    print(f"  novelty: first={n1:.3f}, second={n2:.3f} — OK")

def test_curiosity_reward():
    c = CuriosityDrive()
    x = torch.randn(4, 64, 384)
    r = c.evaluate_curiosity_reward(x, jepa_prediction_error=3.5, domain="physics")
    assert 'novelty' in r
    assert 'curiosity_reward' in r
    print(f"  reward: novelty={r['novelty']}, curiosity={r['curiosity_reward']} — OK")

def test_exploration_budget():
    c = CuriosityDrive(exploration_budget=0.3)
    explores = sum(1 for _ in range(1000) if c.should_explore())
    ratio = explores / 1000
    assert 0.2 < ratio < 0.4, f"Expected ~30% exploration, got {ratio}"
    print(f"  exploration ratio: {ratio:.1%} — OK")
