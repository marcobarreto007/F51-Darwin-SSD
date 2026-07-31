"""
F51 GABA INHIBITION — Equilíbrio Excitação/Inibição
====================================================

Biologia: O cérebro mantém um equilíbrio delicado entre excitação (Glutamato)
e inibição (GABA). Sem inibição, redes neurais entram em estado epiléptico
(instável, overdosificado).

Arquitetura:
- GABAergicLayer: camada de inibição explícita
- EIBalanceMonitor: monitora E/I ratio
- InhibitoryGating: gate baseado em inibição
- HomeostaticControl: mantém equilíbrio

Paper inspiração:
- "Inhibition-stabilized networks" (Tsodyks et al)
- "E/I balance in neural networks" (Hennequin)
"""

from __future__ import annotations

from collections import deque
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GABAConfig:
    """Configuração do sistema de inibição GABA."""
    d_model: int = 2048
    inhibition_ratio: float = 0.2  # 20% inibição, 80% excitação
    gaba_tau: float = 100.0  # constante de tempo decaimento
    target_ei_ratio: float = 0.8  # E/I alvo (excitação/inibição)
    adaptive_inhibition: bool = True
    gaba_receptor_density: float = 1.0  # densidade de receptores GABA
    max_delta_ratio: float = 0.25
    gate_limit: float = 1.0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if not math.isfinite(self.max_delta_ratio) or self.max_delta_ratio < 0.0:
            raise ValueError("max_delta_ratio must be finite and >= 0")
        if not math.isfinite(self.gate_limit) or self.gate_limit < 0.0:
            raise ValueError("gate_limit must be finite and >= 0")
        if not math.isfinite(self.gaba_tau) or self.gaba_tau <= 0.0:
            raise ValueError("gaba_tau must be finite and positive")


@dataclass(frozen=True)
class _ValidatedGABAObservation:
    excitation_mean: torch.Tensor
    inhibition_mean: torch.Tensor
    observation_id: str


@dataclass(frozen=True)
class _DeferredGABAObservationID:
    sample_excitation: torch.Tensor
    sample_inhibition: torch.Tensor
    excitation_version: int
    inhibition_version: int


@dataclass(frozen=True)
class _GABACommitSnapshot:
    gaba_level: torch.Tensor
    excitation_ema: torch.Tensor
    inhibition_ema: torch.Tensor
    state_update_count: torch.Tensor
    observation_ids: frozenset[str]
    observation_order: tuple[str, ...]


class GABAergicLayer(nn.Module):
    """Camada de inibição GABAergic.

    Fornece inibição explícita e competitiva, complementando a excitação
    das camadas convencionais. Isso estabiliza a dinâmica neural.
    """

    def __init__(self, config: GABAConfig):
        super().__init__()
        self.config = config

        # Pesos inibitórios (GABA)
        self.inhibitory_weight = nn.Parameter(
            torch.randn(config.d_model, config.d_model) * 0.02
        )

        # Bias inibitório
        self.inhibitory_bias = nn.Parameter(torch.zeros(config.d_model))

        # Nível de GABA (neuromodulador)
        self.register_buffer('gaba_level', torch.tensor(1.0))

        # Histórico de atividade para homeostase
        self.register_buffer('excitation_ema', torch.ones(config.d_model) * 0.1)
        self.register_buffer('inhibition_ema', torch.ones(config.d_model) * 0.1)

        # The residual is a learned no-op at initialization. State evidence is
        # committed separately only after the caller accepts an attempt.
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "state_update_count", torch.zeros((), dtype=torch.long)
        )
        self._committed_observation_ids: set[str] = set()
        self._committed_observation_order: deque[str] = deque(maxlen=128)

    @torch.no_grad()
    def update_gaba_level(self, excitation: torch.Tensor, inhibition: torch.Tensor) -> None:
        """Atualiza nível de GABA baseado no balanço E/I."""
        e_mean = excitation.abs().mean()
        i_mean = inhibition.abs().mean()

        # E/I ratio
        ei_ratio = e_mean / (i_mean + 1e-8)

        # Se E/I muito alto, aumenta GABA; se muito baixo, diminui
        target = self.config.target_ei_ratio
        error = ei_ratio - target

        # Ajusta GABA level
        updated = (
            self.gaba_level * (1 - 1 / self.config.gaba_tau)
            + 0.1 * error
        )
        self.gaba_level.copy_(updated.clamp(0.1, 5.0))

    @staticmethod
    def _observation_digest(
        sample_excitation: torch.Tensor,
        sample_inhibition: torch.Tensor,
    ) -> str:
        digest = hashlib.sha256()
        for name, tensor in (
            ("sample_excitation", sample_excitation),
            ("sample_inhibition", sample_inhibition),
        ):
            value = (
                tensor.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _assert_finite_predicate(predicate: Any, *, name: str) -> None:
        message = f"{name} must be finite"
        if predicate.device.type == "cuda":
            torch._assert_async(predicate, message)
        elif not predicate:
            raise ValueError(message)

    @classmethod
    def _require_finite(cls, value: torch.Tensor, *, name: str) -> None:
        cls._assert_finite_predicate(
            torch.isfinite(value).all(),
            name=name,
        )

    def _require_finite_masking_state(self) -> None:
        finite_checks = torch.stack(
            tuple(
                torch.isfinite(value).all()
                for value in (
                    self.residual_gate,
                    self.gaba_level,
                )
            )
        )
        self._assert_finite_predicate(
            finite_checks.all(),
            name="GABA masking state",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mutate_state: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a bounded inhibitory residual and detached state evidence.

        Args:
            x: [batch, seq_len, d_model] input
            mutate_state: opt-in compatibility hook; causal callers leave this
                false and commit only after a successful attempt outcome.

        Returns:
            delta: [batch, seq_len, d_model] bounded residual delta
            observation: detached per-sample E/I evidence and stable digest
        """
        if x.ndim != 3 or x.size(-1) != self.config.d_model:
            raise ValueError(
                "x must have shape [batch, sequence, d_model]"
            )
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("GABA batch and sequence must be non-empty")
        self._require_finite_masking_state()

        inhibition = F.linear(
            x,
            self.inhibitory_weight,
            self.inhibitory_bias
        )
        inhibition = inhibition * torch.sigmoid(self.gaba_level)
        self._require_finite(inhibition, name="GABA inhibition")

        gate = torch.tanh(self.residual_gate) * self.config.gate_limit
        raw_delta = -gate * inhibition
        raw_norm = raw_delta.float().norm(dim=-1)
        max_norm = x.float().norm(dim=-1) * self.config.max_delta_ratio
        scale = torch.clamp(
            max_norm / raw_norm.clamp_min(torch.finfo(torch.float32).eps),
            max=1.0,
        )
        delta = raw_delta * scale.to(dtype=raw_delta.dtype).unsqueeze(-1)

        sample_excitation = x.detach().float().abs().mean(dim=(1, 2))
        sample_inhibition = (
            inhibition.detach().float().abs().mean(dim=(1, 2))
        )
        finite_outputs = torch.stack(
            (
                torch.isfinite(delta).all(),
                torch.isfinite(sample_excitation).all(),
                torch.isfinite(sample_inhibition).all(),
            )
        )
        self._assert_finite_predicate(
            finite_outputs.all(),
            name="GABA delta and observation",
        )
        # Inference tensors intentionally have no version counter. Build the
        # deferred digest material outside inference mode so validation can
        # still detect any mutation before commit.
        with torch.inference_mode(False):
            digest_excitation = sample_excitation.clone()
            digest_inhibition = sample_inhibition.clone()
        observation_id = _DeferredGABAObservationID(
            sample_excitation=digest_excitation,
            sample_inhibition=digest_inhibition,
            excitation_version=digest_excitation._version,
            inhibition_version=digest_inhibition._version,
        )
        observation: dict[str, Any] = {
            "sample_excitation": sample_excitation,
            "sample_inhibition": sample_inhibition,
            "observation_id": observation_id,
        }

        if mutate_state:
            self.commit_observation(observation)
        return delta, observation

    @torch.no_grad()
    def _prepare_observation(
        self,
        observation: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        raw_excitation = observation.get("sample_excitation")
        raw_inhibition = observation.get("sample_inhibition")
        observation_id = observation.get("observation_id")
        if (
            not isinstance(raw_excitation, torch.Tensor)
            or not isinstance(raw_inhibition, torch.Tensor)
        ):
            raise ValueError("observation requires per-sample tensors")
        if (
            raw_excitation.dtype != torch.float32
            or raw_inhibition.dtype != torch.float32
        ):
            raise ValueError("observation tensors must use float32 dtype")
        sample_excitation = raw_excitation.detach().clone()
        sample_inhibition = raw_inhibition.detach().clone()
        if (
            sample_excitation.ndim != 1
            or sample_inhibition.ndim != 1
            or sample_excitation.shape != sample_inhibition.shape
            or sample_excitation.numel() == 0
        ):
            # Preserve fail-closed error precedence on malformed evidence
            # without adding per-layer syncs to the valid batched path.
            self._require_finite(
                sample_excitation,
                name="observation excitation",
            )
            self._require_finite(
                sample_inhibition,
                name="observation inhibition",
            )
            raise ValueError(
                "observation requires matching per-sample tensors"
            )
        return sample_excitation, sample_inhibition, observation_id

    @classmethod
    @torch.no_grad()
    def validate_observations_batch(
        cls,
        layer_observations: tuple[
            tuple["GABAergicLayer", dict[str, Any]],
            ...,
        ],
    ) -> tuple[_ValidatedGABAObservation, ...]:
        """Finalize ordered observation IDs with one CPU transfer per device."""
        records: list[
            tuple[
                int,
                GABAergicLayer,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                str | None,
            ]
        ] = []
        for index, (layer, observation) in enumerate(layer_observations):
            excitation, inhibition, observation_id = layer._prepare_observation(
                observation
            )
            expected_id: str | None
            if isinstance(observation_id, _DeferredGABAObservationID):
                reference_excitation = observation_id.sample_excitation
                reference_inhibition = observation_id.sample_inhibition
                expected_id = None
                if (
                    reference_excitation._version
                    != observation_id.excitation_version
                    or reference_inhibition._version
                    != observation_id.inhibition_version
                    or reference_excitation.dtype != torch.float32
                    or reference_inhibition.dtype != torch.float32
                    or reference_excitation.shape != excitation.shape
                    or reference_inhibition.shape != inhibition.shape
                    or reference_excitation.device != excitation.device
                    or reference_inhibition.device != inhibition.device
                ):
                    raise ValueError("observation digest material is invalid")
            elif isinstance(observation_id, str):
                reference_excitation = excitation
                reference_inhibition = inhibition
                expected_id = observation_id
            else:
                raise ValueError("observation digest is invalid")
            records.append(
                (
                    index,
                    layer,
                    excitation,
                    inhibition,
                    reference_excitation,
                    reference_inhibition,
                    expected_id,
                )
            )

        groups: dict[
            tuple[torch.device, tuple[int, ...]],
            list[
                tuple[
                    int,
                    GABAergicLayer,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    str | None,
                ]
            ],
        ] = {}
        for record in records:
            key = (record[2].device, tuple(record[2].shape))
            groups.setdefault(key, []).append(record)

        validated: list[_ValidatedGABAObservation | None] = [
            None
        ] * len(records)
        for grouped_records in groups.values():
            material = torch.stack(
                tuple(
                    torch.stack(
                        (
                            excitation,
                            inhibition,
                            reference_excitation,
                            reference_inhibition,
                        )
                    )
                    for (
                        _,
                        _,
                        excitation,
                        inhibition,
                        reference_excitation,
                        reference_inhibition,
                        _,
                    ) in grouped_records
                )
            ).to(device="cpu")
            for group_index, record in enumerate(grouped_records):
                index, layer, excitation, inhibition, _, _, expected_id = record
                cpu_material = material[group_index]
                if not torch.isfinite(cpu_material).all():
                    raise ValueError("observation tensors must be finite")
                if not (
                    torch.equal(cpu_material[0], cpu_material[2])
                    and torch.equal(cpu_material[1], cpu_material[3])
                ):
                    raise ValueError("observation digest is invalid")
                observation_id = cls._observation_digest(
                    cpu_material[0],
                    cpu_material[1],
                )
                if expected_id is not None and expected_id != observation_id:
                    raise ValueError("observation digest is invalid")
                if observation_id in layer._committed_observation_ids:
                    raise ValueError("observation was already committed")

                excitation_mean = cpu_material[0].mean()
                inhibition_mean = cpu_material[1].mean()
                if not (
                    torch.isfinite(excitation_mean)
                    and torch.isfinite(inhibition_mean)
                ):
                    raise ValueError("observation means must remain finite")
                validated[index] = _ValidatedGABAObservation(
                    excitation_mean=excitation_mean.to(
                        device=layer.excitation_ema.device,
                        dtype=layer.excitation_ema.dtype,
                    ),
                    inhibition_mean=inhibition_mean.to(
                        device=layer.inhibition_ema.device,
                        dtype=layer.inhibition_ema.dtype,
                    ),
                    observation_id=observation_id,
                )

        if any(observation is None for observation in validated):
            raise RuntimeError("GABA observation validation was incomplete")
        return tuple(
            observation
            for observation in validated
            if observation is not None
        )

    @torch.no_grad()
    def validate_observation(
        self,
        observation: dict[str, Any],
    ) -> _ValidatedGABAObservation:
        """Validate and finalize one observation without mutating state."""
        return self.validate_observations_batch(((self, observation),))[0]

    @torch.no_grad()
    def _snapshot_commit_state(self) -> _GABACommitSnapshot:
        return _GABACommitSnapshot(
            gaba_level=self.gaba_level.clone(),
            excitation_ema=self.excitation_ema.clone(),
            inhibition_ema=self.inhibition_ema.clone(),
            state_update_count=self.state_update_count.clone(),
            observation_ids=frozenset(self._committed_observation_ids),
            observation_order=tuple(self._committed_observation_order),
        )

    @torch.no_grad()
    def _restore_commit_state(self, snapshot: _GABACommitSnapshot) -> None:
        self.gaba_level.copy_(snapshot.gaba_level)
        self.excitation_ema.copy_(snapshot.excitation_ema)
        self.inhibition_ema.copy_(snapshot.inhibition_ema)
        self.state_update_count.copy_(snapshot.state_update_count)
        self._committed_observation_ids.clear()
        self._committed_observation_ids.update(snapshot.observation_ids)
        self._committed_observation_order.clear()
        self._committed_observation_order.extend(snapshot.observation_order)

    @torch.no_grad()
    def _commit_validated_observation(
        self,
        observation: _ValidatedGABAObservation,
    ) -> None:
        if observation.observation_id in self._committed_observation_ids:
            raise ValueError("observation was already committed")

        excitation_mean = observation.excitation_mean.to(
            device=self.excitation_ema.device,
            dtype=self.excitation_ema.dtype,
        )
        inhibition_mean = observation.inhibition_mean.to(
            device=self.inhibition_ema.device,
            dtype=self.inhibition_ema.dtype,
        )
        alpha = 0.02
        self.excitation_ema.mul_(1 - alpha).add_(alpha * excitation_mean)
        self.inhibition_ema.mul_(1 - alpha).add_(alpha * inhibition_mean)
        self.update_gaba_level(excitation_mean, inhibition_mean)
        self.state_update_count.add_(1)

        if (
            self._committed_observation_order.maxlen is not None
            and len(self._committed_observation_order)
            == self._committed_observation_order.maxlen
        ):
            expired = self._committed_observation_order.popleft()
            self._committed_observation_ids.discard(expired)
        self._committed_observation_order.append(observation.observation_id)
        self._committed_observation_ids.add(observation.observation_id)

    @torch.no_grad()
    def commit_observation(self, observation: dict[str, Any]) -> None:
        """Validate then commit one forward observation."""
        validated = self.validate_observation(observation)
        snapshot = self._snapshot_commit_state()
        try:
            self._commit_validated_observation(validated)
        except Exception:
            self._restore_commit_state(snapshot)
            raise


class EIBalanceMonitor(nn.Module):
    """Monitor de balanço Excitação/Inibição.

    Rastreia o balanço E/I através das camadas e alerta se
    ficar desequilibrado.
    """

    def __init__(self, config: GABAConfig):
        super().__init__()
        self.config = config

        # Histórico de E/I ratio
        self.register_buffer('ei_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0))

        # Contadores de alerta
        self.register_buffer('overexcitation_count', torch.tensor(0))
        self.register_buffer('overinhibition_count', torch.tensor(0))

    @torch.no_grad()
    def update(self, ei_ratio: float) -> dict[str, Any]:
        """Atualiza monitoramento e retorna alertas."""
        ptr = int(self.history_ptr.item())
        self.ei_history[ptr] = ei_ratio
        self.history_ptr.fill_((ptr + 1) % 100)

        # Verifica desequilíbrio
        alerts = {
            'overexcitation': ei_ratio > self.config.target_ei_ratio * 1.5,
            'overinhibition': ei_ratio < self.config.target_ei_ratio * 0.5,
            'stable': False
        }

        if alerts['overexcitation']:
            self.overexcitation_count += 1
        elif alerts['overinhibition']:
            self.overinhibition_count += 1
        else:
            alerts['stable'] = True

        return alerts

    def get_stability_report(self) -> dict[str, Any]:
        """Retorna relatório de estabilidade."""
        recent = self.ei_history[-50:] if self.history_ptr >= 50 else self.ei_history[:self.history_ptr]
        mean_ei = recent.mean().item() if recent.numel() > 0 else 0.0
        std_ei = recent.std().item() if recent.numel() > 0 else 0.0

        return {
            'mean_ei_ratio': mean_ei,
            'std_ei_ratio': std_ei,
            'overexcitation_events': self.overexcitation_count.item(),
            'overinhibition_events': self.overinhibition_count.item(),
            'is_stable': 0.5 < mean_ei < 1.5 and std_ei < 0.5
        }


class InhibitoryGating(nn.Module):
    """Gate inibitório que controla o fluxo de informação.

    Similar ao gânglio basal: inibe caminhos não selecionados,
    permitindo que apenas informações relevantes passem.
    """

    def __init__(self, config: GABAConfig):
        super().__init__()
        self.config = config

        # Gate inibitório
        self.gate = nn.Linear(config.d_model, config.d_model)

        # Threshold para gate
        self.register_buffer('gate_threshold', torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Aplica gate inibitório.

        Args:
            x: [batch, seq_len, d_model]
            context: [batch, seq_len, d_model] contexto opcional

        Returns:
            [batch, seq_len, d_model] com gate aplicado
        """
        if context is None:
            context = x

        # Computa gate values
        gate_values = self.gate(context)  # [batch, seq_len, d_model]

        # Sigmoid gate: valores altos passam, baixos são bloqueados
        gate_mask = torch.sigmoid(gate_values - self.gate_threshold)

        # Aplica gate
        output = x * gate_mask

        return output


class HomeostaticControl(nn.Module):
    """Controle homeostático do balanço E/I.

    Ajusta dinamicamente a força de inibição para manter equilíbrio.
    """

    def __init__(self, config: GABAConfig):
        super().__init__()
        self.config = config

        # Parâmetros de controle
        self.inhibition_gain = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def adjust_gain(self, ei_ratio: float) -> float:
        """Ajusta ganho de inibição baseado no E/I ratio."""
        target = self.config.target_ei_ratio
        error = ei_ratio - target

        # Se E/I muito alto, aumenta inibição
        if error > 0:
            self.inhibition_gain += 0.01 * error
        else:
            self.inhibition_gain += 0.01 * error

        self.inhibition_gain.data = self.inhibition_gain.data.clamp(0.1, 10.0)

        return self.inhibition_gain.item()


# ═══════════════════════ TESTE ═══════════════════════

def _test_gaba_inhibition():
    """Smoke test do módulo GABA Inhibition."""
    print("Testing GABA Inhibition...")

    config = GABAConfig(d_model=256)

    # Test 1: GABAergic Layer
    gaba = GABAergicLayer(config)
    x = torch.randn(2, 8, 256)
    delta, observation = gaba(x)
    assert delta.shape == (2, 8, 256)
    assert "sample_excitation" in observation
    assert "sample_inhibition" in observation
    print("✓ Test 1: GABAergic layer")

    # Test 2: E/I Balance Monitor
    monitor = EIBalanceMonitor(config)
    alerts = monitor.update(0.8)
    assert 'stable' in alerts
    print("✓ Test 2: E/I monitor")

    # Test 3: Stability report
    for _ in range(10):
        monitor.update(0.7 + random.uniform(-0.1, 0.1))
    report = monitor.get_stability_report()
    assert 'mean_ei_ratio' in report
    assert 'is_stable' in report
    print("✓ Test 3: Stability report")

    # Test 4: Inhibitory Gating
    gate = InhibitoryGating(config)
    gated = gate(x, x)
    assert gated.shape == (2, 8, 256)
    print("✓ Test 4: Inhibitory gating")

    # Test 5: Homeostatic Control
    homeo = HomeostaticControl(config)
    gain = homeo.adjust_gain(1.5)  # E/I alto
    assert gain > 1.0  # inibição deve aumentar
    print("✓ Test 5: Homeostatic control")

    # Test 6: estado só muda após commit explícito
    gaba.train()
    _, observation2 = gaba(x)
    assert gaba.state_update_count.item() == 0
    gaba.commit_observation(observation2)
    assert gaba.state_update_count.item() == 1
    print("✓ Test 6: explicit observation commit updates state")

    # Test 7: GABA level modulation
    initial_gaba = gaba.gaba_level.item()
    # Força high excitation
    gaba.update_gaba_level(torch.ones(1) * 10, torch.ones(1) * 1)
    assert gaba.gaba_level.item() > initial_gaba
    print("✓ Test 7: GABA level modulation")

    # Test 8: Multiple layers integration
    gaba2 = GABAergicLayer(config)
    out3, _ = gaba2(x + delta)
    assert out3.shape == (2, 8, 256)
    print("✓ Test 8: Multiple layers integration")

    print("\n✅ All GABA Inhibition tests passed!")


if __name__ == "__main__":
    import random
    _test_gaba_inhibition()
