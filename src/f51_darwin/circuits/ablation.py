"""Discovery-split ranking and independently anchored paired causal ablation."""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from f51_darwin.circuits.identity import canonical_sha256
from f51_darwin.circuits.manifest import TapContract
from f51_darwin.circuits.taps import CircuitTapError, TapCapture, resolve_module


_MODEL_INSTANCE_COUNTER = itertools.count()
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AblationArm(str, Enum):
    CLEAN = "clean"
    ABLATE = "ablate"
    RESTORE = "restore"
    SHUFFLE = "shuffle"
    RANDOM_MATCHED = "random_matched"


@dataclass(frozen=True)
class CircuitSelector:
    layer_path: str
    channels: tuple[int, ...]
    baseline: tuple[float, ...] = ()
    discovery_input_sha256: str | None = None
    discovery_item_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.layer_path, str) or not self.layer_path:
            raise ValueError("layer_path must be a non-empty string")
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        if not channels:
            raise ValueError("channels must not be empty")
        if any(type(channel) is not int or channel < 0 for channel in channels):
            raise ValueError("channels must contain non-negative integers")
        if len(set(channels)) != len(channels):
            raise ValueError("channels must be unique")
        baseline = tuple(float(value) for value in self.baseline)
        object.__setattr__(self, "baseline", baseline)
        if baseline and len(baseline) != len(channels):
            raise ValueError("baseline must match channels")
        if any(not math.isfinite(value) for value in baseline):
            raise ValueError("baseline must be finite")
        if (
            self.discovery_input_sha256 is not None
            and (
                not isinstance(self.discovery_input_sha256, str)
                or _SHA256_PATTERN.fullmatch(self.discovery_input_sha256) is None
            )
        ):
            raise ValueError("discovery_input_sha256 must be a lowercase SHA-256 digest")
        discovery_items = tuple(self.discovery_item_sha256)
        object.__setattr__(self, "discovery_item_sha256", discovery_items)
        if any(
            not isinstance(value, str)
            or _SHA256_PATTERN.fullmatch(value) is None
            for value in discovery_items
        ):
            raise ValueError(
                "discovery_item_sha256 must contain lowercase SHA-256 digests"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_path": self.layer_path,
            "channels": list(self.channels),
            "baseline": list(self.baseline),
            "discovery_input_sha256": self.discovery_input_sha256,
            "discovery_item_sha256": list(self.discovery_item_sha256),
        }


@dataclass(frozen=True)
class ArmMetrics:
    accuracy: float
    mean_nll: float
    per_item_nll: tuple[float, ...]
    logits_sha256: str
    max_abs_logit_error: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "mean_nll": self.mean_nll,
            "per_item_nll": list(self.per_item_nll),
            "logits_sha256": self.logits_sha256,
            "max_abs_logit_error": self.max_abs_logit_error,
        }


@dataclass(frozen=True)
class AblationResult:
    seed: int
    selector: CircuitSelector
    tap_width: int
    clean: ArmMetrics
    ablate: ArmMetrics
    restore: ArmMetrics
    shuffle: ArmMetrics
    random_matched: ArmMetrics
    domains: tuple[str, ...]
    confirmation_item_sha256: tuple[str, ...]
    confirmation_batch_sizes: tuple[int, ...]
    anchors: Mapping[str, Mapping[str, str]]
    anchors_identical: bool
    shuffle_permutations: tuple[tuple[int, ...], ...]
    random_matched_channels: tuple[int, ...]
    model_instance_ids: Mapping[str, int]
    checks: Mapping[str, bool | int]
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.evidence_sha256:
            if (
                not isinstance(self.evidence_sha256, str)
                or _SHA256_PATTERN.fullmatch(self.evidence_sha256) is None
            ):
                raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")
        else:
            object.__setattr__(
                self,
                "evidence_sha256",
                canonical_sha256(self._evidence_payload()),
            )

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "selector": self.selector.to_dict(),
            "tap_width": self.tap_width,
            "arms": {
                "clean": self.clean.to_dict(),
                "ablate": self.ablate.to_dict(),
                "restore": self.restore.to_dict(),
                "shuffle": self.shuffle.to_dict(),
                "random_matched": self.random_matched.to_dict(),
            },
            "domains": list(self.domains),
            "confirmation_item_sha256": list(self.confirmation_item_sha256),
            "confirmation_batch_sizes": list(self.confirmation_batch_sizes),
            "anchors": {key: dict(value) for key, value in sorted(self.anchors.items())},
            "anchors_identical": self.anchors_identical,
            "shuffle_permutations": [list(item) for item in self.shuffle_permutations],
            "random_matched_channels": list(self.random_matched_channels),
            "model_instance_ids": {
                key: value for key, value in sorted(self.model_instance_ids.items())
            },
            "checks": dict(self.checks),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._evidence_payload()
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


@dataclass(frozen=True)
class CausalGateConfig:
    minimum_seeds: int = 5
    minimum_items_per_domain: int = 200
    minimum_effect_nats: float = 0.15
    minimum_control_ratio: float = 2.0
    minimum_restore_fraction: float = 0.95
    minimum_shuffle_fraction: float = 0.50
    alpha: float = 0.01
    bootstrap_draws: int = 4096
    signflip_draws: int = 8192
    seed: int = 51

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in (
            "minimum_seeds",
            "minimum_items_per_domain",
            "bootstrap_draws",
            "signflip_draws",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError(f"{name} must be an integer in [1, 1000000]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for name in (
            "minimum_effect_nats",
            "minimum_control_ratio",
            "minimum_restore_fraction",
            "minimum_shuffle_fraction",
            "alpha",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if self.minimum_effect_nats < 0.0:
            raise ValueError("minimum_effect_nats must be >= 0")
        if self.minimum_control_ratio <= 0.0:
            raise ValueError("minimum_control_ratio must be > 0")
        if not 0.0 <= self.minimum_restore_fraction <= 1.0:
            raise ValueError("minimum_restore_fraction must be in [0, 1]")
        if not 0.0 <= self.minimum_shuffle_fraction <= 1.0:
            raise ValueError("minimum_shuffle_fraction must be in [0, 1]")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")


@dataclass(frozen=True)
class CausalVerdict:
    causal: bool
    reason: str
    target_effect_mean: float
    target_effect_ci_low: float
    target_effect_ci_high: float
    control_effect_mean: float
    target_to_control_ratio: float
    restore_fraction: float
    shuffle_fraction: float
    domain_p_values: Mapping[str, float]
    holm_rejected: bool


@dataclass(frozen=True)
class _RegisteredTensorState:
    kind: str
    name: str
    alias_index: int
    persistent: bool | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    content: torch.Tensor


def _registered_tensors(
    model: nn.Module,
) -> list[tuple[str, str, torch.Tensor, bool | None]]:
    entries: list[tuple[str, str, torch.Tensor, bool | None]] = []
    for module_path, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_path}." if module_path else ""
        entries.extend(
            ("parameter", prefix + name, value, None)
            for name, value in module._parameters.items()
            if isinstance(value, torch.Tensor)
        )
        entries.extend(
            (
                "buffer",
                prefix + name,
                value,
                name not in module._non_persistent_buffers_set,
            )
            for name, value in module._buffers.items()
            if isinstance(value, torch.Tensor)
        )
    entries.sort(key=lambda item: (item[0], item[1]))
    keys = [(kind, name) for kind, name, _value, _persistent in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("model has duplicate named registration paths")
    return entries


def _snapshot_registered_state(model: nn.Module) -> tuple[_RegisteredTensorState, ...]:
    aliases: dict[int, int] = {}
    records = []
    for kind, name, value, persistent in _registered_tensors(model):
        alias_index = aliases.setdefault(id(value), len(aliases))
        records.append(
            _RegisteredTensorState(
                kind=kind,
                name=name,
                alias_index=alias_index,
                persistent=persistent,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                requires_grad=bool(value.requires_grad),
                content=value.detach().clone(),
            )
        )
    return tuple(records)


def _restore_registered_state(
    model: nn.Module,
    expected: tuple[_RegisteredTensorState, ...],
) -> None:
    current_entries = _registered_tensors(model)
    if [(kind, name) for kind, name, _value, _persistent in current_entries] != [
        (record.kind, record.name) for record in expected
    ]:
        raise ValueError("model registration topology differs from the initial anchor")
    aliases: dict[int, int] = {}
    restored_aliases: set[int] = set()
    with torch.no_grad():
        for record, (kind, name, value, persistent) in zip(expected, current_entries):
            alias_index = aliases.setdefault(id(value), len(aliases))
            metadata = (
                kind,
                name,
                alias_index,
                persistent,
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                bool(value.requires_grad),
            )
            expected_metadata = (
                record.kind,
                record.name,
                record.alias_index,
                record.persistent,
                record.shape,
                record.dtype,
                record.device,
                record.requires_grad,
            )
            if metadata != expected_metadata:
                raise ValueError("model registration metadata differs from the initial anchor")
            if alias_index not in restored_aliases:
                value.copy_(record.content)
                restored_aliases.add(alias_index)


def _registered_state_digest(model: nn.Module) -> str:
    state = _snapshot_registered_state(model)
    return _digest_value(
        [
            (
                record.kind,
                record.name,
                record.alias_index,
                record.persistent,
                record.shape,
                record.dtype,
                record.device,
                record.requires_grad,
                record.content,
            )
            for record in state
        ]
    )


def _unpack_batch(batch: object) -> tuple[object, torch.Tensor, tuple[str, ...]]:
    if isinstance(batch, Mapping):
        if "labels" not in batch:
            raise ValueError("batch mapping requires labels")
        labels = batch["labels"]
        domains_value = batch.get("domains", batch.get("domain", "default"))
        inputs = {
            key: value
            for key, value in batch.items()
            if key not in {"labels", "domains", "domain"}
        }
    elif isinstance(batch, (tuple, list)) and len(batch) in {2, 3}:
        inputs, labels = batch[0], batch[1]
        domains_value = batch[2] if len(batch) == 3 else "default"
    else:
        raise ValueError("batch must be (inputs, labels[, domains]) or a mapping")
    if not isinstance(labels, torch.Tensor) or labels.ndim < 1:
        raise ValueError("labels must be a batched tensor")
    batch_size = int(labels.shape[0])
    if isinstance(domains_value, str):
        domains = (domains_value,) * batch_size
    else:
        domains = tuple(str(item) for item in domains_value)
    if len(domains) != batch_size or any(not item for item in domains):
        raise ValueError("domains must contain one non-empty value per item")
    return inputs, labels, domains


def _forward(model: nn.Module, inputs: object) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        output = model(**inputs)
    elif isinstance(inputs, (tuple, list)):
        output = model(*inputs)
    else:
        output = model(inputs)
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("model output must be a tensor or expose .logits")
    return logits


def _task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and labels.ndim == 1:
        return F.cross_entropy(logits, labels)
    if logits.ndim == labels.ndim + 1 and logits.shape[:-1] == labels.shape:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    raise ValueError("labels do not match classification logits")


def _tap_activation_mean(
    model: nn.Module, batches: Sequence[object], tap: TapContract
) -> torch.Tensor:
    """Média por canal da ativação no tap, sem gradiente."""
    total = torch.zeros(tap.width, dtype=torch.float64)
    items = 0
    with TapCapture(model, tap) as capture:
        for batch in batches:
            inputs, _labels, _domains = _unpack_batch(batch)
            with torch.no_grad():
                _forward(model, inputs)
            activation = capture.activation
            if activation is None:
                raise CircuitTapError("tap produced no activation")
            reduce_dims = tuple(range(activation.ndim - 1))
            total += activation.detach().double().sum(dim=reduce_dims).cpu()
            items += math.prod(activation.shape[:-1])
    if items == 0:
        raise ValueError("discovery batches must not be empty")
    return total / items


def discover_residual_channels(
    model: nn.Module,
    batches: Iterable[object],
    tap: TapContract,
    count: int,
    *,
    center: bool = False,
) -> CircuitSelector:
    """Ranqueia canais do tap por atribuição no split de descoberta.

    ``center=False`` pontua ``|ativação × gradiente|``. Medido no distilgpt2:
    esse escore é dominado por magnitude — 21 dos 32 canais escolhidos eram
    também o top-32 de magnitude bruta, com variação entre itens igual à dos
    demais (1.04 contra 1.01). São canais que sustentam o modelo mas não
    carregam o conteúdo da habilidade: ablacioná-los custou 4.88 nats,
    enquanto permutá-los entre itens custou 0.16 — e o arm SHUFFLE reprovou,
    corretamente.

    ``center=True`` pontua ``|(ativação − média) × gradiente|``. Remover a
    componente constante faz o escore medir desvio específico do item, que é
    a mesma coisa que SHUFFLE testa. Custa uma passada extra sem gradiente
    para estimar a média.

    Em ambos os modos a linha de base devolvida continua sendo a média — é
    ela que o arm ABLATE injeta.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if count > tap.width:
        raise ValueError("count cannot exceed tap width")
    score = torch.zeros(tap.width, dtype=torch.float64)
    activation_sum = torch.zeros(tap.width, dtype=torch.float64)
    activation_items = 0
    observed = 0
    model.zero_grad(set_to_none=True)
    discovery_batches = [_clone_value(batch) for batch in batches]
    centre = (
        _tap_activation_mean(model, discovery_batches, tap) if center else None
    )
    with TapCapture(model, tap) as capture:
        for batch in discovery_batches:
            inputs, labels, _domains = _unpack_batch(batch)
            model.zero_grad(set_to_none=True)
            logits = _forward(model, inputs)
            activation = capture.activation
            if activation is None or not activation.requires_grad:
                raise CircuitTapError("tapped activation does not carry gradients")
            activation.retain_grad()
            _task_loss(logits, labels).backward()
            if activation.grad is None:
                raise CircuitTapError("missing tapped activation gradient")
            reduce_dims = tuple(range(activation.ndim - 1))
            values = activation.detach()
            if centre is not None:
                values = values - centre.to(
                    device=values.device, dtype=values.dtype
                )
            contribution = (
                values * activation.grad.detach()
            ).abs().mean(dim=reduce_dims)
            baseline_sum = activation.detach().double().sum(dim=reduce_dims)
            item_count = math.prod(activation.shape[:-1])
            if not torch.isfinite(contribution).all() or not torch.isfinite(baseline_sum).all():
                raise CircuitTapError("non-finite discovery score or activation")
            score += contribution.detach().cpu().double()
            activation_sum += baseline_sum.detach().cpu()
            activation_items += item_count
            observed += 1
    if observed == 0 or activation_items == 0:
        raise ValueError("discovery batches must not be empty")
    if not torch.isfinite(score).all():
        raise CircuitTapError("non-finite accumulated discovery score")
    ranked = torch.topk(score, k=count, largest=True, sorted=False).indices.tolist()
    channels = tuple(sorted(int(index) for index in ranked))
    if len(set(channels)) != count:
        raise CircuitTapError("discovery produced duplicate channel indices")
    baseline = tuple(float((activation_sum[index] / activation_items).item()) for index in channels)
    return CircuitSelector(
        tap.module_path,
        channels,
        baseline,
        discovery_input_sha256=_digest_value(discovery_batches),
        discovery_item_sha256=tuple(
            digest
            for batch in discovery_batches
            for digest in _batch_item_digests(batch)
        ),
    )


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return copy.deepcopy(value)


def _slice_item(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_item(item, index, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_slice_item(item, index, batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_item(item, index, batch_size) for item in value]
    return value


def _batch_item_digests(batch: object) -> tuple[str, ...]:
    inputs, labels, _domains = _unpack_batch(batch)
    batch_size = int(labels.shape[0])
    return tuple(
        _digest_value(_slice_item(inputs, index, batch_size))
        for index in range(batch_size)
    )


def _digest_value(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _model_digest(model: nn.Module) -> str:
    return _registered_state_digest(model)


def _snapshot_rng() -> tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None, tuple]:
    cuda_state = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    return random.getstate(), torch.get_rng_state().clone(), cuda_state, np.random.get_state()


def _restore_rng(
    snapshot: tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None, tuple],
) -> None:
    python_state, torch_state, cuda_state, numpy_state = snapshot
    random.setstate(python_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(list(cuda_state))
    np.random.set_state(numpy_state)


def _rng_digest() -> str:
    return _digest_value(
        (
            random.getstate(),
            torch.get_rng_state(),
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
            np.random.get_state(),
        )
    )


def _per_item_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and labels.ndim == 1:
        return F.cross_entropy(logits, labels, reduction="none")
    if logits.ndim == labels.ndim + 1 and logits.shape[:-1] == labels.shape:
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(labels.shape)
        mask = labels.ne(-100)
        counts = mask.sum(dim=tuple(range(1, labels.ndim))).clamp_min(1)
        return (token_loss * mask).sum(dim=tuple(range(1, labels.ndim))) / counts
    raise ValueError("labels do not match classification logits")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(-100)
    return int(((predictions == labels) & mask).sum().item()), int(mask.sum().item())


def _derangement(size: int, seed: int, batch_index: int) -> tuple[int, ...]:
    if size <= 1:
        return tuple(range(size))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729 * (batch_index + 1))
    offset = int(torch.randint(1, size, (1,), generator=generator).item())
    return tuple((index + offset) % size for index in range(size))


def _shuffle_permutations_valid(
    permutations: tuple[tuple[int, ...], ...],
    item_count: int,
    batch_sizes: tuple[int, ...] | None = None,
) -> bool:
    if batch_sizes is not None:
        if len(permutations) != len(batch_sizes):
            return False
        if any(len(permutation) != size for permutation, size in zip(permutations, batch_sizes)):
            return False
    if sum(len(permutation) for permutation in permutations) != item_count:
        return False
    for permutation in permutations:
        size = len(permutation)
        if size <= 1 or set(permutation) != set(range(size)):
            return False
        if any(index == target for index, target in enumerate(permutation)):
            return False
    return True


class _RuntimeTap:
    """Tensor-only intervention hook when the selector does not carry a full tap contract."""

    def __init__(
        self,
        model: nn.Module,
        layer_path: str,
        intervention: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        self.module = resolve_module(model, layer_path)
        self.intervention = intervention
        self.activation: torch.Tensor | None = None
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(
        self, _module: nn.Module, _inputs: tuple[object, ...], output: object
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise CircuitTapError("runtime selector requires a tensor module output")
        if output.ndim < 2:
            raise CircuitTapError("runtime selector requires a batched activation")
        if not torch.isfinite(output).all():
            raise CircuitTapError("runtime selector observed a non-finite activation")
        self.activation = output
        if self.intervention is None:
            return output
        replacement = self.intervention(output)
        if replacement.shape != output.shape or replacement.dtype != output.dtype:
            raise CircuitTapError("runtime intervention changed activation metadata")
        if not torch.isfinite(replacement).all():
            raise CircuitTapError("runtime intervention produced a non-finite activation")
        return replacement

    def __enter__(self) -> _RuntimeTap:
        if getattr(self.module, "_f51_circuit_tap_occupied", False):
            raise CircuitTapError("tap is occupied")
        setattr(self.module, "_f51_circuit_tap_occupied", True)
        try:
            self.handle = self.module.register_forward_hook(self._hook)
        except Exception:
            delattr(self.module, "_f51_circuit_tap_occupied")
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        if self.handle is not None:
            self.handle.remove()
        if getattr(self.module, "_f51_circuit_tap_occupied", False):
            delattr(self.module, "_f51_circuit_tap_occupied")


def run_paired_ablation(
    model_factory: Callable[[], nn.Module],
    batches: Iterable[object],
    selector: CircuitSelector,
    seed: int,
    tap: TapContract | None = None,
) -> AblationResult:
    """Executa os cinco arms pareados a partir de âncoras idênticas.

    ``tap`` opcional: sem ele a intervenção usa ``_RuntimeTap``, um forward
    hook *post* que exige saída tensor. Isso impede intervir no residual
    stream, cujo ponto natural é o *pre* hook de um bloco — o bloco devolve
    tupla. Passando um ``TapContract`` a intervenção usa ``TapCapture``, que
    honra ``position`` e valida rank, largura e comprimento mínimo.

    A descoberta e a confirmação **precisam** usar o mesmo tap: intervir em
    pontos diferentes mediria outro circuito.
    """
    if tap is not None:
        if not isinstance(tap, TapContract):
            raise ValueError("tap must be a TapContract")
        if tap.module_path != selector.layer_path:
            raise ValueError("tap module_path must match the selector layer_path")
    if not selector.discovery_item_sha256 or any(
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
        for value in selector.discovery_item_sha256
    ):
        raise ValueError("selector requires non-empty discovery item evidence")
    materialized = [_clone_value(batch) for batch in batches]
    if not materialized:
        raise ValueError("confirmation batches must not be empty")
    unpacked = [_unpack_batch(batch) for batch in materialized]
    domains = tuple(domain for _inputs, _labels, values in unpacked for domain in values)
    confirmation_batch_sizes = tuple(int(labels.shape[0]) for _inputs, labels, _ in unpacked)
    confirmation_item_sha256 = tuple(
        digest for batch in materialized for digest in _batch_item_digests(batch)
    )
    input_digest = _digest_value(materialized)
    discovery_overlap = set(selector.discovery_item_sha256).intersection(
        confirmation_item_sha256
    )
    if discovery_overlap:
        raise ValueError("confirmation evidence reuses the discovery split")
    batch_order_digest = _digest_value(
        [(index, values) for index, (_inputs, _labels, values) in enumerate(unpacked)]
    )

    initial_model = model_factory()
    if not isinstance(initial_model, nn.Module):
        raise TypeError("model_factory must return an nn.Module")
    initial_state = _snapshot_registered_state(initial_model)
    initial_rng = _snapshot_rng()
    tap_width: int | None = None
    baseline_values = selector.baseline or (0.0,) * len(selector.channels)
    arm_metrics: dict[AblationArm, ArmMetrics] = {}
    arm_logits: dict[AblationArm, list[torch.Tensor]] = {}
    anchors: dict[str, dict[str, str]] = {}
    model_instance_ids: dict[str, int] = {}
    actual_model_ids: list[int] = []
    live_models = [initial_model]
    clean_activations: list[torch.Tensor] = []
    shuffle_permutations: list[tuple[int, ...]] = []
    random_channels: tuple[int, ...] = ()

    for arm in AblationArm:
        model = model_factory()
        live_models.append(model)
        _restore_registered_state(model, initial_state)
        _restore_rng(initial_rng)
        actual_model_ids.append(id(model))
        model_instance_ids[arm.value] = next(_MODEL_INSTANCE_COUNTER)
        anchors[arm.value] = {
            "input_digest": input_digest,
            "model_digest": _model_digest(model),
            "rng_digest": _rng_digest(),
            "batch_order_digest": batch_order_digest,
        }
        correct = 0
        total = 0
        item_nll: list[float] = []
        outputs: list[torch.Tensor] = []
        arm_capture_index = 0

        def intervene(activation: torch.Tensor) -> torch.Tensor:
            nonlocal random_channels
            width = int(activation.shape[-1])
            if max(selector.channels) >= width:
                raise ValueError("selector channel exceeds tap width")
            replacement = activation.clone()
            channel_index = torch.tensor(selector.channels, device=activation.device)
            baseline = torch.tensor(
                baseline_values, dtype=activation.dtype, device=activation.device
            )
            if arm is AblationArm.ABLATE:
                replacement[..., channel_index] = baseline
            elif arm is AblationArm.RESTORE:
                replacement[..., channel_index] = baseline
                clean = clean_activations[arm_capture_index].to(
                    device=activation.device, dtype=activation.dtype
                )
                replacement[..., channel_index] = clean[..., channel_index]
            elif arm is AblationArm.SHUFFLE:
                permutation = _derangement(activation.shape[0], seed, arm_capture_index)
                if len(shuffle_permutations) <= arm_capture_index:
                    shuffle_permutations.append(permutation)
                permutation_tensor = torch.tensor(permutation, device=activation.device)
                replacement[..., channel_index] = activation.index_select(
                    0, permutation_tensor
                )[..., channel_index]
            elif arm is AblationArm.RANDOM_MATCHED:
                if not random_channels:
                    available = [
                        index for index in range(width) if index not in selector.channels
                    ]
                    if len(available) < len(selector.channels):
                        raise ValueError("insufficient outside channels for matched control")
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(seed + 65537)
                    order = torch.randperm(len(available), generator=generator).tolist()
                    random_channels = tuple(
                        sorted(available[index] for index in order[: len(selector.channels)])
                    )
                random_index = torch.tensor(random_channels, device=activation.device)
                replacement[..., random_index] = 0.0
            return replacement

        intervention = None if arm is AblationArm.CLEAN else intervene
        capture_context = (
            TapCapture(model, tap, intervention=intervention)
            if tap is not None
            else _RuntimeTap(model, selector.layer_path, intervention)
        )
        with capture_context as capture:
            for batch_index, (inputs, labels, _batch_domains) in enumerate(unpacked):
                arm_capture_index = batch_index
                with torch.no_grad():
                    logits = _forward(model, inputs)
                activation = capture.activation
                if activation is None:
                    raise CircuitTapError("tap produced no activation")
                observed_width = int(activation.shape[-1])
                if tap_width is None:
                    tap_width = observed_width
                elif tap_width != observed_width:
                    raise CircuitTapError("tap width changed across paired arms or batches")
                if any(channel >= tap_width for channel in selector.channels):
                    raise ValueError("selector channel exceeds tap width")
                if arm is AblationArm.CLEAN:
                    clean_activations.append(activation.detach().cpu().clone())
                if not torch.isfinite(logits).all():
                    raise ValueError(f"{arm.value} produced non-finite logits")
                detached_logits = logits.detach().cpu()
                outputs.append(detached_logits)
                losses = _per_item_nll(logits, labels).detach().cpu()
                item_nll.extend(float(value) for value in losses)
                batch_correct, batch_total = _accuracy(logits, labels)
                correct += batch_correct
                total += batch_total
        arm_logits[arm] = outputs
        clean_outputs = arm_logits.get(AblationArm.CLEAN)
        max_error = 0.0
        if clean_outputs is not None and arm is not AblationArm.CLEAN:
            max_error = max(
                float((actual - expected).abs().max().item())
                for actual, expected in zip(outputs, clean_outputs)
            )
        arm_metrics[arm] = ArmMetrics(
            accuracy=correct / total if total else float("nan"),
            mean_nll=sum(item_nll) / len(item_nll),
            per_item_nll=tuple(item_nll),
            logits_sha256=_digest_value(outputs),
            max_abs_logit_error=max_error,
        )

    first_anchor = next(iter(anchors.values()))
    anchors_identical = all(anchor == first_anchor for anchor in anchors.values())
    if not anchors_identical:
        raise RuntimeError("paired ablation arm anchors are not identical")
    if len(set(actual_model_ids)) != len(AblationArm):
        raise RuntimeError("paired ablation arms reused a model instance")
    if tap_width is None:
        raise RuntimeError("paired ablation did not observe a tap width")
    paired_item_count = len(arm_metrics[AblationArm.CLEAN].per_item_nll)
    shuffle_valid = _shuffle_permutations_valid(
        tuple(shuffle_permutations),
        paired_item_count,
        confirmation_batch_sizes,
    )
    matched_control_valid = (
        len(random_channels) == len(selector.channels)
        and len(set(random_channels)) == len(random_channels)
        and set(random_channels).isdisjoint(selector.channels)
        and all(0 <= channel < tap_width for channel in random_channels)
    )
    selector_within_tap_width = all(
        0 <= channel < tap_width for channel in selector.channels
    )
    checks: dict[str, bool | int] = {
        "anchors_complete_and_identical": anchors_identical,
        "independent_model_instances": len(set(actual_model_ids)) == len(AblationArm),
        "shuffle_derangements_valid": shuffle_valid,
        "matched_control_valid": matched_control_valid,
        "discovery_confirmation_disjoint": not discovery_overlap,
        "selector_within_tap_width": selector_within_tap_width,
        "paired_item_count": paired_item_count,
    }
    return AblationResult(
        seed=seed,
        selector=selector,
        tap_width=tap_width,
        clean=arm_metrics[AblationArm.CLEAN],
        ablate=arm_metrics[AblationArm.ABLATE],
        restore=arm_metrics[AblationArm.RESTORE],
        shuffle=arm_metrics[AblationArm.SHUFFLE],
        random_matched=arm_metrics[AblationArm.RANDOM_MATCHED],
        domains=domains,
        confirmation_item_sha256=confirmation_item_sha256,
        confirmation_batch_sizes=confirmation_batch_sizes,
        anchors=anchors,
        anchors_identical=True,
        shuffle_permutations=tuple(shuffle_permutations),
        random_matched_channels=random_channels,
        model_instance_ids=model_instance_ids,
        checks=checks,
    )


def _paired_effect(run: AblationResult, arm: ArmMetrics) -> np.ndarray:
    clean = np.asarray(run.clean.per_item_nll, dtype=np.float64)
    other = np.asarray(arm.per_item_nll, dtype=np.float64)
    if clean.shape != other.shape or clean.ndim != 1 or clean.size == 0:
        raise ValueError("paired arm evidence is missing or misaligned")
    effect = other - clean
    if not np.isfinite(effect).all():
        raise ValueError("paired arm evidence must be finite")
    return effect


def _bootstrap_interval(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        means[start : start + size] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _signflip_p_value(values: np.ndarray, draws: int, seed: int) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        signs = rng.integers(0, 2, size=(size, values.size), dtype=np.int8) * 2 - 1
        means = np.abs((signs * values).mean(axis=1))
        extreme += int((means >= observed).sum())
    return (extreme + 1.0) / (draws + 1.0)


_ANCHOR_FIELDS = frozenset(
    {"input_digest", "model_digest", "rng_digest", "batch_order_digest"}
)
_CHECK_FIELDS = frozenset(
    {
        "anchors_complete_and_identical",
        "independent_model_instances",
        "shuffle_derangements_valid",
        "matched_control_valid",
        "discovery_confirmation_disjoint",
        "selector_within_tap_width",
        "paired_item_count",
    }
)


def _validate_arm_metrics(metrics: ArmMetrics, item_count: int) -> None:
    if (
        not math.isfinite(metrics.accuracy)
        or not 0.0 <= metrics.accuracy <= 1.0
        or not math.isfinite(metrics.mean_nll)
        or not math.isfinite(metrics.max_abs_logit_error)
        or metrics.max_abs_logit_error < 0.0
        or not isinstance(metrics.logits_sha256, str)
        or _SHA256_PATTERN.fullmatch(metrics.logits_sha256) is None
        or len(metrics.per_item_nll) != item_count
        or any(not math.isfinite(value) for value in metrics.per_item_nll)
    ):
        raise ValueError("retention or logit control evidence is missing")
    observed_mean = sum(metrics.per_item_nll) / item_count
    if not math.isclose(metrics.mean_nll, observed_mean, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("arm mean_nll does not match paired item evidence")


def _validate_run_provenance(run: AblationResult) -> set[int]:
    expected_evidence = canonical_sha256(run._evidence_payload())
    if run.evidence_sha256 != expected_evidence:
        raise ValueError("evidence_sha256 does not authenticate the canonical result")
    if not run.selector.discovery_item_sha256 or any(
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
        for value in run.selector.discovery_item_sha256
    ):
        raise ValueError("causal verdict requires non-empty discovery item evidence")
    if type(run.tap_width) is not int or run.tap_width < 1:
        raise ValueError("tap width evidence is missing or invalid")
    selector_channels = tuple(run.selector.channels)
    if (
        not selector_channels
        or any(type(channel) is not int for channel in selector_channels)
        or len(set(selector_channels)) != len(selector_channels)
        or any(not 0 <= channel < run.tap_width for channel in selector_channels)
    ):
        raise ValueError("selector channels are outside the authenticated tap width")
    arm_names = {arm.value for arm in AblationArm}
    if set(run.anchors) != arm_names:
        raise ValueError("anchor maps are incomplete")
    normalized_anchors: list[dict[str, str]] = []
    for arm in AblationArm:
        anchor = dict(run.anchors[arm.value])
        if set(anchor) != _ANCHOR_FIELDS or any(
            not isinstance(value, str)
            or _SHA256_PATTERN.fullmatch(value) is None
            for value in anchor.values()
        ):
            raise ValueError("anchor maps are incomplete or malformed")
        normalized_anchors.append(anchor)
    if not run.anchors_identical or any(
        anchor != normalized_anchors[0] for anchor in normalized_anchors[1:]
    ):
        raise ValueError("anchor maps are not actually identical")

    if set(run.model_instance_ids) != arm_names:
        raise ValueError("model instance evidence is incomplete")
    instance_ids = list(run.model_instance_ids.values())
    if any(type(value) is not int or value < 0 for value in instance_ids):
        raise ValueError("model instance evidence is malformed")
    if len(set(instance_ids)) != len(AblationArm):
        raise ValueError("model instance IDs are not independent")

    item_count = len(run.clean.per_item_nll)
    if item_count == 0 or len(run.domains) != item_count:
        raise ValueError("paired domain evidence is incomplete")
    if len(run.confirmation_item_sha256) != item_count or any(
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
        for value in run.confirmation_item_sha256
    ):
        raise ValueError("confirmation item digest evidence is incomplete")
    if (
        not run.confirmation_batch_sizes
        or any(type(size) is not int or size < 2 for size in run.confirmation_batch_sizes)
        or sum(run.confirmation_batch_sizes) != item_count
    ):
        raise ValueError("confirmation batch evidence is incomplete")
    for metrics in (
        run.clean,
        run.ablate,
        run.restore,
        run.shuffle,
        run.random_matched,
    ):
        _validate_arm_metrics(metrics, item_count)

    shuffle_valid = _shuffle_permutations_valid(
        run.shuffle_permutations,
        item_count,
        run.confirmation_batch_sizes,
    )
    if not shuffle_valid:
        raise ValueError("shuffle permutation evidence is incomplete or not a derangement")
    matched_control_valid = (
        len(run.random_matched_channels) == len(run.selector.channels)
        and len(set(run.random_matched_channels)) == len(run.random_matched_channels)
        and all(
            type(channel) is int and 0 <= channel < run.tap_width
            for channel in run.random_matched_channels
        )
        and set(run.random_matched_channels).isdisjoint(run.selector.channels)
    )
    if not matched_control_valid:
        raise ValueError("matched control channels are invalid")
    discovery_disjoint = not set(run.selector.discovery_item_sha256).intersection(
        run.confirmation_item_sha256
    )
    if not discovery_disjoint:
        raise ValueError("discovery and confirmation item evidence overlaps")
    expected_checks: dict[str, bool | int] = {
        "anchors_complete_and_identical": True,
        "independent_model_instances": True,
        "shuffle_derangements_valid": True,
        "matched_control_valid": True,
        "discovery_confirmation_disjoint": True,
        "selector_within_tap_width": True,
        "paired_item_count": item_count,
    }
    if set(run.checks) != _CHECK_FIELDS or dict(run.checks) != expected_checks:
        raise ValueError("required provenance checks are missing or inconsistent")
    return set(instance_ids)


def build_causal_verdict(
    runs: Sequence[AblationResult], config: CausalGateConfig
) -> CausalVerdict:
    if not isinstance(config, CausalGateConfig):
        raise ValueError("config must be a validated CausalGateConfig")
    config._validate()
    if len(runs) < config.minimum_seeds or len({run.seed for run in runs}) < config.minimum_seeds:
        raise ValueError("minimum_seeds evidence is missing")
    common_selector = runs[0].selector
    common_tap_width = runs[0].tap_width
    all_instance_ids: set[int] = set()
    for run in runs:
        if run.selector != common_selector:
            raise ValueError("all causal runs must use a common selector")
        if run.tap_width != common_tap_width:
            raise ValueError("all causal runs must use a common tap width")
        run_instance_ids = _validate_run_provenance(run)
        if all_instance_ids.intersection(run_instance_ids):
            raise ValueError("model instance IDs are reused across seeds")
        all_instance_ids.update(run_instance_ids)
    target_parts: list[np.ndarray] = []
    control_parts: list[np.ndarray] = []
    restore_parts: list[np.ndarray] = []
    shuffle_parts: list[np.ndarray] = []
    by_domain: dict[str, list[float]] = {}
    for run in runs:
        target = _paired_effect(run, run.ablate)
        control = _paired_effect(run, run.random_matched)
        restore = _paired_effect(run, run.restore)
        shuffle = _paired_effect(run, run.shuffle)
        if len(run.domains) != target.size:
            raise ValueError("domain evidence is missing")
        counts: dict[str, int] = {}
        for domain, effect in zip(run.domains, target):
            counts[domain] = counts.get(domain, 0) + 1
            by_domain.setdefault(domain, []).append(float(effect))
        if any(count < config.minimum_items_per_domain for count in counts.values()):
            raise ValueError("minimum_items_per_domain evidence is missing")
        target_parts.append(target)
        control_parts.append(control)
        restore_parts.append(restore)
        shuffle_parts.append(shuffle)
    if not by_domain:
        raise ValueError("domain evidence is missing")
    target = np.concatenate(target_parts)
    control = np.concatenate(control_parts)
    restore = np.concatenate(restore_parts)
    shuffle = np.concatenate(shuffle_parts)
    target_mean = float(target.mean())
    control_mean = float(control.mean())
    ci_low, ci_high = _bootstrap_interval(target, config.bootstrap_draws, config.seed)
    ratio = (
        float("inf")
        if control_mean <= 0.0 and target_mean > 0.0
        else target_mean / max(control_mean, np.finfo(np.float64).eps)
    )
    restore_fraction = max(0.0, 1.0 - float(np.abs(restore).mean()) / max(abs(target_mean), 1e-12))
    shuffle_fraction = float(shuffle.mean()) / max(target_mean, 1e-12)
    p_values = {
        domain: _signflip_p_value(
            np.asarray(values, dtype=np.float64),
            config.signflip_draws,
            config.seed + index,
        )
        for index, (domain, values) in enumerate(sorted(by_domain.items()))
    }
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    holm_rejected = True
    domain_count = len(ordered)
    for index, (_domain, p_value) in enumerate(ordered):
        if p_value > config.alpha / (domain_count - index):
            holm_rejected = False
            break

    reason = "causal"
    causal = True
    if target_mean < config.minimum_effect_nats or ci_low <= 0.0:
        causal, reason = False, "target_effect_too_small"
    elif ratio < config.minimum_control_ratio:
        causal, reason = False, "matched_control_not_beaten"
    elif restore_fraction < config.minimum_restore_fraction:
        causal, reason = False, "restore_insufficient"
    elif shuffle_fraction < config.minimum_shuffle_fraction:
        causal, reason = False, "shuffle_insufficient"
    elif not holm_rejected:
        causal, reason = False, "holm_not_rejected"
    return CausalVerdict(
        causal=causal,
        reason=reason,
        target_effect_mean=target_mean,
        target_effect_ci_low=ci_low,
        target_effect_ci_high=ci_high,
        control_effect_mean=control_mean,
        target_to_control_ratio=ratio,
        restore_fraction=restore_fraction,
        shuffle_fraction=shuffle_fraction,
        domain_p_values=p_values,
        holm_rejected=holm_rejected,
    )


__all__ = [
    "AblationArm",
    "AblationResult",
    "ArmMetrics",
    "CausalGateConfig",
    "CausalVerdict",
    "CircuitSelector",
    "build_causal_verdict",
    "discover_residual_channels",
    "run_paired_ablation",
]
