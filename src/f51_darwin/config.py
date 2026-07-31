from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml


DEFAULT_MODULE_STATES = (
    "candidate",
    "active",
    "frozen",
    "quarantine",
    "merged",
    "dead",
)


def _coerce_scalar(value: Any, target_type: Any) -> Any:
    """Coerce a value to target_type.

    Used by ``from_mapping`` so checkpoints that serialised everything as
    strings (e.g. ``{'vocab_size': '58162', 'dropout': '0.0'}``) still load.
    ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int``.
    """

    origin = get_origin(target_type)

    # Optional[T] / Union[T, None] — find the non-None arg and recurse.
    if origin is Union:
        args = get_args(target_type)
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            return _coerce_scalar(value, arg)
        return value

    # ``bool`` first: ``isinstance(True, int)`` is True in Python.
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes", "on")

    if target_type is int:
        if isinstance(value, bool):
            return int(value)
        return int(value)

    if target_type is float:
        return float(value)

    if target_type is str:
        return str(value)

    # tuple[..., ...], list[...], custom types — pass through untouched.
    return value


def coerce_mapping(cls: type, mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``mapping`` with values coerced to ``cls`` field types.

    Resolves forward-referenced annotations (active here because every module
    uses ``from __future__ import annotations``).
    """

    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # pragma: no cover — defensive: fall back to raw field.type
        hints = {key: field.type for key, field in cls.__dataclass_fields__.items()}

    coerced: dict[str, Any] = {}
    for key, field in cls.__dataclass_fields__.items():
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        target = hints.get(key, field.type)
        coerced[key] = _coerce_scalar(value, target)
    return coerced


@dataclass(frozen=True)
class DarwinConfig:
    model_name: str = "F51-Darwin-SSD-30M"
    init: str = "random"
    tokenizer: str = "f51_sentencepiece_or_bpe"
    vocab_size: int = 16000
    context_length: int = 1024
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ssd_attention_ratio: str = "3:1"
    mlp_ratio: int = 4
    dropout: float = 0.0
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    positioning: str = "local_or_rope_only_for_attention"
    weight_tying: bool = True
    module_states: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODULE_STATES)

    def __post_init__(self) -> None:
        if self.init != "random":
            raise ValueError("F51 Darwin-SSD v0 only supports random initialization.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.context_length <= 1:
            raise ValueError("context_length must be greater than 1.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if self.ssd_attention_ratio != "3:1":
            raise ValueError("v0 contract requires ssd_attention_ratio='3:1'.")
        if tuple(self.module_states) != DEFAULT_MODULE_STATES:
            raise ValueError(f"module_states must equal {DEFAULT_MODULE_STATES}.")

    @property
    def attention_layer_indices(self) -> tuple[int, ...]:
        return tuple(index for index in range(self.n_layers) if (index + 1) % 4 == 0)

    @property
    def ssd_layer_indices(self) -> tuple[int, ...]:
        attention_indices = set(self.attention_layer_indices)
        return tuple(index for index in range(self.n_layers) if index not in attention_indices)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DarwinConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("YAML config must be a mapping.")
        allowed = cls.__dataclass_fields__.keys()
        kwargs = {key: raw[key] for key in allowed if key in raw}
        if "module_states" in kwargs:
            kwargs["module_states"] = tuple(kwargs["module_states"])
        return cls(**kwargs)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DarwinConfig":
        kwargs = coerce_mapping(cls, raw)
        if "module_states" in kwargs:
            kwargs["module_states"] = tuple(kwargs["module_states"])
        return cls(**kwargs)

