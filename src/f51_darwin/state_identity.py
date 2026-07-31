"""Content identities for online-learning lineage.

The adapter is causal only for the exact frozen backbone and tokenizer contract
on which its losses were measured. These identities deliberately exclude the
heartbeat because it is an independently persisted, inference-mutable organ.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from typing import Any

import torch


_CORE_PREFIXES = ("token_embedding.", "blocks.", "norm.", "lm_head.")
_CAUSAL_COGNITIVE_CONFIG_KEYS = frozenset(
    {
        "ttm_residual_enabled",
        "ttm_residual_max_scale",
        "spider_calibration_enabled",
        "spider_calibration_weight",
    }
)
_THREE_ORGAN_CONFIG_KEYS = frozenset(
    {
        "cognitive_architecture_version",
        "cognitive_shadow_enabled",
        "cognitive_pulse_enabled",
        "cognitive_organ_width",
        "cognitive_residual_max_scale",
    }
)
_TRAINING_ONLY_CONFIG_KEYS = frozenset(
    {
        "loss_semantics_version",
        *_CAUSAL_COGNITIVE_CONFIG_KEYS,
        *_THREE_ORGAN_CONFIG_KEYS,
    }
)
_TRAINING_CONTRACT_ID = "darwin-organism-causal-training-v1"
_CAUSAL_COGNITIVE_STATE_SCHEMA = "darwin-causal-cognitive-state-v1"


def _canonical_config(config: Any) -> bytes:
    raw = config if isinstance(config, Mapping) else getattr(config, "__dict__", {})
    raw = dict(raw)
    safe: dict[str, Any] = {}
    for key, value in raw.items():
        if str(key) in _TRAINING_ONLY_CONFIG_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        elif isinstance(value, (list, tuple)):
            safe[str(key)] = list(value)
        elif isinstance(value, Mapping):
            safe[str(key)] = dict(value)
        else:
            safe[str(key)] = str(value)
    return json.dumps(
        safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _feed(hasher: Any, value: Any) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def backbone_identity(state: Mapping[str, Any], config: Any) -> str:
    """Hash canonical config and only tensors that affect frozen inference."""
    hasher = hashlib.sha256()
    _feed(hasher, b"darwin-model-core-v1")
    _feed(hasher, _canonical_config(config))
    normalized = {
        str(key).replace("_orig_mod.", ""): value for key, value in state.items()
    }
    keys = sorted(key for key in normalized if key.startswith(_CORE_PREFIXES))
    if not keys:
        # Small unit-test models may not use Darwin module names. Production
        # Darwin checkpoints always take the explicit core-prefix path above.
        keys = sorted(
            key
            for key, value in normalized.items()
            if isinstance(value, torch.Tensor) and not key.startswith("heartbeat.")
        )
    if not keys:
        raise ValueError("no backbone tensors found for content identity")
    for key in keys:
        tensor = normalized[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        contiguous = tensor.detach().to(device="cpu").contiguous()
        _feed(hasher, key.encode("utf-8"))
        _feed(hasher, str(contiguous.dtype).encode("ascii"))
        _feed(hasher, json.dumps(list(contiguous.shape)).encode("ascii"))
        # ``view(dtype)`` rejects zero-dimensional BF16 tensors because the
        # target element size differs. Flattening preserves the exact byte
        # order for every rank and makes scalar organism buffers hashable.
        byte_view = contiguous.reshape(-1).view(torch.uint8).numpy()
        _feed(hasher, memoryview(byte_view).cast("B"))
    return f"darwin-model-core-v1:{hasher.hexdigest()}"


def tokenizer_identity(tokenizer: Any) -> str:
    """Hash the loaded BPE behavior, independent of its filesystem path."""
    identity_payload = getattr(tokenizer, "identity_payload", None)
    if callable(identity_payload):
        payload = identity_payload()
        if not isinstance(payload, Mapping):
            raise ValueError("tokenizer identity_payload() must return a mapping")
        schema = str(payload.get("schema") or "")
        if schema != "smol-tokenizer-contract-v1":
            raise ValueError(f"unsupported tokenizer identity schema: {schema}")
        digest = hashlib.sha256(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"{schema}:{digest}"

    vocab = sorted(
        (
            (unicodedata.normalize("NFC", str(token)), int(token_id))
            for token, token_id in dict(tokenizer.vocab).items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    merges = [
        [
            unicodedata.normalize("NFC", str(left)),
            unicodedata.normalize("NFC", str(right)),
        ]
        for left, right in tokenizer.merges
    ]
    payload = {
        "schema": "f51-bpe-contract-v1",
        "vocab": vocab,
        "merges": merges,
        "special_ids": {
            name: int(getattr(tokenizer, f"{name}_id"))
            for name in ("pad", "unk", "bos", "eos")
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"f51-bpe-contract-v1:{digest}"


def _raw_config(config: Any) -> dict[str, Any]:
    raw = config if isinstance(config, Mapping) else getattr(config, "__dict__", {})
    return {str(key): value for key, value in dict(raw).items()}


def causal_cognitive_required(config: Any) -> bool:
    """Whether state outside the frozen backbone can affect causal execution."""

    raw = _raw_config(config)
    return bool(
        raw.get("ttm_residual_enabled", False)
        or raw.get("spider_calibration_enabled", False)
    )


def _feed_causal_value(hasher: Any, value: Any) -> None:
    """Hash nested checkpoint state with explicit type and ordering."""

    if isinstance(value, torch.Tensor):
        contiguous = value.detach().to(device="cpu").contiguous()
        _feed(hasher, b"tensor")
        _feed(hasher, str(contiguous.dtype).encode("ascii"))
        _feed(hasher, json.dumps(list(contiguous.shape)).encode("ascii"))
        _feed(
            hasher,
            memoryview(
                contiguous.reshape(-1).view(torch.uint8).numpy()
            ).cast("B"),
        )
        return
    if isinstance(value, Mapping):
        _feed(hasher, b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _feed(hasher, str(key).encode("utf-8"))
            _feed_causal_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        _feed(hasher, b"sequence")
        for item in value:
            _feed_causal_value(hasher, item)
        return
    _feed(hasher, b"scalar")
    _feed(
        hasher,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
    )


def causal_cognitive_state_identity(
    model_state: Mapping[str, Any],
    heartbeat_state: Mapping[str, Any] | None,
    config: Any,
) -> str:
    """Bind every non-backbone state component used by causal cognition.

    ``base_checkpoint_id`` intentionally remains the frozen-core identity.
    Checkpoint v8 persists and verifies this separate identity so a changed TTM
    gate, Spider calibration head, heartbeat projection, or memory slot cannot
    silently resume under the same causal contract.
    """

    raw = _raw_config(config)
    if not causal_cognitive_required(raw):
        raise ValueError("causal cognitive state identity requires an enabled organ")
    normalized = {
        str(key).replace("_orig_mod.", ""): value
        for key, value in model_state.items()
    }
    selected_model_state: dict[str, Any] = {}
    if bool(raw.get("ttm_residual_enabled", False)):
        if "ttm_residual_gate" not in normalized:
            raise ValueError("causal cognitive state is missing ttm_residual_gate")
        if not isinstance(heartbeat_state, Mapping):
            raise ValueError("causal cognitive state is missing heartbeat_state")
        selected_model_state["ttm_residual_gate"] = normalized[
            "ttm_residual_gate"
        ]
    if bool(raw.get("spider_calibration_enabled", False)):
        spider_keys = sorted(
            key
            for key in normalized
            if key.startswith("_spider_sense_module.")
        )
        if not spider_keys:
            raise ValueError("causal cognitive state is missing Spider weights")
        for key in spider_keys:
            selected_model_state[key] = normalized[key]

    contract_config = {
        key: raw.get(key)
        for key in sorted(_CAUSAL_COGNITIVE_CONFIG_KEYS)
    }
    hasher = hashlib.sha256()
    _feed(hasher, _CAUSAL_COGNITIVE_STATE_SCHEMA.encode("ascii"))
    _feed_causal_value(hasher, contract_config)
    _feed_causal_value(hasher, selected_model_state)
    if bool(raw.get("ttm_residual_enabled", False)):
        _feed_causal_value(hasher, heartbeat_state)
    return f"{_CAUSAL_COGNITIVE_STATE_SCHEMA}:{hasher.hexdigest()}"


def training_contract_identity() -> str:
    """Return the causal training protocol identity, separate from the model.

    Loss semantics and the selected causal arm are checkpoint fields validated
    independently.  This identity anchors the step/ledger protocol itself and
    therefore must not be folded into ``base_checkpoint_id``.
    """

    return _TRAINING_CONTRACT_ID


__all__ = [
    "backbone_identity",
    "causal_cognitive_required",
    "causal_cognitive_state_identity",
    "tokenizer_identity",
    "training_contract_identity",
]
