"""Per-organ SHA-256 content identities for the Darwin organism.

Every organ gets its own content-addressable identity covering its weights,
config, and runtime state.  This closes the audit loop: backbone_identity
covers the frozen core, causal_cognitive_state_identity covers the TTM + Spider
bridge, and organ_identity covers each organ individually.

Schema: ``organ:<name>:v1:<sha256>``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch


_SCHEMA = "organ"
_VERSION = 1

# ── Organ definitions ──────────────────────────────────────────────
# Each organ is defined by:
#   state_prefixes  — model_state_dict keys that start with these
#   config_keys     — DarwinXConfig fields that control this organ
#   runtime_keys    — keys in the checkpoint's runtime state (heartbeat_state, etc.)

ORGAN_DEFINITIONS: dict[str, dict[str, Any]] = {
    # ── Structural / always present ──
    "core": {
        "state_prefixes": (
            "token_embedding.",
            "blocks.",
            "norm.",
            "lm_head.",
        ),
        "config_keys": (
            "model_name", "d_model", "n_layers", "n_heads", "n_kv_heads",
            "vocab_size", "context_length", "dropout", "qkv_bias",
            "residual_scale_multiplier", "rope_base_train",
        ),
        "runtime_keys": (),
    },
    "moe": {
        # fine_experts.*, shared_experts.*, router weights, expert buffers
        # nested under blocks.*.moe.
        "state_prefixes": ("blocks.",),
        "config_keys": (
            "fine_experts", "shared_experts", "experts_per_token",
            "fine_expert_hidden_dim", "shared_expert_hidden_dim",
            "vertical_routing_scale",
        ),
        "runtime_keys": ("neuroendocrine_state",),
    },
    # ── Organs with dedicated weight modules ──
    "jepa": {
        "state_prefixes": ("jepa_predictor.", "jepa_ref."),
        "config_keys": ("jepa_weight",),
        "runtime_keys": (),
    },
    "mtp": {
        "state_prefixes": ("mtp_heads.",),
        "config_keys": ("mtp_depth", "mtp_weight"),
        "runtime_keys": (),
    },
    "spider_sense": {
        "state_prefixes": ("_spider_sense_module.",),
        "config_keys": ("spider_sense_enabled",),
        "runtime_keys": ("spider_ram",),
    },
    "gaba": {
        # nested per-layer: blocks.*.gaba.*
        "state_prefixes": ("blocks.",),
        "config_keys": ("gaba_enabled",),
        "runtime_keys": (),
    },
    "inter_hemispheric": {
        "state_prefixes": ("inter_hemispheric.",),
        "config_keys": ("inter_hemispheric_enabled",),
        "runtime_keys": (),
    },
    "ttm_residual": {
        "state_prefixes": ("ttm_residual_gate",),
        "config_keys": (
            "ttm_residual_enabled", "ttm_residual_max_scale",
        ),
        "runtime_keys": ("heartbeat_state",),
    },
    # ── Runtime-state-only organs (no dedicated weights) ──
    "heartbeat": {
        "state_prefixes": (),
        "config_keys": (
            "heartbeat_enabled", "heartbeat_think_interval",
            "heartbeat_explore_interval", "heartbeat_memory_capacity",
            "heartbeat_surprise_threshold", "heartbeat_ff_layers",
        ),
        "runtime_keys": ("heartbeat_state",),
    },
    "ghost": {
        # Ghost reuses the core forward pass.  Neuroendocrine ghost buffers
        # (blocks.*.moe.neuroendocrine.ghost_*) are covered by moe/nitro.
        "state_prefixes": (),
        "config_keys": (
            "ghost_enabled", "ghost_weight", "ghost_mask_ratio",
        ),
        "runtime_keys": (),
    },
    "spider_calibration": {
        # Uses SpiderSense weights; calibration is a loss term, not a module.
        "state_prefixes": (),
        "config_keys": (
            "spider_calibration_enabled", "spider_calibration_weight",
        ),
        "runtime_keys": ("spider_ram",),
    },
    # ── Config-only organs (created dynamically, no persistent weights) ──
    "curiosity": {
        "state_prefixes": (),
        "config_keys": ("curiosity_enabled", "curiosity_weight"),
        "runtime_keys": (),
    },
    "dae": {
        "state_prefixes": (),
        "config_keys": ("dae_enabled", "dae_shadow_mode"),
        "runtime_keys": ("dae",),
    },
    "nitro": {
        "state_prefixes": (),
        "config_keys": (
            "nitro_enabled", "nitro_gpu_expert_capacity",
        ),
        "runtime_keys": ("neuroendocrine_state",),
    },
    "decision_engine": {
        "state_prefixes": (),
        "config_keys": ("decision_engine_enabled",),
        "runtime_keys": (),
    },
    "unified_mesh": {
        "state_prefixes": (),
        "config_keys": ("unified_mesh_enabled",),
        "runtime_keys": (),
    },
    "sleep": {
        "state_prefixes": (),
        "config_keys": ("sleep_enabled",),
        "runtime_keys": (),
    },
}


def _select_state(
    state: Mapping[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Return the subset of *state* whose keys start with any prefix."""
    normalized = {
        str(key).replace("_orig_mod.", ""): value
        for key, value in state.items()
    }
    if not prefixes:
        return {}
    selected: dict[str, Any] = {}
    for key in sorted(normalized):
        if key.startswith(prefixes):
            selected[key] = normalized[key]
    return selected


def _select_config(
    config: Mapping[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Return the subset of *config* for the given keys."""
    raw = dict(config) if isinstance(config, Mapping) else {}
    return {
        key: raw.get(key)
        for key in sorted(keys)
        if key in raw
    }


def _feed(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _feed_value(hasher: Any, value: Any) -> None:
    """Hash arbitrary nested value with type tag (same algo as state_identity)."""
    if isinstance(value, torch.Tensor):
        contiguous = value.detach().to(device="cpu").contiguous()
        _feed(hasher, b"tensor")
        _feed(hasher, str(contiguous.dtype).encode("ascii"))
        _feed(hasher, json.dumps(list(contiguous.shape)).encode("ascii"))
        _feed(hasher, memoryview(contiguous.reshape(-1).view(torch.uint8).numpy()).cast("B"))
        return
    if isinstance(value, Mapping):
        _feed(hasher, b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _feed(hasher, str(key).encode("utf-8"))
            _feed_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        _feed(hasher, b"sequence")
        for item in value:
            _feed_value(hasher, item)
        return
    _feed(hasher, b"scalar")
    _feed(hasher, json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=str).encode("utf-8"))


def organ_identity(
    name: str,
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> str:
    """Compute the SHA-256 content identity for a single organ.

    Args:
        name: organ name (must be in ORGAN_DEFINITIONS)
        state: full model_state_dict (will be filtered by organ prefixes)
        config: full DarwinXConfig (will be filtered by organ keys)
        runtime: optional runtime state dict (heartbeat_state, SpiderRAM.state(), etc.)

    Returns:
        identity string: ``organ:<name>:v1:<sha256>``
    """
    definition = ORGAN_DEFINITIONS.get(name)
    if definition is None:
        raise ValueError(f"unknown organ: {name}")
    selected_state = _select_state(state, definition["state_prefixes"])
    selected_config = _select_config(config, definition["config_keys"])
    selected_runtime: dict[str, Any] = {}
    if runtime is not None:
        for key in definition["runtime_keys"]:
            if key in runtime:
                selected_runtime[key] = runtime[key]

    hasher = hashlib.sha256()
    _feed(hasher, f"{_SCHEMA}:{name}:v{_VERSION}".encode("ascii"))
    _feed(hasher, b"config")
    _feed_value(hasher, selected_config)
    _feed(hasher, b"weights")
    _feed_value(hasher, selected_state)
    if selected_runtime:
        _feed(hasher, b"runtime")
        _feed_value(hasher, selected_runtime)
    return f"{_SCHEMA}:{name}:v{_VERSION}:{hasher.hexdigest()}"


def all_organ_identities(
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Compute organ identities for every defined organ.

    Returns a mapping of ``{organ_name: identity_string}``.
    """
    identities: dict[str, str] = {}
    for name in sorted(ORGAN_DEFINITIONS):
        try:
            identities[name] = organ_identity(
                name, state=state, config=config, runtime=runtime,
            )
        except (KeyError, ValueError, TypeError):
            identities[name] = f"{_SCHEMA}:{name}:v{_VERSION}:unavailable"
    return identities


def organ_identity_report(
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full organ identity report with per-organ breakdowns.

    Returns a dict suitable for embedding in a checkpoint manifest.
    """
    identities = all_organ_identities(
        state=state, config=config, runtime=runtime,
    )
    report: dict[str, Any] = {
        "schema": f"{_SCHEMA}-manifest-v{_VERSION}",
        "identities": identities,
        "details": {},
    }
    for name in sorted(ORGAN_DEFINITIONS):
        definition = ORGAN_DEFINITIONS[name]
        selected_state = _select_state(state, definition["state_prefixes"])
        selected_config = _select_config(config, definition["config_keys"])
        selected_runtime: dict[str, Any] = {}
        if runtime is not None:
            for key in definition["runtime_keys"]:
                if key in runtime:
                    selected_runtime[key] = runtime[key]
        report["details"][name] = {
            "state_keys": sorted(selected_state),
            "state_key_count": len(selected_state),
            "config_keys": sorted(selected_config),
            "runtime_keys": sorted(selected_runtime),
            "has_weights": len(selected_state) > 0,
            "has_runtime": len(selected_runtime) > 0,
        }
    return report


__all__ = [
    "ORGAN_DEFINITIONS",
    "all_organ_identities",
    "organ_identity",
    "organ_identity_report",
]
