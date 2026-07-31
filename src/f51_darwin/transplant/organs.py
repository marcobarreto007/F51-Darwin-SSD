from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from f51_darwin.gaba_inhibition import GABAConfig, GABAergicLayer
from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
from f51_darwin.inter_hemispheric import (
    HemisphereConfig,
    InterHemisphericSystem,
)
from f51_darwin.jepa_v2 import JEPAHeadV2

from .bundle import OrganBundle


@dataclass
class ReconstructedOrgan:
    name: str
    module: nn.Module
    source_keys: tuple[str, ...]
    source_hashes: dict[str, str]
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


class HeartbeatOrgan(nn.Module):
    """Register Heartbeat submodules while preserving its Python runtime state."""

    def __init__(self, heartbeat: Heartbeat) -> None:
        super().__init__()
        self.ff_stack = heartbeat.ff_stack
        self.tt_memory = heartbeat.tt_memory
        self.thinker = heartbeat.thinker
        object.__setattr__(self, "_heartbeat", heartbeat)

    def beat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._heartbeat.beat(*args, **kwargs)

    @property
    def d_model(self) -> int:
        return int(self._heartbeat.d_model)

    @property
    def memory_capacity(self) -> int:
        return int(self._heartbeat.tt_memory.capacity)

    def export_state(self) -> dict[str, Any]:
        return self._heartbeat.state_dict()

    def get_extra_state(self) -> dict[str, Any]:
        return self.export_state()

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("heartbeat extra state must be a mapping")
        self._heartbeat.load_state_dict(deepcopy(dict(state)))


def _verified_prefixed_state(
    bundle: OrganBundle,
    *,
    organ_name: str,
    prefix: str,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...], dict[str, str]]:
    if organ_name not in bundle.manifest.organs:
        raise ValueError(f"organ {organ_name} is not declared in bundle")
    hash_problems = bundle.verify_tensor_hashes()
    if hash_problems:
        raise ValueError(f"organ bundle tensor hash mismatch: {hash_problems}")
    source_keys = tuple(
        key for key in sorted(bundle.tensors) if key.startswith(prefix)
    )
    state = {
        key.removeprefix(prefix): bundle.tensors[key].detach().clone()
        for key in source_keys
    }
    record_hashes = {
        record.key: record.sha256 for record in bundle.manifest.tensors
    }
    source_hashes = {key: record_hashes[key] for key in source_keys}
    return state, source_keys, source_hashes


def _strict_load(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    organ_name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected = set(module.state_dict())
    actual = set(state)
    missing = tuple(sorted(expected - actual))
    unexpected = tuple(sorted(actual - expected))
    if missing:
        raise ValueError(f"incomplete {organ_name} organ: missing={missing}")
    if unexpected:
        raise ValueError(
            f"unexpected {organ_name} organ tensors: unexpected={unexpected}"
        )
    result = module.load_state_dict(state, strict=True)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return tuple(result.missing_keys), tuple(result.unexpected_keys)


def build_original_ihs(bundle: OrganBundle) -> ReconstructedOrgan:
    prefix = "inter_hemispheric."
    state, source_keys, source_hashes = _verified_prefixed_state(
        bundle,
        organ_name="ihs",
        prefix=prefix,
    )
    donor_dim = int(bundle.donor_config.get("d_model", 512))
    if donor_dim != 512:
        raise ValueError(f"unsupported ihs donor dimension: {donor_dim}")
    module = InterHemisphericSystem(
        HemisphereConfig(
            d_model=donor_dim,
            n_heads=int(bundle.donor_config.get("n_heads", 8)),
            dropout=float(bundle.donor_config.get("dropout", 0.0)),
        )
    )
    missing, unexpected = _strict_load(module, state, organ_name="ihs")
    return ReconstructedOrgan(
        name="ihs",
        module=module,
        source_keys=source_keys,
        source_hashes=source_hashes,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def _heartbeat_component_state(
    bundle: OrganBundle,
    *,
    state_prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(state_prefix): tensor.detach().clone()
        for key, tensor in bundle.tensors.items()
        if key.startswith(state_prefix)
    }


def build_original_heartbeat(bundle: OrganBundle) -> ReconstructedOrgan:
    if "heartbeat" not in bundle.manifest.organs:
        raise ValueError("organ heartbeat is not declared in bundle")
    hash_problems = bundle.verify_tensor_hashes()
    if hash_problems:
        raise ValueError(f"organ bundle tensor hash mismatch: {hash_problems}")
    raw_runtime = bundle.runtime_state.get("heartbeat_state")
    if not isinstance(raw_runtime, Mapping):
        raise ValueError("bundle has no heartbeat runtime state")

    config = HeartbeatConfig(
        think_interval=int(
            bundle.donor_config.get("heartbeat_think_interval", 1)
        ),
        explore_interval=int(
            bundle.donor_config.get("heartbeat_explore_interval", 10)
        ),
        self_reward_interval=int(
            bundle.donor_config.get("heartbeat_explore_interval", 10)
        ),
        ff_layers=int(bundle.donor_config.get("heartbeat_ff_layers", 3)),
        memory_capacity=int(
            bundle.donor_config.get("heartbeat_memory_capacity", 1024)
        ),
        surprise_threshold=float(
            bundle.donor_config.get("heartbeat_surprise_threshold", 0.3)
        ),
    )
    heartbeat = Heartbeat(512, config)
    component_specs = {
        "ff_stack_state_dict": (
            "heartbeat.ff_stack.",
            heartbeat.ff_stack.state_dict(),
        ),
        "tt_memory_state_dict": (
            "heartbeat.tt_memory.",
            heartbeat.tt_memory.state_dict(),
        ),
        "thinker_state_dict": (
            "heartbeat.thinker.",
            heartbeat.thinker.state_dict(),
        ),
    }
    restored_state = deepcopy(dict(raw_runtime))
    source_keys: list[str] = []
    for state_key, (prefix, expected) in component_specs.items():
        component = _heartbeat_component_state(bundle, state_prefix=prefix)
        missing = sorted(set(expected) - set(component))
        unexpected = sorted(set(component) - set(expected))
        if missing:
            raise ValueError(
                f"incomplete heartbeat organ: {state_key} missing={missing}"
            )
        if unexpected:
            raise ValueError(
                f"unexpected heartbeat organ tensors: {state_key} "
                f"unexpected={unexpected}"
            )
        restored_state[state_key] = component
        source_keys.extend(f"{prefix}{key}" for key in sorted(component))

    runtime_slots = restored_state.get("tt_memory_slots", [])
    if not isinstance(runtime_slots, list):
        raise TypeError("heartbeat runtime memory slots must be a list")
    for index, slot in enumerate(runtime_slots):
        if not isinstance(slot, dict):
            raise TypeError("heartbeat runtime memory slot must be a dictionary")
        for tensor_name in ("key", "value"):
            tensor_key = f"heartbeat.tt_memory_slots.{index}.{tensor_name}"
            value = bundle.tensors.get(tensor_key)
            if value is None:
                raise ValueError(
                    f"incomplete heartbeat organ: missing=('{tensor_key}',)"
                )
            slot[tensor_name] = value.detach().clone()
            source_keys.append(tensor_key)
    unexpected_slots = sorted(
        key
        for key in bundle.tensors
        if key.startswith("heartbeat.tt_memory_slots.")
        and key not in source_keys
    )
    if unexpected_slots:
        raise ValueError(
            "unexpected heartbeat organ tensors: "
            f"unexpected={unexpected_slots}"
        )

    heartbeat.load_state_dict(restored_state)
    module = HeartbeatOrgan(heartbeat)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    record_hashes = {
        record.key: record.sha256 for record in bundle.manifest.tensors
    }
    ordered_keys = tuple(sorted(source_keys))
    return ReconstructedOrgan(
        name="heartbeat",
        module=module,
        source_keys=ordered_keys,
        source_hashes={key: record_hashes[key] for key in ordered_keys},
    )


def build_original_jepa(bundle: OrganBundle) -> ReconstructedOrgan:
    prefix = "jepa_predictor."
    state, source_keys, source_hashes = _verified_prefixed_state(
        bundle,
        organ_name="jepa",
        prefix=prefix,
    )
    architecture_shapes = {
        "predictor.0.weight": (256, 512),
        "predictor.3.weight": (768, 256),
        "predictor.6.weight": (256, 768),
        "predictor.9.weight": (512, 256),
    }
    for key, expected_shape in architecture_shapes.items():
        value = state.get(key)
        if value is None:
            raise ValueError(f"incomplete jepa organ: missing=('{key}',)")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "unsupported jepa organ architecture: "
                f"{key} expected={expected_shape} actual={tuple(value.shape)}"
            )
    module = JEPAHeadV2(
        d_model=512,
        hidden_dim=768,
        dropout=0.0,
        bottleneck_dim=256,
    )
    missing, unexpected = _strict_load(module, state, organ_name="jepa")
    return ReconstructedOrgan(
        name="jepa",
        module=module,
        source_keys=source_keys,
        source_hashes=source_hashes,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def build_original_gaba(
    bundle: OrganBundle,
    *,
    donor_layer: int,
) -> ReconstructedOrgan:
    organ_name = f"gaba.{int(donor_layer)}"
    prefix = f"blocks.{int(donor_layer)}.gaba."
    state, source_keys, source_hashes = _verified_prefixed_state(
        bundle,
        organ_name=organ_name,
        prefix=prefix,
    )
    module = GABAergicLayer(GABAConfig(d_model=512))
    missing, unexpected = _strict_load(module, state, organ_name=organ_name)
    return ReconstructedOrgan(
        name=organ_name,
        module=module,
        source_keys=source_keys,
        source_hashes=source_hashes,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def build_original_spider(bundle: OrganBundle) -> ReconstructedOrgan:
    """Reconstruct Spider-Sense organ from extracted bundle tensors."""
    from f51_darwin.spider_sense import SpiderSense

    prefix = "_spider_sense_module."
    state = {key[len(prefix):]: tensor for key, tensor in bundle.tensors.items()
             if key.startswith(prefix)}
    if not state:
        raise ValueError("bundle has no Spider-Sense tensors")
    module = SpiderSense(d_model=512)
    missing, unexpected = _strict_load(module, state, organ_name="spider")
    for param in module.parameters():
        param.requires_grad_(False)
    source_keys = sorted(state.keys())
    source_hashes = {k: {r.key: r.sha256 for r in bundle.manifest.tensors}.get(prefix + k, None)
                     for k in state}
    return ReconstructedOrgan(
        name="spider",
        module=module,
        source_keys=source_keys,
        source_hashes=source_hashes,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def build_original_mtp(bundle: OrganBundle) -> ReconstructedOrgan:
    """Reconstruct MTP heads from extracted bundle tensors."""
    from torch import nn

    prefix = "mtp_heads."
    state = {key[len(prefix):]: tensor for key, tensor in bundle.tensors.items()
             if key.startswith(prefix)}
    if not state:
        raise ValueError("bundle has no MTP tensors")
    # Build heads from state keys
    num_heads = len([k for k in state if k.endswith(".weight")])
    heads = nn.ModuleList([
        nn.Linear(512, 512, bias=False) for _ in range(num_heads)
    ])
    missing, unexpected = _strict_load(heads, state, organ_name="mtp")
    for head in heads:
        for param in head.parameters():
            param.requires_grad_(False)
    source_keys = sorted(state.keys())
    tensor_map = {r.key: r.sha256 for r in bundle.manifest.tensors}
    source_hashes = {k: tensor_map.get(prefix + k) for k in source_keys}
    return ReconstructedOrgan(
        name="mtp",
        module=heads,
        source_keys=source_keys,
        source_hashes=source_hashes,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_ttm_memory(checkpoint_path: Path) -> tuple[nn.Linear, nn.Linear]:
    """Load TTM memory (proj_key, proj_value) from checkpoint heartbeat_state."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hb = ckpt.get("heartbeat_state")
    if hb is None:
        raise ValueError("checkpoint has no heartbeat_state")
    tt = hb.get("tt_memory_state_dict")
    if tt is None:
        raise ValueError("heartbeat_state has no tt_memory_state_dict")
    proj_key = nn.Linear(512, 128, bias=False)
    proj_value = nn.Linear(512, 512, bias=False)
    proj_key.load_state_dict({"weight": tt["proj_key.weight"]})
    proj_value.load_state_dict({"weight": tt["proj_value.weight"]})
    for p in proj_key.parameters(): p.requires_grad_(False)
    for p in proj_value.parameters(): p.requires_grad_(False)
    return proj_key, proj_value
