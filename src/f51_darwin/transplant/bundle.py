from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_BUNDLE_SCHEMA_VERSION = 1
_RUNTIME_KEYS = ("organ_identity_report",)
_EXTRA_ORGAN_NAMES = {
    "_spider_sense_module.": "spider",
    "spider_confidence_head.": "spider",
    "mtp_heads.": "mtp",
    "inter_hemispheric.": "ihs",
}
_HEARTBEAT_COMPONENTS = {
    "ff_stack_state_dict": "heartbeat.ff_stack.",
    "tt_memory_state_dict": "heartbeat.tt_memory.",
    "thinker_state_dict": "heartbeat.thinker.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(
        memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorRecord:
    key: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class OrganBundleManifest:
    schema_version: int
    source_checkpoint: str
    source_checkpoint_sha256: str
    base_checkpoint_id: str
    organs: tuple[str, ...]
    tensors: tuple[TensorRecord, ...]


@dataclass
class OrganBundle:
    manifest: OrganBundleManifest
    tensors: dict[str, torch.Tensor]
    donor_config: dict[str, Any]
    runtime_state: dict[str, Any]

    def verify_tensor_hashes(self) -> list[str]:
        expected = {
            record.key: record.sha256 for record in self.manifest.tensors
        }
        problems = [
            key
            for key in sorted(set(expected) | set(self.tensors))
            if key not in expected
            or key not in self.tensors
            or tensor_sha256(self.tensors[key]) != expected[key]
        ]
        return problems


def _normalize_state(
    raw_state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key).replace("_orig_mod.", "")
        normalized[key] = value
    return normalized


def _selected_key(
    key: str,
    gaba_layers: tuple[int, ...],
    extra_organs: tuple[str, ...] = (),
) -> bool:
    if key.startswith("jepa_predictor."):
        return True
    if any(key.startswith(f"blocks.{layer}.gaba.") for layer in gaba_layers):
        return True
    return any(key.startswith(prefix) for prefix in extra_organs)


def _manifest_from_payload(raw: Mapping[str, Any]) -> OrganBundleManifest:
    records = tuple(
        TensorRecord(
            key=str(item["key"]),
            shape=tuple(int(value) for value in item["shape"]),
            dtype=str(item["dtype"]),
            sha256=str(item["sha256"]),
        )
        for item in raw["tensors"]
    )
    return OrganBundleManifest(
        schema_version=int(raw["schema_version"]),
        source_checkpoint=str(raw["source_checkpoint"]),
        source_checkpoint_sha256=str(raw["source_checkpoint_sha256"]),
        base_checkpoint_id=str(raw["base_checkpoint_id"]),
        organs=tuple(str(value) for value in raw["organs"]),
        tensors=records,
    )


def _extract_heartbeat_state(
    raw: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise ValueError("source checkpoint has no heartbeat_state")
    tensors: dict[str, torch.Tensor] = {}
    runtime = {
        str(key): deepcopy(value)
        for key, value in raw.items()
        if key not in _HEARTBEAT_COMPONENTS and key != "tt_memory_slots"
    }
    for state_key, prefix in _HEARTBEAT_COMPONENTS.items():
        component = raw.get(state_key)
        if not isinstance(component, Mapping):
            raise ValueError(f"heartbeat_state has no {state_key}")
        for raw_key, value in component.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"heartbeat tensor state is not a tensor: {state_key}.{raw_key}"
                )
            tensors[f"{prefix}{raw_key}"] = value.detach().cpu().clone()

    runtime_slots: list[dict[str, Any]] = []
    raw_slots = raw.get("tt_memory_slots", ())
    if not isinstance(raw_slots, (list, tuple)):
        raise TypeError("heartbeat tt_memory_slots must be a sequence")
    for index, raw_slot in enumerate(raw_slots):
        if not isinstance(raw_slot, Mapping):
            raise TypeError("heartbeat memory slot must be a mapping")
        metadata = {
            str(key): deepcopy(value)
            for key, value in raw_slot.items()
            if key not in {"key", "value"}
        }
        for tensor_name in ("key", "value"):
            value = raw_slot.get(tensor_name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"heartbeat memory slot {index} has no tensor {tensor_name}"
                )
            tensors[
                f"heartbeat.tt_memory_slots.{index}.{tensor_name}"
            ] = value.detach().cpu().clone()
        runtime_slots.append(metadata)
    runtime["tt_memory_slots"] = runtime_slots
    return tensors, runtime


def _save_bundle(bundle: OrganBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {
        "manifest": asdict(bundle.manifest),
        "tensors": bundle.tensors,
        "donor_config": bundle.donor_config,
        "runtime_state": bundle.runtime_state,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def extract_first_slice_bundle(
    checkpoint_path: Path,
    output_path: Path,
    *,
    source_checkpoint_sha256: str,
    gaba_layers: tuple[int, ...] = (0,),
    extra_organs: tuple[str, ...] = (),
    include_heartbeat: bool = False,
) -> OrganBundleManifest:
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    expected_hash = source_checkpoint_sha256.lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise ValueError("source checkpoint sha256 must be 64 lowercase hex characters")
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "source checkpoint hash mismatch: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    if not gaba_layers or any(layer < 0 for layer in gaba_layers):
        raise ValueError("gaba_layers must contain non-negative layer indices")
    if len(set(gaba_layers)) != len(gaba_layers):
        raise ValueError("gaba_layers must be unique")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("source checkpoint has no model_state_dict")
    state = _normalize_state(raw_state)
    selected = {
        key: tensor.detach().to(device="cpu").clone()
        for key, tensor in state.items()
        if _selected_key(key, gaba_layers, extra_organs)
    }
    heartbeat_runtime = None
    if include_heartbeat:
        heartbeat_tensors, heartbeat_runtime = _extract_heartbeat_state(
            payload.get("heartbeat_state")
        )
        selected.update(heartbeat_tensors)
    if not any(key.startswith("jepa_predictor.") for key in selected):
        raise ValueError("source checkpoint has no jepa organ tensors")
    for layer in gaba_layers:
        prefix = f"blocks.{layer}.gaba."
        if not any(key.startswith(prefix) for key in selected):
            raise ValueError(f"source checkpoint has no gaba organ at layer {layer}")

    records = tuple(
        TensorRecord(
            key=key,
            shape=tuple(int(value) for value in selected[key].shape),
            dtype=str(selected[key].dtype),
            sha256=tensor_sha256(selected[key]),
        )
        for key in sorted(selected)
    )
    organ_names = ["jepa", *(f"gaba.{layer}" for layer in gaba_layers)]
    for prefix in extra_organs:
        organ_name = _EXTRA_ORGAN_NAMES.get(prefix)
        if (
            organ_name is not None
            and organ_name not in organ_names
            and any(key.startswith(prefix) for key in selected)
        ):
            organ_names.append(organ_name)
    if include_heartbeat:
        organ_names.extend(("ttm", "heartbeat"))
    organs = tuple(organ_names)
    manifest = OrganBundleManifest(
        schema_version=_BUNDLE_SCHEMA_VERSION,
        source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=actual_hash,
        base_checkpoint_id=str(payload.get("base_checkpoint_id", "")),
        organs=organs,
        tensors=records,
    )
    raw_config = payload.get("config")
    donor_config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
    runtime_state = {
        key: payload[key]
        for key in _RUNTIME_KEYS
        if key in payload
    }
    if heartbeat_runtime is not None:
        runtime_state["heartbeat_state"] = heartbeat_runtime
    bundle = OrganBundle(
        manifest=manifest,
        tensors=selected,
        donor_config=donor_config,
        runtime_state=runtime_state,
    )
    _save_bundle(bundle, output_path)
    restored = load_organ_bundle(output_path)
    problems = restored.verify_tensor_hashes()
    if problems:
        raise ValueError(f"saved organ bundle failed verification: {problems}")
    return manifest


def load_organ_bundle(path: Path) -> OrganBundle:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("organ bundle payload must be a mapping")
    raw_manifest = payload.get("manifest")
    raw_tensors = payload.get("tensors")
    if not isinstance(raw_manifest, Mapping) or not isinstance(raw_tensors, Mapping):
        raise ValueError("organ bundle is missing manifest or tensors")
    manifest = _manifest_from_payload(raw_manifest)
    if manifest.schema_version != _BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported organ bundle schema: {manifest.schema_version}"
        )
    tensors = {
        str(key): value
        for key, value in raw_tensors.items()
        if isinstance(value, torch.Tensor)
    }
    donor_config = payload.get("donor_config")
    runtime_state = payload.get("runtime_state")
    bundle = OrganBundle(
        manifest=manifest,
        tensors=tensors,
        donor_config=(
            dict(donor_config) if isinstance(donor_config, Mapping) else {}
        ),
        runtime_state=(
            dict(runtime_state) if isinstance(runtime_state, Mapping) else {}
        ),
    )
    problems = bundle.verify_tensor_hashes()
    if problems:
        raise ValueError(f"organ bundle tensor hash mismatch: {problems}")
    return bundle
