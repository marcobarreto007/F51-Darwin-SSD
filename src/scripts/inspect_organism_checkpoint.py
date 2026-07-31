#!/usr/bin/env python3
"""Inspect an organism checkpoint without materializing its tensor storage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel, _MUTATIONAL_STATE_SUFFIXES
from f51_darwin.state_identity import backbone_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--verify-identity",
        action="store_true",
        help="Recompute the full tensor identity (reads the entire checkpoint).",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_state = {
        key.replace("_orig_mod.", ""): value
        for key, value in payload.get("model_state_dict", {}).items()
    }
    embedded_raw = payload.get("config")
    file_raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(embedded_raw, dict) or not isinstance(file_raw, dict):
        raise ValueError("checkpoint and YAML must contain mapping configs")
    embedded_config = DarwinXConfig.from_mapping(embedded_raw)
    file_config = DarwinXConfig.from_mapping(file_raw)
    embedded_normalized = dict(embedded_config.__dict__)
    file_normalized = dict(file_config.__dict__)
    config_diff = {
        key: {
            "checkpoint": embedded_normalized.get(key),
            "file": file_normalized.get(key),
        }
        for key in sorted(set(embedded_normalized) | set(file_normalized))
        if embedded_normalized.get(key) != file_normalized.get(key)
    }

    topology_manifest = payload.get("topology_manifest")
    topology_restore_error: str | None = None
    topology_repairs = 0
    with torch.device("meta"):
        expected_model = DarwinXModel(file_config)
        if isinstance(topology_manifest, dict):
            try:
                topology_repairs = DarwinXModel.restore_topology(
                    topology_manifest, expected_model
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                topology_restore_error = str(exc)
    expected_state = expected_model.state_dict()
    expected_parameters = list(expected_model.named_parameters())
    missing = sorted(set(expected_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(expected_state))
    shape_mismatches = [
        {
            "key": key,
            "checkpoint": list(checkpoint_state[key].shape),
            "expected": list(expected_state[key].shape),
        }
        for key in sorted(set(expected_state) & set(checkpoint_state))
        if tuple(checkpoint_state[key].shape) != tuple(expected_state[key].shape)
    ]
    optimizer_state = payload.get("optimizer_state_dict") or {}
    optimizer_groups = optimizer_state.get("param_groups") or []
    optimizer_slots = optimizer_state.get("state") or {}
    saved_param_ids = [
        parameter_id
        for group in optimizer_groups
        for parameter_id in group.get("params", [])
    ]
    saved_shapes = [
        tuple(optimizer_slots[parameter_id]["exp_avg"].shape)
        for parameter_id in saved_param_ids
        if parameter_id in optimizer_slots and "exp_avg" in optimizer_slots[parameter_id]
    ]
    current_shapes = [tuple(parameter.shape) for _, parameter in expected_parameters]
    inserted_current_parameters: list[str] = []
    saved_index = 0
    current_index = 0
    while saved_index < len(saved_shapes) and current_index < len(current_shapes):
        if saved_shapes[saved_index] == current_shapes[current_index]:
            saved_index += 1
            current_index += 1
        else:
            inserted_current_parameters.append(expected_parameters[current_index][0])
            current_index += 1
    inserted_current_parameters.extend(
        name for name, _ in expected_parameters[current_index:]
    )
    optimizer_type = str(payload.get("optimizer_type") or "")
    optimizer_resume_compatible = (
        "adamw" in optimizer_type.lower()
        and len(saved_param_ids) == len(expected_parameters)
        and saved_index == len(saved_shapes)
    )
    topology_manifest_valid = (
        isinstance(topology_manifest, dict) and topology_restore_error is None
    )
    # Known-migration keys: mutational state buffers that are initialized
    # via migrate_mutational_state_for_load() when absent from the checkpoint.
    def _is_mutational_migration_key(key: str) -> bool:
        return any(key.endswith("." + suffix) for suffix in _MUTATIONAL_STATE_SUFFIXES)

    missing_migration = sorted(k for k in missing if _is_mutational_migration_key(k))
    missing_strict = sorted(k for k in missing if not _is_mutational_migration_key(k))

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_version": int(payload.get("version", 0)),
        "model_name": embedded_config.model_name,
        "embedded_config_matches_file": embedded_config == file_config,
        "config_diff": config_diff,
        "training_state": payload.get("training_state", {}),
        "optimizer_type": payload.get("optimizer_type"),
        "optimizer_compatibility": {
            "saved_parameters": len(saved_param_ids),
            "saved_momentum_shapes": len(saved_shapes),
            "current_parameters": len(expected_parameters),
            "old_shapes_are_ordered_subsequence": saved_index == len(saved_shapes),
            "inserted_current_parameters": inserted_current_parameters,
        },
        "optimizer_resume_compatible": optimizer_resume_compatible,
        "topology_manifest_valid": topology_manifest_valid,
        "topology_restore_error": topology_restore_error,
        "topology_repairs": topology_repairs,
        "base_checkpoint_id": payload.get("base_checkpoint_id"),
        "checkpoint_tensor_entries": len(checkpoint_state),
        "checkpoint_state_numel": sum(value.numel() for value in checkpoint_state.values()),
        "expected_state_numel": sum(value.numel() for value in expected_state.values()),
        "missing_keys": missing,
        "missing_migration_keys": missing_migration,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "resume_shape_compatible": not missing_strict and not unexpected and not shape_mismatches,
    }
    if args.verify_identity:
        raw_identity = backbone_identity(checkpoint_state, embedded_raw)
        report["identity"] = {
            "declared": payload.get("base_checkpoint_id"),
            "with_raw_embedded_config": raw_identity,
            "with_normalized_embedded_config": backbone_identity(
                checkpoint_state, embedded_config
            ),
            "with_current_file_config": backbone_identity(
                checkpoint_state, file_config
            ),
        }
        report["identity_verified"] = (
            report["identity"]["declared"] == raw_identity
        )
    else:
        report["identity_verified"] = None
    report["strict_resume_compatible"] = bool(
        report["checkpoint_version"] >= 7
        and report["embedded_config_matches_file"]
        and report["resume_shape_compatible"]
        and report["topology_manifest_valid"]
        and report["optimizer_resume_compatible"]
        and (not args.verify_identity or report["identity_verified"])
    )
    print(json.dumps(report, indent=2))
    return 0 if report["strict_resume_compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
