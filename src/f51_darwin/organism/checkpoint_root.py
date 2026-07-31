from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from f51_darwin.artifact_manifest import write_json_atomic
from f51_darwin.darwin_x import DarwinXConfig


LINEAGE_ROOT_FILENAME = "lineage_root.json"


_COGNITION_CONFIG_FIELDS = frozenset(
    {
        "cognitive_architecture_version",
        "cognitive_shadow_enabled",
        "cognitive_pulse_enabled",
        "cognitive_organ_width",
        "cognitive_residual_max_scale",
    }
)


def _identity_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if result.get("cognitive_architecture_version", "disabled") == "disabled":
        for key in _COGNITION_CONFIG_FIELDS:
            result.pop(key, None)
    return result


def normalized_model_config(raw: Mapping[str, object] | DarwinXConfig) -> dict[str, Any]:
    if isinstance(raw, DarwinXConfig):
        return dict(raw.__dict__)
    mapping = dict(raw) if isinstance(raw, Mapping) else dict(vars(raw))
    try:
        return dict(DarwinXConfig.from_mapping(mapping).__dict__)
    except (KeyError, TypeError, ValueError):
        # Lightweight unit-test and diagnostic models may expose only an
        # identity-bearing subset instead of a full DarwinXConfig.
        return mapping


# Fields that CAN change between resume cycles without breaking architecture.
# Everything NOT in this set is considered "structural" — changing it would
# cause shape mismatch, parameter count change, or state_dict key divergence.
_OPERATIONAL_CONFIG_FIELDS: frozenset[str] = frozenset({
    # ── Organ / feature toggles (behavioral only, no new modules) ──
    "ghost_enabled",
    "heartbeat_enabled",
    "nitro_enabled",
    "dae_enabled",
    "dae_shadow_mode",
    "spider_calibration_enabled",
    "sleep_enabled",
    "decision_engine_enabled",
    "unified_mesh_enabled",
    # ── Loss weights — tuneable per cycle ──
    "mtp_weight",
    "jepa_weight",
    "ghost_weight",
    "curiosity_weight",
    "aux_loss_scale",
    "aux_loss_adaptive",
    # ── Heartbeat intervals / thresholds ──
    "heartbeat_think_interval",
    "heartbeat_explore_interval",
    "heartbeat_memory_capacity",
    "heartbeat_surprise_threshold",
    # NOTE: heartbeat_ff_layers is intentionally EXCLUDED — it controls
    # ForwardForwardStack layer count, which changes state_dict keys.
    # ── Calibration weights ──
    "spider_calibration_weight",
    "ttm_residual_max_scale",
    "ttm_associative_weight",
    # ── Nitro — capacity tuning ──
    "nitro_gpu_expert_capacity",
    # ── Blockchain & Senate — infrastructure / loss-weight tuning only ──
    "blockchain_enabled",
    "blockchain_path",
    "senate_enabled",
    "senate_interval",
    "senate_min_weight",
    "senate_death_threshold",
    "senate_rising_bonus",
    "senate_newborn_period",
    # ── Hyperparameters / tuning knobs ──
    "dropout",
    "rope_base_train",
    "rope_base_infer",
    "residual_scale_multiplier",
    "ghost_mask_ratio",
    "loss_semantics_version",
    "scan_chunk_size",
    "gradient_checkpointing",
    "vertical_routing_scale",
    "legacy_tiers",
    # ── DAE optimizer bounds ──
    "dae_update_gain_min",
    "dae_update_gain_max",
    # ── Non-architectural: inference context is a runtime hint ──
    "inference_context_length",
    # ── Operational paths — not part of model contract ──
    "checkpoint_root",
    "module_states",
    # ── Three-organ cognition operational toggles ──
    "cognitive_shadow_enabled",
    "cognitive_pulse_enabled",
    "cognitive_residual_max_scale",
})


def _strip_operational(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with operational fields removed."""
    return {k: v for k, v in config.items() if k not in _OPERATIONAL_CONFIG_FIELDS}


def model_config_identity(raw: Mapping[str, object] | DarwinXConfig) -> str:
    encoded = json.dumps(
        _identity_config(normalized_model_config(raw)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "darwin-config-v1:" + hashlib.sha256(encoded).hexdigest()


def structural_config_identity(raw: Mapping[str, object] | DarwinXConfig) -> str:
    """Config identity from architecture fields only — safe to match across
    resume cycles even when loss weights, intervals, or thresholds change."""
    normalized = _identity_config(normalized_model_config(raw))
    structural = _strip_operational(normalized)
    encoded = json.dumps(
        structural,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "darwin-structural-v1:" + hashlib.sha256(encoded).hexdigest()


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"checkpoint has no embedded mapping config: {checkpoint}")
    config = DarwinXConfig.from_mapping(raw_config)
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "checkpoint_version": int(payload.get("version", 0)),
        "model_name": config.model_name,
        "config_identity": model_config_identity(config),
        "structural_config_identity": structural_config_identity(config),
        "tokenizer_id": str(payload.get("tokenizer_id") or ""),
        "base_checkpoint_id": str(payload.get("base_checkpoint_id") or ""),
        "training_state": dict(payload.get("training_state") or {}),
    }


def _root_conflicts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    conflicts = sorted(root.glob("organism_cycle_*.pt"))
    conflicts.extend(sorted(root.glob("organism_cycle_*.pt.tmp")))
    for name in ("organism_latest.json", LINEAGE_ROOT_FILENAME):
        path = root / name
        if path.exists():
            conflicts.append(path)
    return conflicts


def read_lineage_root_identity(root: str | Path) -> dict[str, Any] | None:
    path = Path(root) / LINEAGE_ROOT_FILENAME
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid checkpoint root identity: {path}")
    return value


def write_lineage_root_identity(
    root: str | Path, identity: Mapping[str, object]
) -> Path:
    checkpoint_root = Path(root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    path = checkpoint_root / LINEAGE_ROOT_FILENAME
    payload = dict(identity)
    existing = read_lineage_root_identity(checkpoint_root)
    if existing is not None:
        # Allow updates to mutable fields: base_checkpoint_id evolves
        # with model weights every cycle.  structural_config_identity
        # is added during v1→v2 migration.  Immutable anchors
        # (model_name, config_identity, tokenizer_id, creation_mode)
        # must never change.
        # config_identity and base_checkpoint_id can legitimately change:
        # config_identity includes operational paths; base_checkpoint_id
        # evolves with model weights every cycle.  Both are rewound in
        # preflight_checkpoint_root when lineage_root gets ahead of the
        # checkpoint (partial save / crash).  structural_config_identity
        # is the true immutable architecture guard.
        immutable_keys = {
            "model_name", "tokenizer_id",
            "creation_mode", "created_at", "source_checkpoint_sha256",
        }
        for key in immutable_keys:
            if str(existing.get(key) or "") != str(payload.get(key) or ""):
                raise ValueError(
                    f"checkpoint root identity conflict for {key}: {path}"
                )
        if existing == payload:
            return path  # no-op
    write_json_atomic(path, payload)
    return path


def build_lineage_root_identity(
    *,
    model_config: Mapping[str, object] | DarwinXConfig,
    tokenizer_id: str,
    creation_mode: str,
    base_checkpoint_id: str,
    source_checkpoint_sha256: str | None = None,
    genesis_block_hash: str | None = None,
) -> dict[str, Any]:
    config = normalized_model_config(model_config)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "model_name": config["model_name"],
        "config_identity": model_config_identity(config),
        "structural_config_identity": structural_config_identity(config),
        "tokenizer_id": tokenizer_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "creation_mode": creation_mode,
        "base_checkpoint_id": base_checkpoint_id,
        # ── v9 blockchain anchor fields ──
        "genesis_block_hash": genesis_block_hash or "0" * 64,
        "checkpoint_chain": [],
    }
    if source_checkpoint_sha256:
        payload["source_checkpoint_sha256"] = source_checkpoint_sha256
    return payload


def preflight_checkpoint_root(
    *,
    root: str | Path,
    model_raw: Mapping[str, object] | DarwinXConfig,
    mode: str,
    resume: str | Path | None,
) -> dict[str, Any]:
    checkpoint_root = Path(root).resolve()
    requested_config_id = model_config_identity(model_raw)
    requested_model = normalized_model_config(model_raw)["model_name"]
    if mode == "fresh_start":
        conflicts = _root_conflicts(checkpoint_root)
        if conflicts:
            rendered = ", ".join(str(path) for path in conflicts[:5])
            raise ValueError(
                "fresh-start checkpoint root must be empty; "
                f"root={checkpoint_root}; conflicts={rendered}"
            )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        return {
            "root": str(checkpoint_root),
            "mode": mode,
            "model_name": requested_model,
            "config_identity": requested_config_id,
        }
    if mode != "resume" or resume is None:
        raise ValueError("checkpoint root preflight requires fresh_start or resume")
    # ── v9 blockchain integrity check ──
    _ledger_path = checkpoint_root / "causal_events.jsonl"
    if _ledger_path.exists():
        try:
            from f51_darwin.organism.causal_ledger import verify_ledger

            _verification = verify_ledger(_ledger_path)
            if not _verification.valid:
                raise ValueError(
                    "Blockchain integrity check failed: "
                    + "; ".join(_verification.errors)
                )
        except ImportError:
            pass  # causal ledger module not available
    metadata = checkpoint_metadata(resume)
    # Resume requires STRUCTURAL match only — weights, intervals, and thresholds
    # may change between cycles without breaking architecture compatibility.
    requested_structural = structural_config_identity(model_raw)
    if metadata["structural_config_identity"] != requested_structural:
        raise ValueError(
            "resume checkpoint architecture does not match requested model config: "
            f"{resume}"
        )
    identity = read_lineage_root_identity(checkpoint_root)
    if identity is None:
        raise ValueError(
            f"resume checkpoint root has no {LINEAGE_ROOT_FILENAME}: {checkpoint_root}"
        )
    for key in ("model_name", "config_identity", "tokenizer_id", "base_checkpoint_id"):
        if str(identity.get(key) or "") != str(metadata.get(key) or ""):
            if key in ("base_checkpoint_id", "config_identity"):
                # base_checkpoint_id evolves with model weights every cycle.
                # config_identity includes operational paths that may change
                # between runs.  If lineage_root got ahead of the checkpoint
                # (e.g. emergency save mutated lineage_root but crashed before
                # writing the .pt), rewind lineage_root to match the checkpoint
                # — the checkpoint is the source of truth for resume.
                identity[key] = metadata[key]
                write_lineage_root_identity(checkpoint_root, identity)
                continue
            raise ValueError(
                f"checkpoint root identity mismatch for {key}: {checkpoint_root}"
            )
    return metadata


def assert_new_checkpoint_target(root: str | Path, cycle: int) -> Path:
    checkpoint_root = Path(root)
    target = checkpoint_root / f"organism_cycle_{cycle:03d}.pt"
    temporary = target.with_suffix(target.suffix + ".tmp")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {target}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete checkpoint: {temporary}")
    return target


def ensure_lineage_root_identity(
    *,
    root: str | Path,
    model_config: Mapping[str, object] | DarwinXConfig,
    tokenizer_id: str,
    creation_mode: str,
    base_checkpoint_id: str,
) -> dict[str, Any]:
    checkpoint_root = Path(root)
    existing = read_lineage_root_identity(checkpoint_root)
    if existing is None:
        identity = build_lineage_root_identity(
            model_config=model_config,
            tokenizer_id=tokenizer_id,
            creation_mode=creation_mode,
            base_checkpoint_id=base_checkpoint_id,
        )
        write_lineage_root_identity(checkpoint_root, identity)
        return identity
    expected = {
        "model_name": normalized_model_config(model_config)["model_name"],
        "config_identity": model_config_identity(model_config),
        "structural_config_identity": structural_config_identity(model_config),
        "tokenizer_id": tokenizer_id,
    }
    for key, value in expected.items():
        stored = str(existing.get(key) or "")
        if key == "config_identity":
            # config_identity is the FULL hash (includes weights).  Resume is
            # validated via structural_config_identity instead.  v1→v2
            # migration: old lineage_roots lack the structural field — skip
            # the full-hash comparison unconditionally so weight/threshold
            # changes between cycles don't break the identity check.
            continue
        if key == "structural_config_identity":
            # structural_config_identity can change when _OPERATIONAL_CONFIG_FIELDS
            # is expanded between code versions (new fields classified as
            # non-structural).  Treat it as mutable like base_checkpoint_id —
            # the real architecture guard is preflight_checkpoint_root at resume.
            if stored != str(value or ""):
                existing["structural_config_identity"] = str(value or "")
                write_lineage_root_identity(checkpoint_root, existing)
            continue
        if stored != str(value or ""):
            raise ValueError(f"checkpoint root identity mismatch for {key}: {checkpoint_root}")
    # base_checkpoint_id evolves with model weights — update it so
    # preflight_checkpoint_root can validate future resumes against the
    # current head identity, not just the fresh-start snapshot.
    existing_ckpt_id = str(existing.get("base_checkpoint_id") or "")
    if existing_ckpt_id != str(base_checkpoint_id):
        existing["base_checkpoint_id"] = str(base_checkpoint_id)
        write_lineage_root_identity(checkpoint_root, existing)
    # v1→v2 migration  /  _OPERATIONAL_CONFIG_FIELDS expansion:
    # structural_config_identity may be absent (old lineage_root) or stale
    # (new code classifies more fields as operational).  Persist the current
    # computed identity in both cases — the real architecture guard is
    # preflight_checkpoint_root at resume time.
    _stored_structural = str(existing.get("structural_config_identity") or "")
    if _stored_structural != str(expected["structural_config_identity"]):
        existing["structural_config_identity"] = str(
            expected["structural_config_identity"]
        )
        existing["schema_version"] = 2
        write_lineage_root_identity(checkpoint_root, existing)
    # v2→v9: ensure blockchain anchor fields are present
    _needs_write = False
    if "genesis_block_hash" not in existing:
        existing["genesis_block_hash"] = "0" * 64
        _needs_write = True
    if "checkpoint_chain" not in existing:
        existing["checkpoint_chain"] = []
        _needs_write = True
    if _needs_write:
        write_lineage_root_identity(checkpoint_root, existing)
    return existing
