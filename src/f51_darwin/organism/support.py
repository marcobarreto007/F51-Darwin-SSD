from __future__ import annotations

from .dependencies import *  # noqa: F403

def replay_step_due(step: int, ratio: float) -> bool:
    """Deterministic scheduler with an exact long-run replay fraction."""
    if step <= 0 or not 0.0 < ratio < 1.0:
        return False
    return int(step * ratio) > int((step - 1) * ratio)


def metric_channel_lm_loss(channel: object, *, default: float | None) -> float | None:
    """Return the numeric LM loss exposed by an observability channel."""
    snapshot = getattr(channel, "snapshot", lambda: {})()
    if not isinstance(snapshot, dict):
        return default
    value = snapshot.get("lm_loss")
    if value is None:
        return default
    return float(value)


def advance_ashes_streak(previous_streak: int, entering_ashes: bool) -> int:
    return 1 if entering_ashes else previous_streak + 1


def token_source_state(path: str | Path | None, token_count: int) -> dict[str, object]:
    if path is None:
        return {"path": None, "token_count": int(token_count), "exists": False}
    token_path = Path(path).resolve()
    if not token_path.exists():
        return {"path": str(token_path), "token_count": int(token_count), "exists": False}
    stat = token_path.stat()
    return {
        "path": str(token_path),
        "token_count": int(token_count),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "int32_aligned": stat.st_size % 4 == 0,
        "exists": True,
    }


def canary_token_source_identity(path: str | Path, token_count: int) -> dict[str, object]:
    """Load and validate the durable identity declared beside the token bin."""
    token_path = Path(path).resolve()
    identity = token_source_state(token_path, token_count)
    manifest_path = Path(str(token_path) + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"token identity manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if int(manifest.get("tokens", -1)) != int(token_count):
        raise RuntimeError("token manifest count disagrees with loaded source")
    if int(manifest.get("bytes", -1)) != int(identity.get("size_bytes", -2)):
        raise RuntimeError("token manifest size disagrees with loaded source")
    if manifest.get("dtype") != "int32" or not bool(manifest.get("little_endian")):
        raise RuntimeError("token manifest must declare little-endian int32")
    declared_sha256 = str(manifest.get("sha256", "")).strip()
    if len(declared_sha256) != 64:
        raise RuntimeError("token manifest SHA-256 is missing or invalid")
    return {
        **identity,
        "manifest_path": str(manifest_path),
        "manifest_sha256": declared_sha256,
        "dtype": manifest["dtype"],
        "little_endian": True,
    }


def build_training_data_contract(
    *,
    sampler_mode: str,
    sampler_version: int,
    base_seed: int,
    block_size: int,
    batch_size: int,
    source_identity: Mapping[str, object],
    train_token_count: int,
    holdout_definition: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the durable data-order contract persisted in every new checkpoint."""

    mode = str(sampler_mode).strip().lower()
    if mode not in {"sequential", "permuted_blocks"}:
        raise ValueError("unsupported sampler mode")
    if isinstance(sampler_version, bool) or sampler_version != 1:
        raise ValueError("unsupported sampler version")
    if block_size < 2 or batch_size < 1 or train_token_count <= block_size:
        raise ValueError("invalid training data geometry")
    stable_source = {
        key: value
        for key, value in dict(source_identity).items()
        if key != "mtime_ns"
    }
    return {
        "schema": "darwin-training-data-v1",
        "sampler": {
            "mode": mode,
            "version": int(sampler_version),
            "base_seed": int(base_seed),
            "seed_policy": (
                "fixed"
                if mode == "permuted_blocks"
                else "base_plus_cycle"
            ),
        },
        "batch_size": int(batch_size),
        "block_size": int(block_size),
        "source": stable_source,
        "train_token_count": int(train_token_count),
        "holdout": (
            None
            if holdout_definition is None
            else dict(holdout_definition)
        ),
    }


def validate_training_data_contract(
    saved: Mapping[str, object] | None,
    current: Mapping[str, object],
    *,
    is_resume: bool = True,
) -> None:
    """Reject resume when data order, split, source, or geometry changed."""

    if not is_resume:
        if saved is not None:
            raise ValueError("fresh-start cannot supply a saved data contract")
        return
    if saved is None:
        sampler = dict(current.get("sampler") or {})
        if (
            sampler.get("mode") == "sequential"
            and current.get("holdout") is None
        ):
            return
        raise ValueError(
            "checkpoint has no training_data_contract; migration to a shuffled "
            "sampler or holdout requires a fresh checkpoint root"
        )
    if dict(saved) != dict(current):
        changed = sorted(
            key
            for key in set(saved) | set(current)
            if saved.get(key) != current.get(key)
        )
        raise ValueError(
            "checkpoint training_data_contract mismatch: " + ", ".join(changed)
        )


def rng_state_dict() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        # Saved states are per-device and count-specific to the hardware that
        # produced the checkpoint. Resuming on a host with a different visible
        # device count (e.g. moving from a 2-GPU box to a 1-GPU cloud
        # instance) must not crash -- RNG continuity is a reproducibility
        # nicety, not part of the training_data_contract, so we restore
        # whatever devices we actually have and skip the rest.
        saved_states = list(state["torch_cuda"])
        available = torch.cuda.device_count()
        for device_index, device_state in enumerate(saved_states[:available]):
            torch.cuda.set_rng_state(device_state, device_index)


def small_file_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    import hashlib

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
    }


def load_online_learning_state(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a bounded adapter payload for checkpoint embedding."""
    if not path.exists():
        return None, "state_file_missing"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return None, f"state_load_failed:{type(exc).__name__}"
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None, "unsupported_state_version"
    return payload, None


def retired_expert_ids(pool: ExpertPool, legacy: LegacyLayers) -> list[str]:
    """Find reusable real experts from persistent state, not transient streaks."""
    return sorted(
        expert_id
        for expert_id in legacy.layers[LayerTier.ASHES].experts
        if expert_id in pool.records
        and pool.records[expert_id].state in {ModuleState.DEAD, ModuleState.QUARANTINE}
        and pool.records[expert_id]._real_expert is not None
    )


__all__ = [name for name in globals() if not name.startswith('__')]
