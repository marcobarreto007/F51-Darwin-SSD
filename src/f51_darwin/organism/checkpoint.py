"""Single checkpoint authority for Darwin training and organism state.

All public compatibility imports in :mod:`f51_darwin.checkpointing` delegate
here. Model loads are strict after explicit schema migration.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from f51_darwin.artifacts import inspect_checkpoint_file
from f51_darwin.config import DarwinConfig
from f51_darwin.state_identity import (
    causal_cognitive_required,
    causal_cognitive_state_identity,
    training_contract_identity,
)
from .dependencies import *  # noqa: F403

EMPTY_CAUSAL_LEDGER_HEAD = "0" * 64
_CAUSAL_CHECKPOINT_SCHEMA = "darwin-causal-checkpoint-v1"
_CAUSAL_MODE_TO_ARM = {
    "disabled": None,
    "control": "CONTROL",
    "shadow": "SHADOW",
    "enforce": "APPLY",
}


def _normalized_causal_mode(causal_mode: str) -> str:
    mode = str(causal_mode).strip().lower()
    if mode not in _CAUSAL_MODE_TO_ARM:
        raise ValueError(
            "causal_mode must be one of: "
            + ", ".join(sorted(_CAUSAL_MODE_TO_ARM))
        )
    return mode


def _validated_ledger_head(ledger_head: str) -> str:
    if (
        not isinstance(ledger_head, str)
        or len(ledger_head) != 64
        or any(char not in "0123456789abcdef" for char in ledger_head)
    ):
        raise ValueError("ledger_head must be a lowercase SHA-256 digest")
    return ledger_head


def causal_v8_required(
    *, loss_semantics_version: int, causal_mode: str
) -> bool:
    """Return whether runtime semantics require the causal v8 envelope."""

    mode = _normalized_causal_mode(causal_mode)
    if isinstance(loss_semantics_version, bool) or not isinstance(
        loss_semantics_version, int
    ):
        raise TypeError("loss_semantics_version must be an integer")
    return loss_semantics_version != 1 or mode != "disabled"


def checkpoint_version_for_runtime(
    *,
    loss_semantics_version: int,
    causal_mode: str,
    causal_v8_migration: bool,
) -> int:
    """Return the checkpoint version for the active runtime semantics.

    v7  — legacy loss-v1 with causal bus disabled
    v9  — causal bus enabled or loss semantics v2 (blockchain-aware checkpoint)
    """

    requires_v8 = causal_v8_required(
        loss_semantics_version=loss_semantics_version,
        causal_mode=causal_mode,
    )
    if requires_v8 and causal_v8_migration is not True:
        raise ValueError(
            "causal_v8_migration=true is required before enabling the "
            "causal bus or loss semantics v2"
        )
    return 9 if requires_v8 else 7


def build_v8_causal_contract(
    *,
    loss_semantics_version: int,
    causal_mode: str,
    ledger_head: str,
) -> dict[str, Any]:
    """Build the exact causal state persisted inside a v8 checkpoint."""

    mode = _normalized_causal_mode(causal_mode)
    if not causal_v8_required(
        loss_semantics_version=loss_semantics_version,
        causal_mode=mode,
    ):
        raise ValueError("legacy loss-v1 with disabled bus must remain checkpoint v7")
    head = _validated_ledger_head(ledger_head)
    return {
        "schema": _CAUSAL_CHECKPOINT_SCHEMA,
        "mode": mode,
        "state": {
            "bus_enabled": mode != "disabled",
            "ablation_arm": _CAUSAL_MODE_TO_ARM[mode],
        },
        "ledger_head": head,
    }


def validate_v8_causal_contract(
    payload: dict[str, Any],
    *,
    loss_semantics_version: int,
    causal_mode: str,
    causal_v8_migration: bool,
    ledger_head: str,
) -> dict[str, Any]:
    """Strictly bind a v8 checkpoint to runtime semantics and ledger state."""

    if int(payload.get("version", 0)) not in (8, 9):
        raise ValueError("causal checkpoint validation requires version 8 or 9")
    checkpoint_version_for_runtime(
        loss_semantics_version=loss_semantics_version,
        causal_mode=causal_mode,
        causal_v8_migration=causal_v8_migration,
    )
    expected_training_contract_id = training_contract_identity()
    if payload.get("training_contract_id") != expected_training_contract_id:
        raise ValueError("v8 checkpoint training_contract_id mismatch")
    saved_loss_semantics = payload.get("loss_semantics_version")
    if (
        isinstance(saved_loss_semantics, bool)
        or not isinstance(saved_loss_semantics, int)
        or saved_loss_semantics != loss_semantics_version
    ):
        raise ValueError("v8 checkpoint loss_semantics_version mismatch")
    embedded_config = payload.get("config")
    if isinstance(embedded_config, dict) and causal_cognitive_required(
        embedded_config
    ):
        declared_cognitive_state_id = payload.get(
            "causal_cognitive_state_id"
        )
        try:
            recomputed_cognitive_state_id = causal_cognitive_state_identity(
                payload.get("model_state_dict") or {},
                payload.get("heartbeat_state"),
                embedded_config,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "v8 checkpoint causal_cognitive_state_id cannot be verified"
            ) from exc
        if declared_cognitive_state_id != recomputed_cognitive_state_id:
            raise ValueError(
                "v8 checkpoint causal_cognitive_state_id mismatch"
            )
    elif payload.get("causal_cognitive_state_id") is not None:
        raise ValueError(
            "v8 checkpoint causal_cognitive_state_id cannot be verified "
            "without config"
        )

    contract = payload.get("causal_contract")
    if not isinstance(contract, dict):
        raise ValueError("v8 checkpoint is missing causal_contract")
    expected_fields = {"schema", "mode", "state", "ledger_head"}
    if set(contract) != expected_fields:
        raise ValueError("v8 checkpoint causal_contract fields mismatch")
    if contract.get("schema") != _CAUSAL_CHECKPOINT_SCHEMA:
        raise ValueError("v8 checkpoint causal_contract schema mismatch")

    expected = build_v8_causal_contract(
        loss_semantics_version=loss_semantics_version,
        causal_mode=causal_mode,
        ledger_head=_validated_ledger_head(ledger_head),
    )
    if contract.get("mode") != expected["mode"]:
        raise ValueError("v8 checkpoint causal mode mismatch")
    if contract.get("state") != expected["state"]:
        raise ValueError("v8 checkpoint causal state mismatch")
    if contract.get("ledger_head") != expected["ledger_head"]:
        # If the current ledger is empty (fresh chain after _rewind_ledger
        # archived the old session), the checkpoint's ledger_head belongs to
        # the archived session — skip the comparison.
        if ledger_head == EMPTY_CAUSAL_LEDGER_HEAD:
            return contract
        raise ValueError("v8 checkpoint ledger_head mismatch")
    return contract


def current_causal_ledger_head(ledger: Any | None) -> str:
    """Verify the local ledger and return the head used by checkpoint v8."""

    if ledger is None:
        return EMPTY_CAUSAL_LEDGER_HEAD
    verification = ledger.verify()
    if not verification.valid:
        raise ValueError(
            "causal ledger verification failed: "
            + ", ".join(verification.errors)
        )
    return _validated_ledger_head(verification.head_hash)


def _legacy_v6_missing_key_allowed(key: str) -> bool:
    return (
        ".moe.neuroendocrine." in key
        or key.endswith(".moe._expert_usage_buffer")
    )


def validate_legacy_v6_resume(
    payload: dict[str, Any],
    state: dict[str, Any],
    model: DarwinXModel,
    model_config: DarwinXConfig,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the one supported v6 -> v7 anatomy migration.

    v6 predates the Topology Manifest and the persistent neuroendocrine
    buffers.  It is accepted only when its embedded config is exact, every
    overlapping tensor shape matches, and all key differences belong to that
    known schema transition.  A legacy content id is preserved as provenance;
    it is not falsely presented as verified by the current hashing schema.
    """
    if int(payload.get("version", 0)) != 6:
        raise ValueError("only checkpoint v6 has an explicit legacy migration")
    embedded_raw = payload.get("config")
    if not isinstance(embedded_raw, dict):
        raise ValueError("v6 checkpoint is missing its embedded model config")
    embedded_config = DarwinXConfig.from_mapping(embedded_raw)
    if embedded_config != model_config:
        checkpoint_values = dict(embedded_config.__dict__)
        current_values = dict(model_config.__dict__)
        changed = sorted(
            key
            for key in set(checkpoint_values) | set(current_values)
            if checkpoint_values.get(key) != current_values.get(key)
        )
        raise ValueError(
            "v6 checkpoint config does not match current config: "
            + ", ".join(changed)
        )

    declared = payload.get("base_checkpoint_id")
    digest = declared[len(_LEGACY_ID_PREFIX):] if isinstance(declared, str) and declared.startswith(_LEGACY_ID_PREFIX) else ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("v6 checkpoint has an invalid declared backbone identity")

    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(state))
    unexpected = sorted(set(state) - set(expected_state))
    shape_mismatches = sorted(
        key
        for key in set(expected_state) & set(state)
        if tuple(expected_state[key].shape) != tuple(state[key].shape)
    )
    forbidden_missing = [key for key in missing if not _legacy_v6_missing_key_allowed(key)]
    if forbidden_missing or unexpected or shape_mismatches:
        details = {
            "forbidden_missing": forbidden_missing,
            "unexpected": unexpected,
            "shape_mismatches": shape_mismatches,
        }
        raise ValueError(f"v6 checkpoint anatomy is incompatible: {details}")

    pointer_status = "not_checked"
    if checkpoint_path is not None:
        pointer_path = checkpoint_path.parent / "organism_latest.json"
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if pointer.get("path") == checkpoint_path.name:
                expected = {
                    "size_bytes": checkpoint_path.stat().st_size,
                    "base_checkpoint_id": declared,
                    "cycle": payload.get("training_state", {}).get("cycle"),
                    "step": payload.get("training_state", {}).get("step"),
                }
                mismatched = [key for key, value in expected.items() if pointer.get(key) != value]
                if mismatched:
                    raise ValueError(
                        "organism_latest.json disagrees with checkpoint: "
                        + ", ".join(mismatched)
                    )
                pointer_status = "matched"
            else:
                pointer_status = "points_elsewhere"
        else:
            pointer_status = "absent"

    recomputed = backbone_identity(state, embedded_config)
    return {
        "schema": "darwin-v6-to-v7-anatomy-migration-v1",
        "from_version": 6,
        "declared_base_checkpoint_id": declared,
        "current_schema_checkpoint_id": recomputed,
        "legacy_identity_recomputed": recomputed == declared,
        "config_exact": True,
        "missing_initialized_buffers": missing,
        "pointer_status": pointer_status,
        "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


def validate_v7_checkpoint_identity(
    payload: dict[str, Any], state: dict[str, torch.Tensor]
) -> str:
    """Validate the same-checkpoint v7 identity with its historical raw config.

    The config mapping is part of the identity bytes. Reconstructing it through
    today's dataclass would inject fields added after an older checkpoint was
    written and create a false mismatch. Runtime compatibility with today's
    normalized config is a separate gate at the call site.
    """

    if int(payload.get("version", 0)) < 7:
        raise ValueError("v7 checkpoint identity validation requires version >= 7")
    embedded_raw = payload.get("config")
    if not isinstance(embedded_raw, dict):
        raise ValueError("v7 checkpoint is missing its raw embedded config")
    declared = payload.get("base_checkpoint_id")
    if not declared:
        raise ValueError("v7 checkpoint is missing base_checkpoint_id")
    recomputed = backbone_identity(state, embedded_raw)
    if declared != recomputed:
        raise ValueError("checkpoint backbone identity does not match its tensors")
    return recomputed


def migrate_adamw_state_with_new_baselines(
    saved_state: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[str]:
    """Restore AdamW by order while initializing only new hormonal baselines.

    PyTorch optimizer checkpoints have no parameter names.  The v6 -> v7
    transition inserted exactly one ``baseline_dopamine`` parameter per layer;
    every legacy momentum tensor must otherwise remain an ordered shape match.
    """
    saved_groups = saved_state.get("param_groups") or []
    current_template = optimizer.state_dict()
    current_groups = current_template.get("param_groups") or []
    if len(saved_groups) != 1 or len(current_groups) != 1:
        raise ValueError("partial AdamW migration requires one parameter group")

    saved_ids = list(saved_groups[0].get("params", []))
    current_ids = list(current_groups[0].get("params", []))
    current_named = list(model.named_parameters())
    if len(current_named) != len(current_ids):
        raise ValueError("model parameter order does not match optimizer order")

    saved_slots = saved_state.get("state") or {}
    mapping: dict[int, int] = {}
    inserted: list[str] = []
    saved_index = 0
    for current_index, (name, parameter) in enumerate(current_named):
        if saved_index < len(saved_ids):
            saved_id = saved_ids[saved_index]
            slot = saved_slots.get(saved_id, {})
            exp_avg = slot.get("exp_avg")
            if isinstance(exp_avg, torch.Tensor) and tuple(exp_avg.shape) == tuple(parameter.shape):
                mapping[saved_id] = current_ids[current_index]
                saved_index += 1
                continue
        if not name.endswith(".moe.neuroendocrine.baseline_dopamine"):
            raise ValueError(f"unexpected new optimizer parameter: {name}")
        inserted.append(name)

    if saved_index != len(saved_ids):
        raise ValueError(
            f"only {saved_index}/{len(saved_ids)} legacy optimizer states aligned"
        )
    expected_insertions = len(current_ids) - len(saved_ids)
    if len(inserted) != expected_insertions or not inserted:
        raise ValueError("legacy optimizer insertion count is inconsistent")

    migrated_groups = [dict(saved_groups[0])]
    migrated_groups[0]["params"] = current_ids
    migrated_state = {
        "state": {
            mapping[saved_id]: slot
            for saved_id, slot in saved_slots.items()
            if saved_id in mapping
        },
        "param_groups": migrated_groups,
    }
    optimizer.load_state_dict(migrated_state)
    return inserted


def _tensors_to_cpu(obj: Any) -> Any:
    """Recursively detach+clone every tensor in a nested structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _tensors_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_tensors_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_tensors_to_cpu(value) for value in obj)
    return obj


def _optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Snapshot optimizer state_dict with every moment tensor detached on CPU.

    The training loop keeps mutating the live optimizer; the async checkpoint
    writer (R4) must receive tensors the next optimizer.step() cannot touch.
    """
    return _tensors_to_cpu(optimizer.state_dict())


@torch.no_grad()
def move_optimizer_state_for_parameters(
    optimizer: torch.optim.Optimizer,
    parameters,
) -> int:
    """Move materialized optimizer tensors to their parameter device.

    Nitro may move a parameter CPU -> CUDA after its optimizer moments were
    created.  PyTorch does not migrate those moments with ``module.to()``.
    Scalar counters are moved too when possible so the saved live state is
    internally consistent.
    """
    moved = 0
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state:
            continue
        for key, value in tuple(state.items()):
            if not torch.is_tensor(value):
                continue
            if value.device != parameter.device or not value.is_contiguous():
                state[key] = value.to(device=parameter.device).contiguous()
                moved += 1
    return moved


_ADAMW_STATE_KEYS: tuple[str, ...] = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def offload_optimizer_states_to_cpu(optimizer: torch.optim.Optimizer) -> int:
    """Move AdamW moment tensors to CPU to free GPU VRAM.

    After ``optimizer.step()`` the live moments (exp_avg, exp_avg_sq) are
    only needed for the NEXT weight update.  PCIe 4.0 x16 (~32 GB/s)
    transfers the full 1.2 GB state in ~37ms — negligible against a ~6s
    step time at 800 tok/s.

    ``move_optimizer_state_for_parameters`` restores states to GPU
    transparently before the next ``step()``.
    """
    moved = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state:
                continue
            for key in _ADAMW_STATE_KEYS:
                value = state.get(key)
                if value is None or value.device.type != "cuda":
                    continue
                state[key] = value.to(device="cpu")
                moved += 1
    if moved:
        # Per-device synchronize so a single faulty GPU does not hang the
        # whole process.  Each device is synced independently inside a
        # try/except so the error message names the failing GPU.
        for dev_idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(dev_idx):
                    torch.cuda.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"CUDA async error on GPU {dev_idx} "
                    f"({torch.cuda.get_device_name(dev_idx)}): {exc}. "
                    f"Re-run with CUDA_LAUNCH_BLOCKING=1 for a precise stack trace."
                ) from exc
    return moved


def assert_optimizer_device_invariants(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Fail before weight mutation when live optimizer tensors diverge."""
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and gradient.device != parameter.device:
            raise RuntimeError(
                "optimizer device invariant: "
                f"{name}.grad is {gradient.device}, parameter is {parameter.device}"
            )
        state = optimizer.state.get(parameter, {})
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            # CPU scalar counters (for example Adam's step) are accepted.
            if value.numel() == 1:
                continue
            if value.device != parameter.device:
                raise RuntimeError(
                    "optimizer device invariant: "
                    f"{name}.{key} is {value.device}, parameter is "
                    f"{parameter.device}"
                )


def remap_optimizer_state(
    old_optimizer: torch.optim.Optimizer,
    old_named_params: dict[str, torch.nn.Parameter],
    new_optimizer: torch.optim.Optimizer,
    new_named_params: dict[str, torch.nn.Parameter],
) -> dict[str, int]:
    """Carry AdamW moments across a topology change (prune/neurogenesis/expand).

    PyTorch optimizers key per-parameter state by object identity, so a freshly
    built optimizer starts empty.  The old moments are re-keyed by
    ``(name, shape)`` and reattached to every parameter that survives unchanged:

    * unchanged parameter -> moments preserved (exp_avg / exp_avg_sq / step)
    * new parameter      -> left empty, lazily initialised on first step
                            (neurogenesis / born experts)
    * missing parameter  -> discarded (prune / apoptosis)
    * reshaped parameter -> treated as new (expert expansion changes the shape)

    This generalises ``migrate_adamw_state_with_new_baselines`` to arbitrary
    topology edits, stopping the loss-spike caused by rebuilding AdamW cold
    after every structural action.
    """
    old_state_by_key: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for name, param in old_named_params.items():
        state = old_optimizer.state.get(param)
        if state:
            old_state_by_key[(name, tuple(param.shape))] = dict(state)

    preserved = 0
    for name, new_param in new_named_params.items():
        old_state = old_state_by_key.pop((name, tuple(new_param.shape)), None)
        if old_state is None:
            continue  # lazily initialised by the first optimizer.step()
        cloned: dict[str, Any] = {}
        for slot_name, value in old_state.items():
            if isinstance(value, torch.Tensor):
                cloned[slot_name] = value.detach().clone()
            else:
                cloned[slot_name] = value
        new_optimizer.state[new_param] = cloned
        preserved += 1

    return {
        "preserved": preserved,
        "initialised": len(new_named_params) - preserved,
        "dropped": len(old_state_by_key),
    }


def remap_optimizer_state_from_checkpoint(
    saved_opt_state: dict[str, Any],
    old_model_state: dict[str, torch.Tensor],
    new_optimizer: torch.optim.Optimizer,
    new_named_params: dict[str, torch.nn.Parameter],
) -> dict[str, int]:
    """Re-key AdamW moments from a checkpoint onto a freshly-built optimizer.

    Unlike :func:`remap_optimizer_state` this operates directly from the
    checkpoint payload (``optimizer_state_dict`` + ``model_state_dict``),
    without requiring a live old-optimizer object.  Parameters are matched
    by ``(name, shape)`` so insertions anywhere (beginning, middle, end)
    are handled correctly.

    * unchanged parameter -> moments preserved (exp_avg / exp_avg_sq / step)
    * new parameter        -> left empty, lazily initialised on first step
    * missing parameter    -> discarded
    * reshaped parameter   -> treated as new
    """
    old_param_ids = saved_opt_state.get("param_groups", [{}])[0].get("params", [])
    old_names = list(old_model_state.keys())
    if len(old_param_ids) != len(old_names):
        raise ValueError(
            f"old optimizer param count ({len(old_param_ids)}) != "
            f"old model state count ({len(old_names)})"
        )
    saved_state = saved_opt_state.get("state", {})
    old_by_key: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for idx, param_id in enumerate(old_param_ids):
        if param_id in saved_state:
            name = old_names[idx]
            shape = tuple(old_model_state[name].shape)
            old_by_key[(name, shape)] = dict(saved_state[param_id])

    preserved = 0
    for name, new_param in new_named_params.items():
        old_state = old_by_key.pop((name, tuple(new_param.shape)), None)
        if old_state is None:
            continue  # new or reshaped — lazy init on first optimizer.step()
        cloned: dict[str, Any] = {}
        for slot_name, value in old_state.items():
            cloned[slot_name] = (
                value.detach().clone() if isinstance(value, torch.Tensor) else value
            )
        new_optimizer.state[new_param] = cloned
        preserved += 1

    return {
        "preserved": preserved,
        "initialised": len(new_named_params) - preserved,
        "dropped": len(old_by_key),
    }


def _write_checkpoint_file(payload: dict[str, Any], path: Path) -> None:
    """Atomic no-replace checkpoint write: save to .tmp, validate, rename."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_acquired = False
    try:
        with lock_path.open("x", encoding="ascii") as lock_stream:
            lock_stream.write(f"pid={os.getpid()}\n")
        lock_acquired = True
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
        if temporary_path.exists():
            raise FileExistsError(
                f"refusing to overwrite incomplete checkpoint: {temporary_path}"
            )
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as stream:
            checkpoint_magic = stream.read(4)
        if checkpoint_magic != b"PK\x03\x04":
            raise RuntimeError("checkpoint validation failed before atomic publish")
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite checkpoint created during save: {path}"
            )
        os.rename(temporary_path, path)
    except BaseException:
        if lock_acquired:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)


def _publish_checkpoint_pointer(
    ckpt_dir: Path,
    path: Path,
    payload: dict[str, Any],
    cycle: int,
    step: int,
    base_checkpoint_id: str,
) -> None:
    """Atomically publish organism_latest.json pointing at the saved cycle."""
    pointer = ckpt_dir / "organism_latest.json"
    pointer_tmp = pointer.with_suffix(pointer.suffix + ".tmp")
    pointer_tmp.write_text(
        json.dumps(
            {
                "version": 1,
                "checkpoint_version": payload["version"],
                "path": path.name,
                "cycle": cycle,
                "step": step,
                "size_bytes": path.stat().st_size,
                "base_checkpoint_id": base_checkpoint_id,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(pointer_tmp, pointer)



def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: DarwinConfig,
    metrics: dict[str, Any] | None = None,
) -> Path:
    return save_training_checkpoint(path, model, config, metrics=metrics)


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: DarwinConfig,
    *,
    metrics: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "metrics": metrics or {},
        "lineage": "F51 Darwin-SSD random seed lineage",
    }
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, output, _use_new_zipfile_serialization=False)
    return output


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(path)
    health = inspect_checkpoint_file(checkpoint)
    if health.status not in {"torch_zip", "legacy_pickle_or_raw"}:
        raise ValueError(
            "Invalid checkpoint file "
            f"{checkpoint}: status={health.status}, error={health.error or 'none'}"
        )
    return torch.load(checkpoint, map_location=map_location, weights_only=False)


def load_training_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    if "model_state_dict" not in payload:
        raise ValueError("Invalid training checkpoint: missing model_state_dict.")
    return payload


def load_model_from_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple["DarwinXModel", "DarwinXConfig", dict[str, Any]]:
    target_device = torch.device(map_location)
    payload = load_checkpoint(path, map_location=target_device)
    raw = payload.get("config", {})
    if int(payload.get("version", 1)) == 1:
        from f51_darwin.model import F51DarwinModel

        config = DarwinConfig(**raw)
        model = F51DarwinModel(config)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model = model.to(target_device)
        return model, config, payload.get("metrics", {})

    from f51_darwin.darwin_x import (
        DarwinXConfig,
        DarwinXModel,
        migrate_mutational_state_for_load,
    )

    config = DarwinXConfig.from_mapping(raw)
    model = DarwinXModel(config)
    state = {
        key.replace("_orig_mod.", ""): value
        for key, value in payload["model_state_dict"].items()
    }
    migrate_mutational_state_for_load(state, model)
    model.load_state_dict(state, strict=True)
    model = model.to(target_device)
    metrics = payload.get("metrics", {})
    return model, config, metrics


__all__ = [name for name in globals() if not name.startswith("__")]
