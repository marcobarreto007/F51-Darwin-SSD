"""Run a paired, local-only causal ablation on a deterministic tiny model.

This harness never resolves a Darwin checkpoint, dataset, or training
entrypoint.  Its only purpose is to validate the CONTROL/SHADOW/APPLY contract
from identical in-memory model, optimizer, RNG, and batch anchors.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.organism.causal_adapters import (
    CausalTrainingExecutor,
    ExplicitRequestAdapter,
)
from f51_darwin.organism.causal_bus import (
    AblationArm,
    OrganCausalBus,
    Phase,
    StepIdentity,
)


SCHEMA_VERSION = 1
REFERENCE_ARM = "DISABLED"
REGISTERED_ARMS = ("CONTROL", "SHADOW", "APPLY")
DEFAULT_TRAINING_CONTRACT = "tiny-causal-ablation-v1"
DEFAULT_DATA_CURSOR = "tiny-batch-0"
_INTERVENTION_ID = "zero-head-gradient"
_ARTIFACT_PATHS = (
    "arms/disabled.json",
    "arms/control.json",
    "arms/shadow.json",
    "arms/apply.json",
    "verdict.json",
)


class AblationContractError(ValueError):
    """The paired registration does not share an exact experimental anchor."""


@dataclass(frozen=True)
class AblationAnchors:
    base_checkpoint_id: str
    data_cursor: str
    seed: int
    training_contract_id: str
    batch_digest: str
    model_digest: str
    optimizer_digest: str
    rng_digest: str


@dataclass(frozen=True)
class ArmRegistration:
    arm: str
    base_checkpoint_id: str
    data_cursor: str
    seed: int
    training_contract_id: str


@dataclass(frozen=True)
class AblationManifest:
    schema_version: int
    manifest_id: str
    reference_arm: str
    registered_arms: tuple[str, ...]
    anchors: AblationAnchors
    registrations: tuple[ArmRegistration, ...]
    intervention: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intervention",
            MappingProxyType(copy.deepcopy(dict(self.intervention))),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise AblationContractError("schema_version mismatch")
        if self.reference_arm != REFERENCE_ARM:
            raise AblationContractError("reference_arm mismatch")
        if self.registered_arms != REGISTERED_ARMS:
            raise AblationContractError("registered_arms mismatch")
        if tuple(item.arm for item in self.registrations) != REGISTERED_ARMS:
            raise AblationContractError("registration arm set mismatch")

        for registration in self.registrations:
            for field_name in (
                "base_checkpoint_id",
                "data_cursor",
                "seed",
                "training_contract_id",
            ):
                expected = getattr(self.anchors, field_name)
                actual = getattr(registration, field_name)
                if actual != expected:
                    raise AblationContractError(
                        f"{field_name} mismatch for {registration.arm}: "
                        f"expected {expected!r}, got {actual!r}"
                    )

        expected_id = _manifest_id(
            self.anchors, self.registrations, self.intervention
        )
        if self.manifest_id != expected_id:
            raise AblationContractError("manifest_id mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "reference_arm": self.reference_arm,
            "registered_arms": list(self.registered_arms),
            "anchors": asdict(self.anchors),
            "registrations": [asdict(item) for item in self.registrations],
            "intervention": dict(self.intervention),
        }


@dataclass
class _RngSnapshot:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


@dataclass
class _PreparedExperiment:
    manifest: AblationManifest
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    rng: _RngSnapshot
    inputs: torch.Tensor
    targets: torch.Tensor


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(3, 5)
        self.dropout = nn.Dropout(p=0.25)
        self.head = nn.Linear(5, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.trunk(inputs))
        return self.head(self.dropout(hidden))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported digest value: {type(value).__name__}")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parameter_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: _digest(parameter)
        for name, parameter in model.named_parameters()
    }


def _snapshot_rng() -> _RngSnapshot:
    return _RngSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state().clone(),
        torch_cuda=(
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    )


def _restore_rng(snapshot: _RngSnapshot) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.random.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda is not None:
        torch.cuda.set_rng_state_all(snapshot.torch_cuda)


def _rng_digest(snapshot: _RngSnapshot) -> str:
    return _digest(
        {
            "python": snapshot.python,
            "numpy": snapshot.numpy,
            "torch_cpu": snapshot.torch_cpu,
            "torch_cuda": snapshot.torch_cuda,
        }
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _intervention_request() -> dict[str, Any]:
    return {
        "intervention_id": _INTERVENTION_ID,
        "phase": Phase.PRE_OPTIMIZER.value,
        "target": "GRADIENT_GROUP",
        "operation": "SCALE",
        "subject": "head.weight",
        "value": 0.0,
        "valid_from_step": 0,
        "valid_through_step": 0,
    }


def _manifest_id(
    anchors: AblationAnchors,
    registrations: tuple[ArmRegistration, ...],
    intervention: Mapping[str, Any],
) -> str:
    digest = _digest(
        {
            "schema_version": SCHEMA_VERSION,
            "reference_arm": REFERENCE_ARM,
            "registered_arms": REGISTERED_ARMS,
            "anchors": asdict(anchors),
            "registrations": [asdict(item) for item in registrations],
            "intervention": dict(intervention),
        }
    )
    return f"causal-ablation-v1:{digest}"


def _prepare(
    *,
    seed: int,
    data_cursor: str,
    training_contract_id: str,
    registration_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> _PreparedExperiment:
    _seed_all(seed)
    model = _TinyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    inputs = torch.randn(4, 3)
    targets = torch.randn(4, 2)
    rng = _snapshot_rng()
    model_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    batch_digest = _digest({"inputs": inputs, "targets": targets})
    model_digest = _digest(model_state)
    optimizer_digest = _digest(optimizer_state)
    base_checkpoint_id = f"tiny-causal-v1:{model_digest}"
    anchors = AblationAnchors(
        base_checkpoint_id=base_checkpoint_id,
        data_cursor=data_cursor,
        seed=seed,
        training_contract_id=training_contract_id,
        batch_digest=batch_digest,
        model_digest=model_digest,
        optimizer_digest=optimizer_digest,
        rng_digest=_rng_digest(rng),
    )
    registrations = tuple(
        ArmRegistration(
            arm=arm,
            base_checkpoint_id=base_checkpoint_id,
            data_cursor=data_cursor,
            seed=seed,
            training_contract_id=training_contract_id,
        )
        for arm in REGISTERED_ARMS
    )

    overrides = dict(registration_overrides or {})
    unknown_arms = set(overrides) - set(REGISTERED_ARMS)
    if unknown_arms:
        raise AblationContractError(
            f"unknown registration arms: {sorted(unknown_arms)}"
        )
    registrations = tuple(
        replace(
            registration,
            **dict(overrides.get(registration.arm, {})),
        )
        for registration in registrations
    )
    intervention = _intervention_request()
    manifest = AblationManifest(
        schema_version=SCHEMA_VERSION,
        manifest_id=_manifest_id(anchors, registrations, intervention),
        reference_arm=REFERENCE_ARM,
        registered_arms=REGISTERED_ARMS,
        anchors=anchors,
        registrations=registrations,
        intervention=intervention,
    )
    return _PreparedExperiment(
        manifest=manifest,
        model_state=model_state,
        optimizer_state=optimizer_state,
        rng=rng,
        inputs=inputs.detach().clone(),
        targets=targets.detach().clone(),
    )


def _assert_pre_anchors(
    prepared: _PreparedExperiment,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    anchors = prepared.manifest.anchors
    actual = {
        "base_checkpoint_id": anchors.base_checkpoint_id,
        "data_cursor": anchors.data_cursor,
        "seed": anchors.seed,
        "training_contract_id": anchors.training_contract_id,
        "batch_digest": _digest({"inputs": inputs, "targets": targets}),
        "model_digest": _digest(model.state_dict()),
        "optimizer_digest": _digest(optimizer.state_dict()),
        "rng_digest": _rng_digest(_snapshot_rng()),
    }
    expected = asdict(anchors)
    if actual != expected:
        differences = sorted(
            key for key in expected if actual.get(key) != expected[key]
        )
        raise AblationContractError(
            f"pre-arm anchor restoration failed: {differences}"
        )
    return actual


def _run_arm(prepared: _PreparedExperiment, arm: str) -> dict[str, Any]:
    model = _TinyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.load_state_dict(copy.deepcopy(prepared.model_state))
    optimizer.load_state_dict(copy.deepcopy(prepared.optimizer_state))
    inputs = prepared.inputs.detach().clone()
    targets = prepared.targets.detach().clone()
    _restore_rng(prepared.rng)

    pre_anchors = _assert_pre_anchors(
        prepared, model, optimizer, inputs, targets
    )
    pre_parameters = _parameter_digests(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = F.mse_loss(predictions, targets)
    loss.backward()

    decision = None
    if arm != REFERENCE_ARM:
        bus_arm = AblationArm(arm)
        bus = OrganCausalBus(
            arm=bus_arm,
            adapters=(ExplicitRequestAdapter(),),
        )
        identity = StepIdentity(
            run_id="tiny-causal-ablation",
            cycle=0,
            optimizer_step=0,
            accumulation_window=0,
            batch_digest=prepared.manifest.anchors.batch_digest,
            rng_digest=prepared.manifest.anchors.rng_digest,
            base_checkpoint_id=prepared.manifest.anchors.base_checkpoint_id,
            training_contract_id=(
                prepared.manifest.anchors.training_contract_id
            ),
            ablation_plan_id=prepared.manifest.manifest_id,
            attempt_id="paired-attempt-0",
        )
        decision = bus.decide(
            identity,
            Phase.PRE_OPTIMIZER,
            {
                "losses": {"mse": float(loss.detach())},
                "optimizer_due": True,
                "requests": [dict(prepared.manifest.intervention)],
            },
        )
        if decision.blocked:
            reasons = [item.reason for item in decision.rejected]
            raise AblationContractError(
                f"{arm} decision blocked: {reasons}"
            )
        CausalTrainingExecutor(model).scale_gradients(decision)

    optimizer.step()
    probes = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(())),
    }
    proposed_ids = (
        [item.intervention_id for item in decision.accepted]
        if decision is not None
        else []
    )
    effective_ids = (
        [item.intervention_id for item in decision.effective]
        if decision is not None
        else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": prepared.manifest.manifest_id,
        "arm": arm,
        "pre_anchors": pre_anchors,
        "batch_digest": pre_anchors["batch_digest"],
        "loss": float(loss.detach()),
        "pre_parameter_digests": pre_parameters,
        "parameter_digests": _parameter_digests(model),
        "post_model_digest": _digest(model.state_dict()),
        "post_optimizer_digest": _digest(optimizer.state_dict()),
        "post_rng_digest": _rng_digest(_snapshot_rng()),
        "rng_probes": probes,
        "proposed_intervention_ids": proposed_ids,
        "effective_intervention_ids": effective_ids,
    }


def _build_verdict(
    manifest: AblationManifest,
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reference = evidence[REFERENCE_ARM]
    paired_noops = tuple(evidence[arm] for arm in ("CONTROL", "SHADOW"))
    anchors_restored = all(
        arm["pre_anchors"] == asdict(manifest.anchors)
        for arm in evidence.values()
    )
    disabled_control_shadow_bitwise = all(
        arm["post_model_digest"] == reference["post_model_digest"]
        and arm["post_optimizer_digest"] == reference["post_optimizer_digest"]
        and arm["parameter_digests"] == reference["parameter_digests"]
        for arm in paired_noops
    )
    rng_progression_paired = all(
        arm["post_rng_digest"] == reference["post_rng_digest"]
        and arm["rng_probes"] == reference["rng_probes"]
        for arm in evidence.values()
    )
    applied = evidence["APPLY"]
    apply_effect_observed = (
        applied["effective_intervention_ids"] == [_INTERVENTION_ID]
        and applied["post_model_digest"] != reference["post_model_digest"]
        and applied["parameter_digests"]["head.weight"]
        == applied["pre_parameter_digests"]["head.weight"]
        and reference["parameter_digests"]["head.weight"]
        != reference["pre_parameter_digests"]["head.weight"]
    )
    checks = {
        "anchors_restored": anchors_restored,
        "apply_effect_observed": apply_effect_observed,
        "disabled_control_shadow_bitwise": disabled_control_shadow_bitwise,
        "rng_progression_paired": rng_progression_paired,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest.manifest_id,
        "valid": all(checks.values()),
        "checks": checks,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    encoded = data.encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    expected = hashlib.sha256(encoded).hexdigest()
    actual = _sha256_file(path)
    if actual != expected:
        raise AblationContractError(
            f"SHA-256 validation failed after writing {path.name}"
        )
    return actual


def _manifest_from_payload(payload: Mapping[str, Any]) -> AblationManifest:
    expected_fields = {
        "schema_version",
        "manifest_id",
        "reference_arm",
        "registered_arms",
        "anchors",
        "registrations",
        "intervention",
        "artifacts",
    }
    if set(payload) != expected_fields:
        raise AblationContractError("manifest fields mismatch")
    try:
        anchors = AblationAnchors(**dict(payload["anchors"]))
        registrations = tuple(
            ArmRegistration(**dict(item))
            for item in payload["registrations"]
        )
        return AblationManifest(
            schema_version=payload["schema_version"],
            manifest_id=payload["manifest_id"],
            reference_arm=payload["reference_arm"],
            registered_arms=tuple(payload["registered_arms"]),
            anchors=anchors,
            registrations=registrations,
            intervention=dict(payload["intervention"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AblationContractError):
            raise
        raise AblationContractError(
            f"invalid manifest contract: {exc}"
        ) from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AblationContractError(
            f"invalid JSON artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AblationContractError(
            f"JSON artifact {path.name} must be an object"
        )
    return payload


def verify_artifacts(output_dir: Path | str) -> dict[str, Any]:
    """Verify the published manifest, exact file set, and every SHA-256."""

    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise AblationContractError(
            f"ablation artifact directory is absent: {root}"
        )
    manifest_path = root / "manifest.json"
    manifest_payload = _read_json_object(manifest_path)
    manifest = _manifest_from_payload(manifest_payload)

    artifacts = manifest_payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        _ARTIFACT_PATHS
    ):
        raise AblationContractError("manifest artifact set mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = {"manifest.json", *_ARTIFACT_PATHS}
    if actual_files != expected_files:
        raise AblationContractError(
            "published artifact file set mismatch"
        )

    artifact_payloads: dict[str, dict[str, Any]] = {}
    for relative in _ARTIFACT_PATHS:
        entry = artifacts.get(relative)
        if not isinstance(entry, dict) or set(entry) != {"sha256"}:
            raise AblationContractError(
                f"invalid manifest hash entry for {relative}"
            )
        expected_hash = entry.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)
        ):
            raise AblationContractError(
                f"invalid SHA-256 declaration for {relative}"
            )
        artifact_path = (root / Path(relative)).resolve()
        if not artifact_path.is_relative_to(root):
            raise AblationContractError(
                f"artifact escapes publication root: {relative}"
            )
        actual_hash = _sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise AblationContractError(
                f"SHA-256 mismatch for {relative}"
            )
        artifact_payloads[relative] = _read_json_object(artifact_path)

    for arm in (REFERENCE_ARM, *REGISTERED_ARMS):
        relative = f"arms/{arm.lower()}.json"
        arm_payload = artifact_payloads[relative]
        if (
            arm_payload.get("schema_version") != SCHEMA_VERSION
            or arm_payload.get("manifest_id") != manifest.manifest_id
            or arm_payload.get("arm") != arm
        ):
            raise AblationContractError(
                f"arm artifact contract mismatch for {arm}"
            )
    verdict = artifact_payloads["verdict.json"]
    if (
        verdict.get("schema_version") != SCHEMA_VERSION
        or verdict.get("manifest_id") != manifest.manifest_id
        or verdict.get("valid") is not True
    ):
        raise AblationContractError("verdict artifact contract mismatch")
    checks = verdict.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise AblationContractError("verdict checks are not all valid")
    return verdict


def _cleanup_staging(staging: Path, destination: Path) -> None:
    if not staging.exists():
        return
    resolved_staging = staging.resolve()
    resolved_parent = destination.parent.resolve()
    expected_prefix = f".{destination.name}.staging-"
    if (
        resolved_staging.parent != resolved_parent
        or not resolved_staging.name.startswith(expected_prefix)
    ):
        raise RuntimeError(
            f"refusing unsafe staging cleanup: {resolved_staging}"
        )
    shutil.rmtree(resolved_staging)


def run_causal_ablation(
    output_dir: Path | str,
    *,
    seed: int = 51,
    data_cursor: str = DEFAULT_DATA_CURSOR,
    training_contract_id: str = DEFAULT_TRAINING_CONTRACT,
    registration_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run four paired arms and persist immutable local evidence."""

    destination = Path(output_dir)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"ablation artifact already exists: {destination}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AblationContractError("seed must be a non-negative integer")
    if not data_cursor:
        raise AblationContractError("data_cursor must not be empty")
    if not training_contract_id:
        raise AblationContractError(
            "training_contract_id must not be empty"
        )

    caller_rng = _snapshot_rng()
    staging: Path | None = None
    try:
        prepared = _prepare(
            seed=seed,
            data_cursor=data_cursor,
            training_contract_id=training_contract_id,
            registration_overrides=registration_overrides,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=destination.parent,
            )
        )
        evidence = {
            arm: _run_arm(prepared, arm)
            for arm in (REFERENCE_ARM, *REGISTERED_ARMS)
        }
        artifact_hashes: dict[str, dict[str, str]] = {}
        for arm, payload in evidence.items():
            relative = f"arms/{arm.lower()}.json"
            artifact_hashes[relative] = {
                "sha256": _write_json_exclusive(
                    staging / relative,
                    payload,
                )
            }
        verdict = _build_verdict(prepared.manifest, evidence)
        artifact_hashes["verdict.json"] = {
            "sha256": _write_json_exclusive(
                staging / "verdict.json",
                verdict,
            )
        }
        if not verdict["valid"]:
            raise AblationContractError(
                f"paired causal ablation failed: {verdict['checks']}"
            )
        manifest_payload = prepared.manifest.to_dict()
        manifest_payload["artifacts"] = artifact_hashes
        _write_json_exclusive(
            staging / "manifest.json",
            manifest_payload,
        )
        verified = verify_artifacts(staging)
        if destination.exists():
            raise FileExistsError(
                f"ablation artifact already exists: {destination}"
            )
        staging.rename(destination)
        staging = None
        return verified
    except BaseException:
        if staging is not None:
            try:
                _cleanup_staging(staging, destination)
            except Exception as cleanup_exc:
                raise RuntimeError(
                    "ablation failed and verified staging cleanup failed"
                ) from cleanup_exc
        raise
    finally:
        _restore_rng(caller_rng)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated tiny causal ablation harness."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--data-cursor", default=DEFAULT_DATA_CURSOR)
    parser.add_argument(
        "--training-contract-id",
        default=DEFAULT_TRAINING_CONTRACT,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    verdict = run_causal_ablation(
        args.output,
        seed=args.seed,
        data_cursor=args.data_cursor,
        training_contract_id=args.training_contract_id,
    )
    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
