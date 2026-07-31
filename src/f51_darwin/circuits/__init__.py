"""Canonical identities and manifests for portable Darwin-X circuits."""

from f51_darwin.circuits.ablation import (
    AblationArm,
    AblationResult,
    ArmMetrics,
    CausalGateConfig,
    CausalVerdict,
    CircuitSelector,
    build_causal_verdict,
    discover_residual_channels,
    run_paired_ablation,
)
from f51_darwin.circuits.compatibility import (
    CircuitCompatibilityError,
    PreflightReport,
    RecipientIdentity,
    model_state_sha256,
    package_content_sha256,
    preflight_circuit,
)
from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256
from f51_darwin.circuits.ledger import CircuitLedger, LedgerError, LedgerRecord
from f51_darwin.circuits.manifest import CircuitManifest, TapContract, TensorIdentity
from f51_darwin.circuits.package import (
    CircuitPackageError,
    LoadedCircuitPackage,
    PackageIdentity,
    tensor_sha256,
    verify_circuit_package,
    write_circuit_package,
)
from f51_darwin.circuits.pointer import (
    ActivePointer,
    CheckpointCandidate,
    PointerConflict,
    PointerError,
    publish_active_pointer,
    read_active_pointer,
)
from f51_darwin.circuits.rollback import (
    CircuitCheckpoint,
    CircuitRollbackError,
    CircuitSnapshot,
    create_rollback_child,
)
from f51_darwin.circuits.taps import CircuitTapError, TapCapture, resolve_module
from f51_darwin.circuits.transaction import (
    CircuitTransaction,
    CircuitTransactionError,
    CircuitVerification,
    GatedCircuitWrapper,
)

__all__ = [
    "AblationArm",
    "AblationResult",
    "ArmMetrics",
    "CausalGateConfig",
    "CausalVerdict",
    "CheckpointCandidate",
    "CircuitCheckpoint",
    "CircuitSelector",
    "CircuitCompatibilityError",
    "CircuitLedger",
    "CircuitRollbackError",
    "CircuitSnapshot",
    "CircuitTapError",
    "CircuitTransaction",
    "CircuitTransactionError",
    "CircuitVerification",
    "CircuitManifest",
    "PreflightReport",
    "RecipientIdentity",
    "TapCapture",
    "TapContract",
    "TensorIdentity",
    "CircuitPackageError",
    "LoadedCircuitPackage",
    "GatedCircuitWrapper",
    "LedgerError",
    "LedgerRecord",
    "PackageIdentity",
    "PointerConflict",
    "PointerError",
    "ActivePointer",
    "canonical_json_bytes",
    "canonical_sha256",
    "build_causal_verdict",
    "discover_residual_channels",
    "model_state_sha256",
    "package_content_sha256",
    "preflight_circuit",
    "publish_active_pointer",
    "read_active_pointer",
    "resolve_module",
    "run_paired_ablation",
    "tensor_sha256",
    "verify_circuit_package",
    "write_circuit_package",
    "create_rollback_child",
]
