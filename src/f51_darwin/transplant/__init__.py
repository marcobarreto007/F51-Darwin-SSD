"""Two-donor organ transplantation primitives."""

from .bundle import (
    OrganBundle,
    OrganBundleManifest,
    TensorRecord,
    extract_first_slice_bundle,
    load_organ_bundle,
)
from .organs import (
    HeartbeatOrgan,
    ReconstructedOrgan,
    build_original_gaba,
    build_original_heartbeat,
    build_original_ihs,
    build_original_jepa,
    build_original_mtp,
    build_original_spider,
)
from .slots import (
    GABAResidualSlot,
    HeartbeatSlot,
    IHSResidualSlot,
    JEPAAuxiliarySlot,
    MTPSlot,
    SpiderSenseSlot,
    OrganAdapter,
    OrganSlotOutput,
)
from .recipient import (
    GUARDED_TRANSPLANT_POLICY,
    RecipientOutput,
    TwoDonorRecipient,
)
from .lifecycle import (
    OrganState,
    audit_gradients,
    configure_trainable_state,
    configure_trainable_states,
    transition_organ,
)
from .experiment import (
    ArmBudget,
    ArmResult,
    ExperimentalArm,
    build_first_slice_arms,
    validate_matched_arms,
)
from .ledger import (
    LedgerRecord,
    OrganEvidence,
    OrganLedger,
    organ_utility,
)
from .checkpoint import (
    OrganSnapshot,
    RecipientCheckpoint,
    load_recipient_checkpoint,
    recipient_config_identity,
    rollback_organ,
    save_organ_snapshot,
    save_recipient_checkpoint,
)

__all__ = [
    "OrganBundle",
    "OrganBundleManifest",
    "TensorRecord",
    "extract_first_slice_bundle",
    "load_organ_bundle",
    "ReconstructedOrgan",
    "HeartbeatOrgan",
    "build_original_gaba",
    "build_original_heartbeat",
    "build_original_ihs",
    "build_original_jepa",
    "build_original_spider",
    "GABAResidualSlot",
    "HeartbeatSlot",
    "IHSResidualSlot",
    "JEPAAuxiliarySlot",
    "SpiderSenseSlot",
    "OrganAdapter",
    "OrganSlotOutput",
    "RecipientOutput",
    "TwoDonorRecipient",
    "GUARDED_TRANSPLANT_POLICY",
    "OrganState",
    "audit_gradients",
    "configure_trainable_state",
    "configure_trainable_states",
    "transition_organ",
    "ArmBudget",
    "ArmResult",
    "ExperimentalArm",
    "build_first_slice_arms",
    "validate_matched_arms",
    "LedgerRecord",
    "OrganEvidence",
    "OrganLedger",
    "organ_utility",
    "OrganSnapshot",
    "RecipientCheckpoint",
    "load_recipient_checkpoint",
    "recipient_config_identity",
    "rollback_organ",
    "save_organ_snapshot",
    "save_recipient_checkpoint",
]
