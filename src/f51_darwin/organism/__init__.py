"""Operational Darwin organism runtime.

The supported command remains ``src/scripts/darwin_organism.py``.  This package
owns the implementation so importers do not depend on an executable script.
"""

from .runtime import DarwinOrganism, DarwinOrganismConfig

# ── Blockchain & Senate exports ──
# Modules may not exist yet; degrade gracefully so the runtime still works
# when blockchain_enabled=False / senate_enabled=False.
try:
    from f51_darwin.organism.blockchain import Block, BlockHeader, MerkleTree, OrganLedger  # noqa: F401
except ImportError:
    Block = None  # type: ignore[assignment]
    BlockHeader = None  # type: ignore[assignment]
    MerkleTree = None  # type: ignore[assignment]
    OrganLedger = None  # type: ignore[assignment]

try:
    from f51_darwin.organism.organ_reputation import OrganReputation, OrganReputationTracker  # noqa: F401
except ImportError:
    OrganReputation = None  # type: ignore[assignment]
    OrganReputationTracker = None  # type: ignore[assignment]

try:
    from f51_darwin.organism.organ_senate import OrganSenate, SenateDecision, SenateLedger  # noqa: F401
except ImportError:
    OrganSenate = None  # type: ignore[assignment]
    SenateDecision = None  # type: ignore[assignment]
    SenateLedger = None  # type: ignore[assignment]

__all__ = [
    "DarwinOrganism",
    "DarwinOrganismConfig",
    "Block",
    "BlockHeader",
    "MerkleTree",
    "OrganLedger",
    "OrganReputation",
    "OrganReputationTracker",
    "OrganSenate",
    "SenateDecision",
    "SenateLedger",
]
