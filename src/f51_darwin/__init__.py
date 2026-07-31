"""F51 Darwin-SSD public API with dependency-lazy exports.

Importing :mod:`f51_darwin` is intentionally safe in the base distribution;
Torch and the runtime stack load only when a runtime symbol is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from f51_darwin._version import __version__


_EXPORTS: dict[str, tuple[str, str]] = {
    "Brainstem": ("f51_darwin.brainstem", "Brainstem"),
    "BrainstemConfig": ("f51_darwin.brainstem", "BrainstemConfig"),
    "HomeostasisReport": ("f51_darwin.brainstem", "HomeostasisReport"),
    "DarwinConfig": ("f51_darwin.config", "DarwinConfig"),
    "DarwinXConfig": ("f51_darwin.darwin_x", "DarwinXConfig"),
    "DarwinXModel": ("f51_darwin.darwin_x", "DarwinXModel"),
    "DarwinXOutput": ("f51_darwin.darwin_x", "DarwinXOutput"),
    "ExpertModule": ("f51_darwin.expert_pool", "ExpertModule"),
    "ExpertPool": ("f51_darwin.expert_pool", "ExpertPool"),
    "ModuleState": ("f51_darwin.expert_pool", "ModuleState"),
    "EpistemicLevel": ("f51_darwin.grounded_extractor", "EpistemicLevel"),
    "ExtractionEvidence": ("f51_darwin.grounded_extractor", "ExtractionEvidence"),
    "GroundedExtraction": ("f51_darwin.grounded_extractor", "GroundedExtraction"),
    "GroundedExtractor": ("f51_darwin.grounded_extractor", "GroundedExtractor"),
    "Heartbeat": ("f51_darwin.heartbeat", "Heartbeat"),
    "HeartbeatConfig": ("f51_darwin.heartbeat", "HeartbeatConfig"),
    "DarwinOutput": ("f51_darwin.model", "DarwinOutput"),
    "F51DarwinModel": ("f51_darwin.model", "F51DarwinModel"),
    "GenerationOutput": ("f51_darwin.model", "GenerationOutput"),
    "LineageTracker": ("f51_darwin.lineage_tracker", "LineageTracker"),
    "WolframBridge": ("f51_darwin.wolfram_bridge", "WolframBridge"),
    "WolframConfig": ("f51_darwin.wolfram_bridge", "WolframConfig"),
    "WolframResult": ("f51_darwin.wolfram_bridge", "WolframResult"),
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from f51_darwin.brainstem import Brainstem, BrainstemConfig, HomeostasisReport
    from f51_darwin.config import DarwinConfig
    from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel, DarwinXOutput
    from f51_darwin.expert_pool import ExpertModule, ExpertPool, ModuleState
    from f51_darwin.grounded_extractor import (
        EpistemicLevel,
        ExtractionEvidence,
        GroundedExtraction,
        GroundedExtractor,
    )
    from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
    from f51_darwin.lineage_tracker import LineageTracker
    from f51_darwin.model import DarwinOutput, F51DarwinModel, GenerationOutput
    from f51_darwin.wolfram_bridge import WolframBridge, WolframConfig, WolframResult
