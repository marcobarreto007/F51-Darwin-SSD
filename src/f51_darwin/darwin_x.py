from __future__ import annotations

import uuid

from f51_darwin.darwin_x_core import (
    DarwinXBlock,
    DarwinXConfig,
    DarwinXModel,
    DarwinXOutput,
    DenseSwiGLU,
    DeepSeekStyleMoE,
    ExpertFFN,
    FineRouter,
    GQACausalAttention,
    NeuroendocrineSystem,
    SSDMixerOnly,
    estimate_darwin_x_parameters,
    migrate_mutational_state_for_load,
)
from f51_darwin.darwin_x_core.state import _MUTATIONAL_STATE_SUFFIXES


# Preserve the historical pickle/import authority after the implementation was
# decomposed. The shared stdlib module keeps ``darwin_x.uuid.uuid4``
# monkeypatchable for deterministic topology tests and tooling.
for _public_class in (
    DarwinXBlock,
    DarwinXConfig,
    DarwinXModel,
    DarwinXOutput,
    DenseSwiGLU,
    DeepSeekStyleMoE,
    ExpertFFN,
    FineRouter,
    GQACausalAttention,
    NeuroendocrineSystem,
    SSDMixerOnly,
):
    _public_class.__module__ = __name__


__all__ = [
    "DarwinXBlock",
    "DarwinXConfig",
    "DarwinXModel",
    "DarwinXOutput",
    "DenseSwiGLU",
    "DeepSeekStyleMoE",
    "ExpertFFN",
    "FineRouter",
    "GQACausalAttention",
    "NeuroendocrineSystem",
    "SSDMixerOnly",
    "estimate_darwin_x_parameters",
    "migrate_mutational_state_for_load",
]
