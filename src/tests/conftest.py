"""Collection policy for dependency-scoped test suites."""

from __future__ import annotations

import importlib.util


_TRANSPLANT_TEST_GLOBS = [
    "test_circuit_*.py",
    "test_organ_causal_*.py",
    "test_smol_*.py",
    "test_transfer_*.py",
    "test_transplant_*.py",
]

_TRANSPLANT_EXTRA_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("safetensors", "tokenizers", "transformers")
)

collect_ignore_glob = [] if _TRANSPLANT_EXTRA_AVAILABLE else _TRANSPLANT_TEST_GLOBS
