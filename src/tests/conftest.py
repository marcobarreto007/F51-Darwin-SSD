"""Collection policy for dependency-scoped test suites."""

from __future__ import annotations

import importlib.util


_TRANSPLANT_TESTS = [
    "test_circuit_ablation_runtime.py",
    "test_circuit_compatibility.py",
    "test_circuit_ledger_pointer.py",
    "test_circuit_package.py",
    "test_circuit_rollback.py",
    "test_circuit_skills.py",
    "test_circuit_taps.py",
    "test_circuit_transaction.py",
    "test_organ_causal_qa.py",
    "test_smol_dense_assembly.py",
    "test_smol_dense_qa_benchmark.py",
    "test_smol_dense_runtime.py",
    "test_smol_exact_assembly.py",
    "test_transfer_student.py",
    "test_transfer_trainer.py",
    "test_transplant_16b_donor_contract.py",
    "test_transplant_16b_moe.py",
    "test_transplant_checkpoint.py",
    "test_transplant_ledger.py",
    "test_transplant_lifecycle.py",
    "test_transplant_recipient.py",
]

_TRANSPLANT_EXTRA_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("safetensors", "tokenizers", "transformers")
)

collect_ignore = [] if _TRANSPLANT_EXTRA_AVAILABLE else _TRANSPLANT_TESTS
