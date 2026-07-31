from __future__ import annotations

from pathlib import Path

from f51_darwin.transplant_16b.dense_runtime import (
    run_dense_optimizer_smoke,
)


ROOT = Path(__file__).resolve().parents[2]


def test_optimizer_smoke_uses_saved_model_and_all_dense_matrices() -> None:
    names = set(run_dense_optimizer_smoke.__code__.co_names)
    assert "verify_shard_manifest" in names
    assert "load_state_dict" in names
    assert "enable_dual_gpu" in names
    assert "backward" in names
    assert "gate_proj" in names
    assert "up_proj" in names
    assert "down_proj" in names


def test_native_smoke_reports_dense_structure() -> None:
    text = (ROOT / "research/native_smol_darwin_smoke.py").read_text(
        "utf-8"
    )
    assert '"feed_forward_kind"' in text
    assert '"moe_modules"' in text
    assert '"dense_ffn_modules"' in text
    assert "dense candidate contains a MoE block" in text


def test_dense_evaluator_has_isolated_authorities() -> None:
    text = (
        ROOT / "research/evaluate_smol_dense_candidate.py"
    ).read_text("utf-8")
    assert "03_CHECKPOINTS_1.7B_SMOL_DENSE_V1" in text
    assert "darwin_17b_smol_dense_v1" in text
    assert "run_dense_optimizer_smoke" in text
    assert "evaluate_exact_candidate" in text
    assert '"training_steps": 0' in text
