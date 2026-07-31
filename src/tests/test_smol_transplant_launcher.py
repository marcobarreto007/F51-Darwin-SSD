from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_launcher_separates_operator_authorities() -> None:
    text = (
        ROOT
        / "src/f51_darwin/operations/start_smol_darwin_transplant.ps1"
    ).read_text("utf-8")
    assert "[switch]$Canary" in text
    assert "[switch]$Calibrate" in text
    assert "[switch]$PublishCandidate" in text
    assert "Cannot combine" in text
    assert "03_CHECKPOINTS_1.7B_SMOL_DENSE_V1" in text
    assert "evaluate_smol_dense_candidate.py" in text
    assert "transplant_smol_dense_brain.py publish" in text
    assert "03_CHECKPOINTS_1.7B_SMOL_EXACT_V3" not in text
    assert "03_CHECKPOINTS_1.6B_SMOL_TRANSPLANT_V1" not in text
    assert "DARWIN_16B_SMOL_NATIVE_OK" in text


def test_native_smoke_forbids_donor_and_transformers_model() -> None:
    text = (ROOT / "research/native_smol_darwin_smoke.py").read_text("utf-8")
    assert "TRANSFORMERS_OFFLINE" in text
    assert "HF_HUB_OFFLINE" in text
    assert "donor_loaded" in text
    assert "enable_dual_gpu" in text
    assert "AutoModel" not in text
