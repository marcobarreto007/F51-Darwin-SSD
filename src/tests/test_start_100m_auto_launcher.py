from __future__ import annotations

from pathlib import Path


SCRIPT = Path("src/scripts/start_100m_auto.ps1")
CLI = Path("src/f51_darwin/organism/cli.py")


def _launcher_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_launcher_uses_new_shuffled_root_and_preserves_halt_gate() -> None:
    text = _launcher_text()

    assert r"src\\configs\\100m_lm_recovery_v1.yaml" in text
    assert "03_CHECKPOINTS_100M_SHUFFLED_V1" in text
    assert "03_CHECKPOINTS_100M_FULL_V10" not in text
    assert "03_CHECKPOINTS_100M_FULL_V9" not in text
    assert 'Join-Path $ROOT "TRAINING_HALTED.flag"' in text
    assert "if (-not $PreflightOnly)" in text
    assert "exit 1" in text
    assert text.index("$halted = Test-Path") < text.index("while ($true)")


def test_launcher_never_selects_a_checkpoint_by_mtime_or_glob() -> None:
    text = _launcher_text()

    assert "organism_latest.json" in text
    assert "lineage_root.json" in text
    assert "Sort-Object LastWriteTime" not in text
    assert "LastWriteTime" not in text
    assert 'Get-ChildItem "$CKPT_ROOT\\organism_*.pt"' not in text
    assert "Resolve-StrictCheckpointHead" in text


def test_resume_is_bound_to_pointer_lineage_size_identity_and_inspector() -> None:
    text = _launcher_text()

    for required_contract in (
        "pointer path must name a direct child of checkpoint root",
        "pointer size mismatch",
        "pointer and lineage_root base_checkpoint_id disagree",
        "strict_resume_compatible",
        "pointer cycle/step disagree with checkpoint training_state",
        "pointer checkpoint_version disagrees with checkpoint report",
        "--verify-identity",
    ):
        assert required_contract in text

    assert "[IO.FileAttributes]::ReparsePoint" in text
    assert 'mode = "fresh_start"' in text
    assert "$entries.Count -eq 0" in text
    assert "non-empty checkpoint root has no published organism_latest.json" in text


def test_launcher_hashing_does_not_depend_on_optional_powershell_cmdlets() -> None:
    text = _launcher_text()

    assert "function Get-Sha256Hex" in text
    assert "[System.Security.Cryptography.SHA256]::Create()" in text
    assert "Get-FileHash" not in text


def test_launcher_enables_real_fixed_holdout_on_explicit_feast_v2() -> None:
    text = _launcher_text()

    assert "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin" in text
    assert "$HOLDOUT_TOKENS = [int64]65536" in text
    assert "$HOLDOUT_BATCHES = 4" in text
    assert "$HOLDOUT_SEED = 999" in text
    for option in (
        '"--token-bin"',
        '"--holdout-tokens"',
        '"--holdout-batches"',
        '"--holdout-seed"',
        '"--metrics-jsonl"',
        '"--canary-metadata-json"',
    ):
        assert option in text
    assert 'policy = "fixed_tail_excluded_from_training"' in text


def test_launcher_uses_supported_permuted_block_sampler_and_disables_autonomy() -> None:
    launcher = _launcher_text()
    cli = CLI.read_text(encoding="utf-8")

    assert '"--sampler-mode"' in cli
    assert '"--sampler-version"' in cli
    assert '"--sampler-mode", "permuted_blocks"' in launcher
    assert '"--sampler-version", "1"' in launcher
    assert '"--no-autonomous-drives"' in launcher
    assert '"--no-organ-rl"' in launcher
    assert '"--senate-enabled"' not in launcher
    assert '"--senate-interval"' not in launcher
