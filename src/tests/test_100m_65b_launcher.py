from __future__ import annotations

from pathlib import Path


SCRIPT = Path("src/scripts/start_100m_65b.ps1")


def test_launcher_owns_exact_100m_65b_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "[switch]$Canary" in text
    assert "[switch]$Launch" in text
    assert "[switch]$Worker" in text
    # O .ps1 guarda o caminho com barras duplas literais, e PowerShell nao
    # usa "\" como escape. Em Python isso exige raw string: a versao anterior,
    # "src\\configs\\\darwin_x_100m.yaml", produzia "\d" -- escape invalido --
    # e nunca casava com o texto do arquivo.
    assert r"src\\configs\\darwin_x_100m.yaml" in text
    assert "65_000_000_000" in text
    assert "03_CHECKPOINTS_100M_FULL_V1" in text
    assert "00_CORPUS_PRINCIPAL_tokens_feast_v2.bin" in text
    assert "train-budget" in text
    assert "run247" not in text


def test_launcher_fails_closed_before_background_launch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Tracked worktree must be clean" in text
    assert "another Darwin trainer is active" in text
    assert "two CUDA GPUs are required" in text
    assert "MinimumFreeDiskGB" in text
    assert "Start-Process" in text
    assert "-WindowStyle Hidden" in text
    assert "source_commit" in text
    assert "process_id" in text


def test_launcher_runs_three_causal_canaries_and_strict_inspection() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for mode in ("control", "shadow", "enforce"):
        assert f'"{mode}"' in text
    assert '"--canary"' in text
    assert "--verify-identity" in text
    assert "strict_resume_compatible" in text
    assert "DUAL GPU" in text
