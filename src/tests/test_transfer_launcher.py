from pathlib import Path


def test_launcher_has_non_launching_canary_and_isolated_roots() -> None:
    text = Path("src/tools/start_darwin_transfer.ps1").read_text(encoding="utf-8")
    assert "[switch]$Canary" in text
    assert "03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1" in text
    assert "darwin_transfer_distilgpt2_v1" in text
    assert "donor_manifest.json" in text
    assert "Get-CimInstance Win32_Process" in text
    assert '$_.Name -match "^python"' in text
    assert "Start-Process" in text
    assert "if ($Canary)" in text
    assert "--checkpoint-every" in text
    assert "--run-id" in text
    assert "--orientation-steps" in text
    assert "--alignment-steps" in text
    assert "--distillation-steps" in text
    assert "[int]$BatchSize = 8" in text
    assert "--batch-size" in text
    assert "[switch]$DeferPromotion" in text
    assert '"--defer-promotion"' in text
    assert "src/tests/test_transfer_ngram.py" in text
    assert '"--layers", "5"' not in text
