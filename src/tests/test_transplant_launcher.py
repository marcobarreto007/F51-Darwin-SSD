from pathlib import Path


def test_two_donor_launcher_isolated_and_canary_does_not_launch() -> None:
    text = Path("src/tools/start_two_donor_transplant.ps1").read_text("utf-8")
    assert "[switch]$Canary" in text
    assert "[switch]$GuardedStackCanary" in text
    assert "03_CHECKPOINTS_DARWIN_TWO_DONOR_V1" in text
    assert "organism_cycle_002_step_001750.pt" in text
    assert "donor_manifest.json" in text
    assert "Get-CimInstance Win32_Process" in text
    assert "if ($Canary)" in text
    assert "if ($GuardedStackCanary)" in text
    assert "--guarded-stack-canary" in text
    assert "Start-Process" in text

