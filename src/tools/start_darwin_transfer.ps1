param(
    [switch]$Canary,
    [int]$OrientationSteps = 250,
    [int]$AlignmentSteps = 250,
    [int]$DistillationSteps = 500,
    [int]$BatchSize = 8,
    [int]$CheckpointEvery = 25,
    [switch]$DeferPromotion,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $RepoRoot ".venv_nitro\Scripts\python.exe"
$TransferSite = Join-Path $RepoRoot "workspace\runtime\transfer_python"
$DonorRoot = Join-Path $RepoRoot "workspace\00_DONORS\distilgpt2"
$DonorManifest = Join-Path $DonorRoot "donor_manifest.json"
$CheckpointRoot = Join-Path $RepoRoot "workspace\03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1"
$RuntimeRoot = Join-Path $RepoRoot "workspace\runtime\darwin_transfer_distilgpt2_v1"
$PidFile = Join-Path $RuntimeRoot "trainer.pid"
$Stdout = Join-Path $RuntimeRoot "trainer.stdout.log"
$Stderr = Join-Path $RuntimeRoot "trainer.stderr.log"
$RunId = "darwin-transfer-" + (Get-Date -Format "yyyyMMdd-HHmmss")

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment missing: $Python"
}
if (-not (Test-Path -LiteralPath $DonorManifest -PathType Leaf)) {
    throw "Verified donor manifest missing: $DonorManifest"
}
if (-not (Test-Path -LiteralPath (Join-Path $TransferSite "transformers") -PathType Container)) {
    throw "Isolated transfer dependencies missing: $TransferSite"
}
$env:PYTHONPATH = "$TransferSite;$RepoRoot\src"

Push-Location $RepoRoot
try {
    & $Python -c "from pathlib import Path; from f51_darwin.transfer.donor import load_frozen_teacher; load_frozen_teacher(Path(r'$DonorRoot')); print('DONOR_OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "Donor hash verification failed"
    }
    & $Python -m pytest -q -p no:cacheprovider src/tests/test_transfer_donor.py src/tests/test_transfer_ssd_mixer.py src/tests/test_transfer_student.py src/tests/test_transfer_trainer.py src/tests/test_transfer_experience.py src/tests/test_transfer_schedule.py src/tests/test_transfer_ngram.py src/tests/test_transfer_launcher.py
    if ($LASTEXITCODE -ne 0) {
        throw "Transfer source audit failed"
    }
    if ($Canary) {
        Write-Output "CANARY_OK launch=false donor=verified tests=passed"
        exit 0
    }

    $Competing = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.Name -match "^python" -and (
            $_.CommandLine -match "train_darwin_ssd.py" -or
            $_.CommandLine -match "darwin_organism.py" -or
            $_.CommandLine -match "train_darwin_transfer.py"
        )
    }
    if ($Competing) {
        $Ids = ($Competing | Select-Object -ExpandProperty ProcessId) -join ","
        throw "Competing trainer detected; PIDs=$Ids"
    }

    New-Item -ItemType Directory -Force -Path $CheckpointRoot, $RuntimeRoot | Out-Null
    $Arguments = @(
        "-u", "research/train_darwin_transfer.py",
        "--donor", $DonorRoot,
        "--repo-root", $RepoRoot,
        "--checkpoint-root", $CheckpointRoot,
        "--runtime-root", $RuntimeRoot,
        "--corpus-root", (Join-Path $RepoRoot "workspace\\02_CORPUS\\corpus"),
        "--sequence-length", "128",
        "--monitor-holdout-chunks", "256",
        "--promotion-holdout-tokens", "5000000",
        "--promotion-holdout-batches", "256",
        "--orientation-steps", "$OrientationSteps",
        "--alignment-steps", "$AlignmentSteps",
        "--distillation-steps", "$DistillationSteps",
        "--batch-size", "$BatchSize",
        "--checkpoint-every", "$CheckpointEvery",
        "--learning-rate", "0.00003",
        "--device", $Device,
        "--run-id", $RunId
    )
    if ($DeferPromotion) {
        $Arguments += "--defer-promotion"
    }
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
    Write-Output "LAUNCHED pid=$($Process.Id) run_id=$RunId checkpoint_root=$CheckpointRoot runtime_root=$RuntimeRoot"
}
finally {
    Pop-Location
}
