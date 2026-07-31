param(
    [switch]$Canary,
    [switch]$GuardedStackCanary,
    [string]$Device = "cuda:0",
    [int]$Seeds = 3,
    [int]$StepsPerArm = 1,
    [int]$SequenceLength = 32
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $RepoRoot ".venv_nitro\Scripts\python.exe"
$TransferSite = Join-Path $RepoRoot "workspace\runtime\transfer_python"
$DonorRoot = Join-Path $RepoRoot "workspace\00_DONORS\distilgpt2"
$DonorManifest = Join-Path $DonorRoot "donor_manifest.json"
$OrganCheckpoint = Join-Path $RepoRoot "workspace\03_CHECKPOINTS_100M_FULL_ORGANISM_V3\organism_cycle_002_step_001750.pt"
$BundlePath = Join-Path $RepoRoot "workspace\runtime\darwin_two_donor_v1\seven_organs_v2.pt"
$CorpusRoot = Join-Path $RepoRoot "workspace\02_CORPUS\corpus"
$CheckpointRoot = Join-Path $RepoRoot "workspace\03_CHECKPOINTS_DARWIN_TWO_DONOR_V1"
$RuntimeRoot = Join-Path $RepoRoot "workspace\runtime\darwin_two_donor_v1"
$PidFile = Join-Path $RuntimeRoot "worker.pid"
$Stdout = Join-Path $RuntimeRoot "worker.stdout.log"
$Stderr = Join-Path $RuntimeRoot "worker.stderr.log"
$RunId = "darwin-two-donor-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$OrganDonorSha256 = "71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30"

if ($Canary -and $GuardedStackCanary) {
    throw "Choose either -Canary or -GuardedStackCanary"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment missing: $Python"
}
if (-not (Test-Path -LiteralPath $DonorManifest -PathType Leaf)) {
    throw "Verified donor manifest missing: $DonorManifest"
}
if (-not (Test-Path -LiteralPath $OrganCheckpoint -PathType Leaf)) {
    throw "Original organ donor missing: $OrganCheckpoint"
}
if (-not (Test-Path -LiteralPath $CorpusRoot -PathType Container)) {
    throw "Local corpus missing: $CorpusRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $TransferSite "transformers") -PathType Container)) {
    throw "Isolated transfer dependencies missing: $TransferSite"
}

$env:PYTHONPATH = "$TransferSite;$RepoRoot\src"
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

Push-Location $RepoRoot
try {
    $TransplantTests = Get-ChildItem tests -Filter "test_transplant_*.py" |
        Select-Object -ExpandProperty FullName
    & $Python -m pytest -q -p no:cacheprovider @TransplantTests
    if ($LASTEXITCODE -ne 0) {
        throw "Two-donor source audit failed"
    }

    if ($Canary) {
        & $Python research/run_two_donor_canary.py `
            --donor-root $DonorRoot `
            --donor-manifest $DonorManifest `
            --organ-checkpoint $OrganCheckpoint `
            --organ-donor-sha256 $OrganDonorSha256 `
            --bundle-path $BundlePath `
            --corpus-root $CorpusRoot `
            --checkpoint-root $CheckpointRoot `
            --runtime-root $RuntimeRoot `
            --run-id "$RunId-canary" `
            --device cpu `
            --seeds 0 `
            --steps-per-arm 1 `
            --sequence-length 8 `
            --verify-only
        if ($LASTEXITCODE -ne 0) {
            throw "Two-donor CPU bundle smoke failed"
        }
        Write-Output "CANARY_OK launch=false donors=verified tests=passed smoke=passed"
        exit 0
    }

    $Competing = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.Name -match "^python" -and (
            $_.CommandLine -match "train_darwin_transfer.py" -or
            $_.CommandLine -match "darwin_organism.py" -or
            $_.CommandLine -match "run_two_donor_canary.py"
        )
    }
    if ($Competing) {
        $Ids = ($Competing | Select-Object -ExpandProperty ProcessId) -join ","
        throw "Competing trainer detected; PIDs=$Ids"
    }

    New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null
    if ($GuardedStackCanary) {
        & $Python research/run_two_donor_canary.py `
            --donor-root $DonorRoot `
            --donor-manifest $DonorManifest `
            --organ-checkpoint $OrganCheckpoint `
            --organ-donor-sha256 $OrganDonorSha256 `
            --bundle-path $BundlePath `
            --corpus-root $CorpusRoot `
            --checkpoint-root $CheckpointRoot `
            --runtime-root $RuntimeRoot `
            --run-id "$RunId-guarded" `
            --device $Device `
            --seeds $Seeds `
            --steps-per-arm $StepsPerArm `
            --sequence-length $SequenceLength `
            --guarded-stack-canary
        if ($LASTEXITCODE -ne 0) {
            throw "Guarded organ stack canary failed"
        }
        Write-Output "GUARDED_STACK_OK launch=false policy=mixed tests=passed"
        exit 0
    }

    $Arguments = @(
        "-u", "research/run_two_donor_canary.py",
        "--donor-root", $DonorRoot,
        "--donor-manifest", $DonorManifest,
        "--organ-checkpoint", $OrganCheckpoint,
        "--organ-donor-sha256", $OrganDonorSha256,
        "--bundle-path", $BundlePath,
        "--corpus-root", $CorpusRoot,
        "--checkpoint-root", $CheckpointRoot,
        "--runtime-root", $RuntimeRoot,
        "--run-id", $RunId,
        "--device", $Device,
        "--seeds", "$Seeds",
        "--steps-per-arm", "$StepsPerArm",
        "--sequence-length", "$SequenceLength"
    )
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
    Write-Output "LAUNCHED pid=$($Process.Id) run_id=$RunId checkpoint_root=$CheckpointRoot runtime_root=$RuntimeRoot"
}
finally {
    Pop-Location
}
