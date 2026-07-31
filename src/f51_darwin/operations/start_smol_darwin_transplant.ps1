[CmdletBinding()]
param(
    [switch]$Canary,
    [switch]$Calibrate,
    [switch]$PublishCandidate
)

$ErrorActionPreference = "Stop"
$actions = @($Canary, $Calibrate, $PublishCandidate).Where({ $_ }).Count
if ($actions -ne 1) {
    throw "Cannot combine actions; select exactly one."
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$env:PYTHONPATH = Join-Path $repo "src"
$root = Join-Path $repo "workspace\03_CHECKPOINTS_1.7B_SMOL_DENSE_V1"
$checkpoint = Join-Path $root "organism_cycle_000.pt"
$manifest = Join-Path $root "organism_cycle_000.manifest.json"
$tokenizer = Join-Path $repo "workspace\01_TOKENIZER\smol_49152_transplant_v1"

if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "No manifest-approved transplant candidate"
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "No manifest-approved transplant candidate"
}

Push-Location $repo
try {
    if ($Canary) {
        $env:TRANSFORMERS_OFFLINE = "1"
        $env:HF_HUB_OFFLINE = "1"
        python research/native_smol_darwin_smoke.py `
            --checkpoint $checkpoint `
            --manifest $manifest `
            --tokenizer $tokenizer
        if ($LASTEXITCODE -ne 0) {
            throw "Native canary failed with exit code $LASTEXITCODE"
        }
        Write-Output "DARWIN_16B_SMOL_NATIVE_OK donor_loaded=false tokenizer=verified organs=verified"
        exit 0
    }
    if ($Calibrate) {
        python research/evaluate_smol_dense_candidate.py
        if ($LASTEXITCODE -ne 0) {
            throw "Dense-brain knowledge gates failed with exit code $LASTEXITCODE"
        }
        exit 0
    }
    python research/transplant_smol_dense_brain.py publish
    if ($LASTEXITCODE -ne 0) {
        throw "Dense-brain publication failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
