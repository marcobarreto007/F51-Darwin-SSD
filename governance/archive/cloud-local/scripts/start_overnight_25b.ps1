param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$DatasetRoot = "C:\Users\marco\Desktop\F51-Dataset-Organizado",
    [string]$Config = "configs\darwin_x_2.5b.yaml",
    [string]$Checkpoint = "",
    [int]$StepsPerCycle = 500,
    [int]$BlockSize = 64,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.00015,
    [int]$WarmupSteps = 0,
    [int]$AccumSteps = 1,
    [int64]$MinimumModelParameters = 2400000000,
    [int64]$MinimumCheckpointBytes = 4294967296,
    [int]$StabilitySeconds = 15,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"

function Assert-File {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

$Root = (Resolve-Path -LiteralPath $Root).Path
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$python = Join-Path $Root ".venv_nitro\Scripts\python.exe"
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
$tokenBin = Join-Path $DatasetRoot "01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast.bin"
$tokenManifestPath = "$tokenBin.manifest.json"
$checkpointRoot = Join-Path $DatasetRoot "03_CHECKPOINTS"
$manifestRoot = Join-Path $DatasetRoot "04_MANIFESTOS"

Assert-File $python "Training Python"
Assert-File $configPath "Model config"
Assert-File $tokenBin "External token corpus"
Assert-File $tokenManifestPath "Token manifest"

$existing = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match 'darwin_organism\.py\s+run247' -and
            $_.ProcessId -ne $PID
        }
)
if ($existing.Count -gt 0) {
    $pids = ($existing.ProcessId -join ",")
    throw "A run247 process is already active (PID: $pids)."
}

$tokenInfo = Get-Item -LiteralPath $tokenBin
if ($tokenInfo.Length -le 0 -or ($tokenInfo.Length % 4) -ne 0) {
    throw "Token corpus is empty or not aligned as int32: bytes=$($tokenInfo.Length)"
}
$tokenManifest = Get-Content -LiteralPath $tokenManifestPath -Raw | ConvertFrom-Json
if ([int64]$tokenManifest.bytes -ne $tokenInfo.Length) {
    throw "Token manifest size mismatch: manifest=$($tokenManifest.bytes) disk=$($tokenInfo.Length)"
}
if ($tokenManifest.dtype -ne "int32" -or -not [bool]$tokenManifest.little_endian) {
    throw "Token manifest contract must be little-endian int32."
}

$inspectScript = Join-Path $Root "scripts\inspect_darwin_x.py"
$inspectText = & $python $inspectScript --config $configPath --instantiate-meta 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "Model config inspection failed: $inspectText"
}
$modelReport = $inspectText | ConvertFrom-Json
$modelParameters = [int64]$modelReport.meta_instantiated_params
if ($modelParameters -lt $MinimumModelParameters) {
    throw (
        "Refusing a smaller substitute: config has $modelParameters parameters; " +
        "minimum required for the 2.5B lineage is $MinimumModelParameters."
    )
}

$gpuLines = @(
    & "C:\Windows\System32\nvidia-smi.exe" `
        --query-gpu=index,name,memory.total `
        --format=csv,noheader,nounits
)
if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -lt 2) {
    throw "Two NVIDIA GPUs are required; detected $($gpuLines.Count)."
}

$checkpointPath = ""
if ($Checkpoint) {
    $checkpointPath = if ([IO.Path]::IsPathRooted($Checkpoint)) {
        $Checkpoint
    } else {
        Join-Path $checkpointRoot $Checkpoint
    }
    Assert-File $checkpointPath "2.5B checkpoint"
    $first = (Get-Item -LiteralPath $checkpointPath).Length
    if ($first -lt $MinimumCheckpointBytes) {
        throw "Checkpoint is too small or still partial: bytes=$first"
    }
    Start-Sleep -Seconds $StabilitySeconds
    $second = (Get-Item -LiteralPath $checkpointPath).Length
    if ($first -ne $second) {
        throw "Checkpoint is still changing: first=$first second=$second"
    }
} elseif ($Launch) {
    throw "-Checkpoint is required for the trained 2.5B overnight lineage."
}

$configHash = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
$readiness = [ordered]@{
    ready = $true
    checked_at = (Get-Date).ToUniversalTime().ToString("o")
    project_root = $Root
    dataset_root = $DatasetRoot
    config = $configPath
    config_sha256 = $configHash
    model_name = $modelReport.model_name
    model_parameters = $modelParameters
    active_parameters_per_token = [int64]$modelReport.parameter_estimate.active_per_token
    token_bin = $tokenBin
    token_bytes = $tokenInfo.Length
    token_count = [int64]$tokenManifest.tokens
    token_sha256_manifest = $tokenManifest.sha256
    checkpoint = $checkpointPath
    checkpoint_bytes = $(if ($checkpointPath) { (Get-Item -LiteralPath $checkpointPath).Length } else { 0 })
    gpus = $gpuLines
    organs = @("heartbeat", "ghost", "jepa", "curiosity", "spider", "mtp", "expert_pool", "legacy", "lineage")
    block_size = $BlockSize
    batch_size = $BatchSize
    steps_per_cycle = $StepsPerCycle
    learning_rate = $LearningRate
    warmup_steps = $WarmupSteps
    accum_steps = $AccumSteps
}

New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null
$readinessPath = Join-Path $manifestRoot "overnight_25b_readiness.json"
$readiness | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $readinessPath -Encoding utf8

if (-not $Launch) {
    $readiness | ConvertTo-Json -Depth 6
    Write-Host "DRY RUN OK. Add -Checkpoint and -Launch only after the checkpoint is complete."
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $Root "runs\overnight_25b\$stamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$stdout = Join-Path $runDir "train.stdout.log"
$stderr = Join-Path $runDir "train.stderr.log"
$pidPath = Join-Path $runDir "train.pid"
$launchManifest = Join-Path $runDir "launch_manifest.json"

$arguments = @(
    "-u",
    "scripts/darwin_organism.py",
    "run247",
    "--config", $configPath,
    "--token-bin", $tokenBin,
    "--resume", $checkpointPath,
    "--steps", [string]$StepsPerCycle,
    "--block-size", [string]$BlockSize,
    "--batch-size", [string]$BatchSize,
    "--lr", [string]$LearningRate,
    "--warmup-steps", [string]$WarmupSteps,
    "--accum-steps", [string]$AccumSteps,
    "--device", "cuda",
    "--eval-every", [string]$StepsPerCycle
)

$env:F51_DATASET_ROOT = $DatasetRoot
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

Set-Content -LiteralPath $pidPath -Value ([string]$process.Id) -Encoding ascii
$readiness.pid = $process.Id
$readiness.started_at = (Get-Date).ToUniversalTime().ToString("o")
$readiness.stdout = $stdout
$readiness.stderr = $stderr
$readiness | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchManifest -Encoding utf8

Write-Host "Overnight 2.5B launch started. PID=$($process.Id)"
Write-Host "stdout=$stdout"
Write-Host "stderr=$stderr"
