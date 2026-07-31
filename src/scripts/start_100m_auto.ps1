param(
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = Join-Path $ROOT "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$VENV = Join-Path $ROOT ".venv_nitro\Scripts\python.exe"
$CONFIG = Join-Path $ROOT "src\\configs\\100m_lm_recovery_v1.yaml"
$CORPUS = Join-Path $ROOT "workspace\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"
$CORPUS_MANIFEST = "$CORPUS.manifest.json"
$CKPT_ROOT = Join-Path $ROOT "workspace\03_CHECKPOINTS_100M_SHUFFLED_V1"
$RUN_ROOT = Join-Path $ROOT "workspace\runtime\runs\100m_shuffled_v1"
$INSPECTOR_MODULE = "scripts.inspect_organism_checkpoint"
$STEPS = 500
$SAVE_EVERY = 100
$HOLDOUT_TOKENS = [int64]65536
$HOLDOUT_BATCHES = 4
$HOLDOUT_SEED = 999
$KILL_SWITCH = Join-Path $ROOT "TRAINING_HALTED.flag"

function Get-Sha256Hex {
    param(
        [Parameter(Mandatory = $true)][string]$LiteralPath
    )

    $stream = [System.IO.File]::OpenRead($LiteralPath)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash($stream)
        return ([System.BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Get-JsonProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-RequiredJsonProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Source
    )

    $value = Get-JsonProperty -Object $Object -Name $Name
    if ($null -eq $value -or ([string]$value).Length -eq 0) {
        throw "$Source is missing required field '$Name'"
    }
    return $value
}

function Assert-RequiredFiles {
    foreach ($path in @($VENV, $CONFIG, $CORPUS, $CORPUS_MANIFEST)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "required file is missing: $path"
        }
    }

    $manifest = Get-Content -Raw -LiteralPath $CORPUS_MANIFEST |
        ConvertFrom-Json
    $declaredBytes = [int64](Get-RequiredJsonProperty `
        -Object $manifest -Name "bytes" -Source $CORPUS_MANIFEST)
    $declaredTokens = [int64](Get-RequiredJsonProperty `
        -Object $manifest -Name "tokens" -Source $CORPUS_MANIFEST)
    $declaredSha256 = [string](Get-RequiredJsonProperty `
        -Object $manifest -Name "sha256" -Source $CORPUS_MANIFEST)
    $actualBytes = (Get-Item -LiteralPath $CORPUS).Length

    if ($declaredBytes -ne $actualBytes) {
        throw "feast_v2 manifest size mismatch: declared=$declaredBytes actual=$actualBytes"
    }
    if ($declaredTokens -ne ($actualBytes / 4)) {
        throw "feast_v2 token count does not match little-endian int32 size"
    }
    if (
        [string](Get-RequiredJsonProperty `
            -Object $manifest -Name "dtype" -Source $CORPUS_MANIFEST) -ne "int32" -or
        -not [bool](Get-RequiredJsonProperty `
            -Object $manifest -Name "little_endian" -Source $CORPUS_MANIFEST)
    ) {
        throw "feast_v2 manifest must declare little-endian int32"
    }
    if ($declaredSha256 -notmatch "^[0-9a-f]{64}$") {
        throw "feast_v2 manifest SHA-256 is invalid"
    }
}

function Assert-NoConcurrentTrainer {
    $trainers = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "^python(w)?\.exe$" -and
                $_.CommandLine -match "src\\scripts\\.darwin_organism" -and
                $_.CommandLine -match "run247|train-budget"
            }
    )
    if ($trainers.Count -gt 0) {
        throw "another Darwin trainer is active: $($trainers.ProcessId -join ',')"
    }
}

function Resolve-StrictCheckpointHead {
    param(
        [Parameter(Mandatory = $true)][string]$CheckpointRoot
    )

    $rootItem = Get-Item -LiteralPath $CheckpointRoot
    if (-not $rootItem.PSIsContainer) {
        throw "checkpoint root is not a directory: $CheckpointRoot"
    }
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "checkpoint root must not be a reparse point: $CheckpointRoot"
    }

    $canonicalRoot = [IO.Path]::GetFullPath($rootItem.FullName).TrimEnd("\", "/")
    $pointerPath = Join-Path $canonicalRoot "organism_latest.json"
    $lineagePath = Join-Path $canonicalRoot "lineage_root.json"
    if (-not (Test-Path -LiteralPath $pointerPath -PathType Leaf)) {
        throw "non-empty checkpoint root has no published organism_latest.json: $canonicalRoot"
    }
    if (-not (Test-Path -LiteralPath $lineagePath -PathType Leaf)) {
        throw "non-empty checkpoint root has no lineage_root.json: $canonicalRoot"
    }

    $pointer = Get-Content -Raw -LiteralPath $pointerPath | ConvertFrom-Json
    $lineage = Get-Content -Raw -LiteralPath $lineagePath | ConvertFrom-Json
    if ([int](Get-RequiredJsonProperty `
        -Object $pointer -Name "version" -Source $pointerPath) -ne 1) {
        throw "unsupported organism_latest.json version: $pointerPath"
    }

    $relativePath = [string](Get-RequiredJsonProperty `
        -Object $pointer -Name "path" -Source $pointerPath)
    if (
        [IO.Path]::IsPathRooted($relativePath) -or
        [IO.Path]::GetFileName($relativePath) -ne $relativePath
    ) {
        throw "pointer path must name a direct child of checkpoint root: $relativePath"
    }
    $checkpointPath = [IO.Path]::GetFullPath(
        [IO.Path]::Combine($canonicalRoot, $relativePath)
    )
    $checkpointParent = [IO.Path]::GetDirectoryName($checkpointPath)
    if (-not [string]::Equals(
        $checkpointParent, $canonicalRoot, [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "pointer checkpoint escaped checkpoint root: $checkpointPath"
    }
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "pointer checkpoint is missing: $checkpointPath"
    }

    $checkpointItem = Get-Item -LiteralPath $checkpointPath
    if (($checkpointItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "pointer checkpoint must not be a reparse point: $checkpointPath"
    }
    $declaredSize = [int64](Get-RequiredJsonProperty `
        -Object $pointer -Name "size_bytes" -Source $pointerPath)
    if ($declaredSize -le 0 -or $checkpointItem.Length -ne $declaredSize) {
        throw "pointer size mismatch: declared=$declaredSize actual=$($checkpointItem.Length)"
    }

    $pointerBaseId = [string](Get-RequiredJsonProperty `
        -Object $pointer -Name "base_checkpoint_id" -Source $pointerPath)
    if ($pointerBaseId -notmatch "^darwin-model-core-v1:[0-9a-f]{64}$") {
        throw "pointer base_checkpoint_id is invalid"
    }
    $lineageBaseId = [string](Get-RequiredJsonProperty `
        -Object $lineage -Name "base_checkpoint_id" -Source $lineagePath)
    if ($lineageBaseId -ne $pointerBaseId) {
        throw "pointer and lineage_root base_checkpoint_id disagree"
    }
    if ([int](Get-RequiredJsonProperty `
        -Object $lineage -Name "schema_version" -Source $lineagePath) -lt 2) {
        throw "lineage_root schema_version must be at least 2"
    }
    if ([string](Get-RequiredJsonProperty `
        -Object $lineage -Name "creation_mode" -Source $lineagePath) -ne "fresh_start") {
        throw "SHUFFLED_V1 lineage_root must have creation_mode=fresh_start"
    }
    foreach ($identityField in @("config_identity", "structural_config_identity")) {
        $identity = [string](Get-RequiredJsonProperty `
            -Object $lineage -Name $identityField -Source $lineagePath)
        if ($identity -notmatch "^darwin-[a-z-]+-v1:[0-9a-f]{64}$") {
            throw "lineage_root $identityField is invalid"
        }
    }

    $declaredSha256 = Get-JsonProperty -Object $pointer -Name "sha256"
    if ($null -ne $declaredSha256 -and ([string]$declaredSha256).Length -gt 0) {
        if ([string]$declaredSha256 -notmatch "^[0-9a-f]{64}$") {
            throw "pointer SHA-256 is invalid"
        }
        $actualSha256 = Get-Sha256Hex -LiteralPath $checkpointPath
        if ($actualSha256 -ne [string]$declaredSha256) {
            throw "pointer SHA-256 mismatch for $checkpointPath"
        }
    }

    $inspectionRows = @(
        & $VENV -m $INSPECTOR_MODULE $checkpointPath `
            --config $CONFIG --verify-identity 2>&1
    )
    $inspectionExitCode = $LASTEXITCODE
    $inspectionText = $inspectionRows -join [Environment]::NewLine
    if (-not $inspectionText) {
        throw "strict checkpoint inspector produced no report"
    }
    try {
        $inspection = $inspectionText | ConvertFrom-Json
    } catch {
        throw "strict checkpoint inspector returned invalid JSON: $inspectionText"
    }
    if (
        $inspectionExitCode -ne 0 -or
        -not [bool](Get-RequiredJsonProperty `
            -Object $inspection -Name "strict_resume_compatible" `
            -Source "checkpoint inspector")
    ) {
        throw "checkpoint is not strict-resume compatible: $checkpointPath"
    }

    $reportBaseId = [string](Get-RequiredJsonProperty `
        -Object $inspection -Name "base_checkpoint_id" `
        -Source "checkpoint inspector")
    $reportModel = [string](Get-RequiredJsonProperty `
        -Object $inspection -Name "model_name" `
        -Source "checkpoint inspector")
    $lineageModel = [string](Get-RequiredJsonProperty `
        -Object $lineage -Name "model_name" -Source $lineagePath)
    $reportBytes = [int64](Get-RequiredJsonProperty `
        -Object $inspection -Name "checkpoint_bytes" `
        -Source "checkpoint inspector")
    if ($reportBaseId -ne $pointerBaseId) {
        throw "checkpoint report and pointer base_checkpoint_id disagree"
    }
    if ($reportModel -ne $lineageModel) {
        throw "checkpoint report and lineage_root model_name disagree"
    }
    if ($reportBytes -ne $declaredSize) {
        throw "checkpoint report and pointer size disagree"
    }

    $trainingState = Get-RequiredJsonProperty `
        -Object $inspection -Name "training_state" `
        -Source "checkpoint inspector"
    $pointerCycle = [int64](Get-RequiredJsonProperty `
        -Object $pointer -Name "cycle" -Source $pointerPath)
    $pointerStep = [int64](Get-RequiredJsonProperty `
        -Object $pointer -Name "step" -Source $pointerPath)
    if (
        [int64](Get-RequiredJsonProperty `
            -Object $trainingState -Name "cycle" `
            -Source "checkpoint training_state") -ne $pointerCycle -or
        [int64](Get-RequiredJsonProperty `
            -Object $trainingState -Name "step" `
            -Source "checkpoint training_state") -ne $pointerStep
    ) {
        throw "pointer cycle/step disagree with checkpoint training_state"
    }
    if (
        [int](Get-RequiredJsonProperty `
            -Object $inspection -Name "checkpoint_version" `
            -Source "checkpoint inspector") -ne
        [int](Get-RequiredJsonProperty `
            -Object $pointer -Name "checkpoint_version" -Source $pointerPath)
    ) {
        throw "pointer checkpoint_version disagrees with checkpoint report"
    }

    return [ordered]@{
        mode = "resume"
        checkpoint = $checkpointPath
        cycle = $pointerCycle
        step = $pointerStep
        base_checkpoint_id = $pointerBaseId
    }
}

function Get-CheckpointLaunchMode {
    if (-not (Test-Path -LiteralPath $CKPT_ROOT)) {
        return [ordered]@{
            mode = "fresh_start"
            checkpoint = $null
        }
    }

    $rootItem = Get-Item -LiteralPath $CKPT_ROOT
    if (-not $rootItem.PSIsContainer) {
        throw "checkpoint root is not a directory: $CKPT_ROOT"
    }
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "checkpoint root must not be a reparse point: $CKPT_ROOT"
    }
    $entries = @(Get-ChildItem -LiteralPath $CKPT_ROOT -Force)
    if ($entries.Count -eq 0) {
        return [ordered]@{
            mode = "fresh_start"
            checkpoint = $null
        }
    }

    return Resolve-StrictCheckpointHead -CheckpointRoot $CKPT_ROOT
}

function Write-RunMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][object[]]$Arguments,
        [Parameter(Mandatory = $true)][object]$LaunchMode
    )

    $manifest = Get-Content -Raw -LiteralPath $CORPUS_MANIFEST |
        ConvertFrom-Json
    $baseCheckpointId = $null
    if ($LaunchMode.Contains("base_checkpoint_id")) {
        $baseCheckpointId = $LaunchMode["base_checkpoint_id"]
    }
    $payload = [ordered]@{
        schema_version = 1
        run_id = $RunId
        created_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        command = @($VENV) + @($Arguments)
        config_identity = Get-Sha256Hex -LiteralPath $CONFIG
        checkpoint_root = $CKPT_ROOT
        launch_mode = $LaunchMode.mode
        resume_checkpoint = $LaunchMode.checkpoint
        base_checkpoint_id = $baseCheckpointId
        token_source = [ordered]@{
            path = $CORPUS
            bytes = [int64]$manifest.bytes
            tokens = [int64]$manifest.tokens
            sha256_manifest = [string]$manifest.sha256
        }
        holdout = [ordered]@{
            policy = "fixed_tail_excluded_from_training"
            tokens = $HOLDOUT_TOKENS
            batches = $HOLDOUT_BATCHES
            seed = $HOLDOUT_SEED
        }
    }
    $temporary = "$Path.tmp"
    $payload | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$halted = Test-Path -LiteralPath $KILL_SWITCH
if ($halted) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRAINING_HALTED.flag presente em $KILL_SWITCH -- treino bloqueado."
    if (-not $PreflightOnly) {
        exit 1
    }
}

Assert-RequiredFiles
Assert-NoConcurrentTrainer
$initialMode = Get-CheckpointLaunchMode

if ($PreflightOnly) {
    [ordered]@{
        status = "preflight_only"
        training_halted = $halted
        checkpoint_root = $CKPT_ROOT
        launch_mode = $initialMode.mode
        checkpoint = $initialMode.checkpoint
        corpus = $CORPUS
        holdout_tokens = $HOLDOUT_TOKENS
        holdout_batches = $HOLDOUT_BATCHES
        holdout_seed = $HOLDOUT_SEED
    } | ConvertTo-Json -Depth 5
    exit 0
}

# Pinned: recommended_dual_gpu_split() for this config + RTX 5060 Ti
# and RTX 3060 gives layers 0-4 on GPU0 and 5-11 on GPU1.
$env:DUAL_GPU_SPLIT = "5"
New-Item -ItemType Directory -Force -Path $RUN_ROOT | Out-Null
$sessionRunId = "100m-shuffled-v1-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$metricsPath = Join-Path $RUN_ROOT "$sessionRunId.metrics.jsonl"

$cycle = 0
while ($true) {
    if (Test-Path -LiteralPath $KILL_SWITCH) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRAINING_HALTED.flag detectado -- parando loop, nao reiniciando."
        break
    }

    Assert-NoConcurrentTrainer
    $cycle++
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $launchMode = Get-CheckpointLaunchMode
    if ($launchMode.mode -eq "fresh_start") {
        $modeArgs = @("--fresh-start")
        Write-Host "[$ts] CYCLE $cycle | NEW SHUFFLED_V1 LINEAGE | Steps: $STEPS | Save: $SAVE_EVERY"
    } else {
        $modeArgs = @("--resume", [string]$launchMode.checkpoint)
        Write-Host "[$ts] CYCLE $cycle | Strict resume: $($launchMode.checkpoint) | Steps: $STEPS | Save: $SAVE_EVERY"
    }

    $cycleRunId = "$sessionRunId-cycle-$cycle"
    $metadataPath = Join-Path $RUN_ROOT "$cycleRunId.metadata.json"
    $stdoutPath = Join-Path $RUN_ROOT "$cycleRunId.trainer.stdout.log"
    $stderrPath = Join-Path $RUN_ROOT "$cycleRunId.trainer.stderr.log"
    $cmdArgs = @(
        "-u", "-m", "scripts.darwin_organism", "run247",
        "--config", $CONFIG,
        "--token-bin", $CORPUS,
        "--checkpoint-root", $CKPT_ROOT
    ) + $modeArgs + @(
        "--device", "cuda",
        "--block-size", "4096",
        "--batch-size", "1",
        "--sampler-mode", "permuted_blocks",
        "--sampler-version", "1",
        "--steps", "$STEPS",
        "--lr", "3e-4",
        "--optimizer", "adamw",
        "--warmup-steps", "50",
        "--lr-decay-steps", "100000",
        "--lr-final-ratio", "0.1",
        "--causal-mode", "shadow",
        "--causal-v8-migration",
        "--save-every", "$SAVE_EVERY",
        "--metrics-jsonl", $metricsPath,
        "--canary-metadata-json", $metadataPath,
        "--run-id", $cycleRunId,
        "--holdout-tokens", "$HOLDOUT_TOKENS",
        "--holdout-batches", "$HOLDOUT_BATCHES",
        "--holdout-seed", "$HOLDOUT_SEED",
        "--no-autonomous-drives",
        "--no-organ-rl",
        "--blockchain-enabled"
    )
    Write-RunMetadata `
        -Path $metadataPath -RunId $cycleRunId `
        -Arguments $cmdArgs -LaunchMode $launchMode

    if (Test-Path -LiteralPath $KILL_SWITCH) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRAINING_HALTED.flag detectado antes do launch -- abortando."
        break
    }
    $proc = Start-Process -FilePath $VENV -ArgumentList $cmdArgs `
        -WorkingDirectory $ROOT -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($proc.ExitCode -eq 0) {
        Write-Host "[$ts] OK (exit 0). Revalidando pointer/manifest antes do proximo ciclo."
    } else {
        Write-Host "[$ts] CRASH (exit $($proc.ExitCode)). Traceback em: $stderrPath"
        Write-Host "[$ts] O pointer/manifest sera revalidado antes de qualquer retry."
        Start-Sleep -Seconds 5
    }
    Start-Sleep -Seconds 3
}
