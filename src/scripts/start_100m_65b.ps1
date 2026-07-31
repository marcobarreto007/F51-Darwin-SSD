[CmdletBinding()]
param(
    [string]$Root = "",
    [switch]$Canary,
    [switch]$Launch,
    [switch]$Worker,
    [string]$SourceCommit = "",
    [int]$MinimumFreeDiskGB = 50
)

$ErrorActionPreference = "Stop"
$MaxTrainTokens = [int64]65000000000 # 65_000_000_000
$CanarySteps = 250
$BlockSize = 4096
$BatchSize = 1
$ConfigRelative = "src\\configs\\darwin_x_100m.yaml"
$FamilyRelative = "workspace\03_CHECKPOINTS_100M_FULL_V1"
$ProductionRelative = "$FamilyRelative\production"
$CorpusRelative = "workspace\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"

if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }
$Root = [IO.Path]::GetFullPath($Root)
$env:PYTHONPATH = Join-Path $Root "src"
$Python = Join-Path $Root ".venv_nitro\Scripts\python.exe"
$Config = Join-Path $Root $ConfigRelative
$FamilyRoot = Join-Path $Root $FamilyRelative
$ProductionRoot = Join-Path $Root $ProductionRelative
$Corpus = Join-Path $Root $CorpusRelative
$CorpusManifest = "$Corpus.manifest.json"
$RunRoot = Join-Path $Root "workspace\runtime\runs\100m-65b"
$ScriptPath = $MyInvocation.MyCommand.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
function Write-JsonAtomic {
    param([string]$Path, [object]$Value)
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-SourceCommit {
    $commit = (& git -C $Root rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch "^[0-9a-f]{40}$") {
        throw "git source commit is unavailable"
    }
    return $commit
}

function Assert-Preflight {
    param([switch]$RequireEmptyProduction)

    foreach ($path in @($Python, $Config, $Corpus, $CorpusManifest)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "required file is missing: $path"
        }
    }

    $tracked = @(& git -C $Root status --porcelain --untracked-files=no)
    if ($LASTEXITCODE -ne 0) { throw "git status failed" }
    if ($tracked.Count -gt 0) {
        throw "Tracked worktree must be clean: $($tracked -join '; ')"
    }

    $trainers = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "^python(w)?\.exe$" -and
                $_.CommandLine -match "darwin_organism|train-budget"
            }
    )
    if ($trainers.Count -gt 0) {
        throw "another Darwin trainer is active: $($trainers.ProcessId -join ',')"
    }

    $gpuRows = @(& nvidia-smi --query-gpu=index --format=csv,noheader)
    if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -lt 2) {
        throw "two CUDA GPUs are required"
    }

    $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($Root).Substring(0, 1))
    $freeGB = [math]::Floor($drive.Free / 1GB)
    if ($freeGB -lt $MinimumFreeDiskGB) {
        throw "free disk is below MinimumFreeDiskGB=$MinimumFreeDiskGB (actual=$freeGB)"
    }

    if ($RequireEmptyProduction -and (Test-Path -LiteralPath $ProductionRoot)) {
        $entries = @(Get-ChildItem -LiteralPath $ProductionRoot -Force)
        if ($entries.Count -gt 0) {
            throw "fresh production checkpoint root is not empty: $ProductionRoot"
        }
    }
}

function Invoke-CanaryMode {
    param([string]$Mode, [string]$RunId)

    $canaryRoot = Join-Path $FamilyRoot "canary\$RunId-$Mode"
    $metrics = Join-Path $RunRoot "$RunId-$Mode.metrics.jsonl"
    $metadata = Join-Path $RunRoot "$RunId-$Mode.metadata.json"
    $stdout = Join-Path $RunRoot "$RunId-$Mode.stdout.log"
    $stderr = Join-Path $RunRoot "$RunId-$Mode.stderr.log"
    if (Test-Path -LiteralPath $canaryRoot) {
        throw "canary root already exists: $canaryRoot"
    }

    $arguments = @(
        "-u", "-m", "scripts.darwin_organism", "cycle",
        "--config", $Config,
        "--token-bin", $Corpus,
        "--fresh-start",
        "--device", "cuda",
        "--block-size", "$BlockSize",
        "--batch-size", "$BatchSize",
        "--steps", "$CanarySteps",
        "--cycles", "1",
        "--lr", "0.00015",
        "--optimizer", "dae_hybrid",
        "--warmup-steps", "100",
        "--eval-every", "250",
        "--causal-mode", $Mode,
        "--causal-v8-migration",
        "--canary",
        "--checkpoint-root", $canaryRoot,
        "--metrics-jsonl", $metrics,
        "--canary-metadata-json", $metadata,
        "--run-id", "$RunId-$Mode",
        "--holdout-tokens", "65536",
        "--holdout-batches", "4",
        "--require-clean-worktree",
        "--source-git-commit", $SourceCommit
    )

    $tokenManifest = Get-Content -Raw -LiteralPath $CorpusManifest |
        ConvertFrom-Json
    $gpuRows = @(
        & nvidia-smi `
            --query-gpu=index,uuid,name,memory.total `
            --format=csv,noheader,nounits
    )
    if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -lt 2) {
        throw "failed to capture dual-GPU canary metadata"
    }
    $driver = (
        & nvidia-smi --query-gpu=driver_version --format=csv,noheader |
            Select-Object -First 1
    ).Trim()
    $metadataPayload = [ordered]@{
        git_commit = $SourceCommit
        base_checkpoint_sha256 = $null
        command = @($Python) + $arguments
        config_identity = (
            Get-FileHash -LiteralPath $Config -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        driver = $driver
        gpus = $gpuRows
        starting_nvidia_event_record_id = $null
        token_source = [ordered]@{
            path = $Corpus
            bytes = (Get-Item -LiteralPath $Corpus).Length
            tokens = [int64]$tokenManifest.tokens
            sha256_manifest = $tokenManifest.sha256
        }
    }
    Write-JsonAtomic -Path $metadata -Value $metadataPayload

    $process = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $observedGpuUuids = @{}
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 5
        $rows = @(
            & nvidia-smi --query-compute-apps=pid,gpu_uuid `
                --format=csv,noheader,nounits 2>$null
        )
        foreach ($row in $rows) {
            $parts = $row -split ","
            if ($parts.Count -ge 2 -and $parts[0].Trim() -eq "$($process.Id)") {
                $observedGpuUuids[$parts[1].Trim()] = $true
            }
        }
        $process.Refresh()
    }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        throw "canary $Mode failed with exit code $exitCode; see $stderr"
    }

    $combinedLog = (Get-Content -Raw -LiteralPath $stdout) + "`n" +
        (Get-Content -Raw -LiteralPath $stderr)
    if ($combinedLog -notmatch "DUAL GPU") {
        throw "canary $Mode did not report dual-GPU placement"
    }
    if ($combinedLog -match '"loss_end"\s*:\s*(NaN|Infinity|-Infinity)') {
        throw "canary $Mode produced non-finite loss"
    }
    if ($combinedLog -notmatch '"loss_end"') {
        throw "canary $Mode did not publish loss_end"
    }

    $checkpoint = Join-Path $canaryRoot "organism_cycle_001.pt"
    $inspectionText = & $Python -m scripts.inspect_organism_checkpoint `
        $checkpoint --config $Config --verify-identity
    if ($LASTEXITCODE -ne 0) { throw "strict checkpoint inspection failed for $Mode" }
    $inspection = $inspectionText | ConvertFrom-Json
    if (-not $inspection.strict_resume_compatible) {
        throw "strict_resume_compatible is false for $Mode"
    }

    return [ordered]@{
        mode = $Mode
        process_id = $process.Id
        checkpoint = $checkpoint
        strict_resume_compatible = $true
        observed_gpu_uuids = @($observedGpuUuids.Keys)
        dual_gpu_log = $true
    }
}

function Invoke-AllCanaries {
    $runId = "canary-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $results = @()
    foreach ($mode in @("control", "shadow", "enforce")) {
        $results += Invoke-CanaryMode -Mode $mode -RunId $runId
    }
    $report = [ordered]@{
        status = "passed"
        source_commit = $SourceCommit
        run_id = $runId
        results = $results
    }
    Write-JsonAtomic -Path (Join-Path $RunRoot "$runId.report.json") -Value $report
    return $report
}

if ($Worker) {
    if (-not $SourceCommit) { throw "worker requires SourceCommit" }
    Assert-Preflight -RequireEmptyProduction
    if ((Get-SourceCommit) -ne $SourceCommit) { throw "source_commit changed" }
    Invoke-AllCanaries | ConvertTo-Json -Depth 10
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ProductionRoot) | Out-Null
    & $Python -u -m scripts.darwin_organism train-budget `
        --config $Config --token-bin $Corpus --fresh-start `
        --checkpoint-root $ProductionRoot --device cuda `
        --block-size $BlockSize --batch-size $BatchSize --steps 500 `
        --max-train-tokens $MaxTrainTokens --lr 0.00015 `
        --optimizer dae_hybrid --warmup-steps 100 --eval-every 250 `
        --causal-mode enforce --causal-v8-migration
    exit $LASTEXITCODE
}

if ($Canary -and $Launch) { throw "choose either -Canary or -Launch" }
if (-not $Canary -and -not $Launch) { throw "choose -Canary or -Launch" }
$SourceCommit = Get-SourceCommit
Assert-Preflight -RequireEmptyProduction:$Launch
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

if ($Canary) {
    Invoke-AllCanaries | ConvertTo-Json -Depth 10
    exit 0
}

$supervisorOut = Join-Path $RunRoot "supervisor.stdout.log"
$supervisorErr = Join-Path $RunRoot "supervisor.stderr.log"
$workerArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath,
    "-Root", $Root, "-Worker", "-SourceCommit", $SourceCommit,
    "-MinimumFreeDiskGB", "$MinimumFreeDiskGB"
)
$supervisor = Start-Process -FilePath "powershell.exe" -ArgumentList $workerArgs `
    -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $supervisorOut -RedirectStandardError $supervisorErr
$launchManifest = [ordered]@{
    schema_version = 1
    status = "canaries_starting"
    source_commit = $SourceCommit
    process_id = $supervisor.Id
    max_train_tokens = $MaxTrainTokens
    config = $Config
    production_checkpoint_root = $ProductionRoot
    stdout = $supervisorOut
    stderr = $supervisorErr
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
}
Write-JsonAtomic -Path (Join-Path $RunRoot "launch.json") -Value $launchManifest
$launchManifest | ConvertTo-Json -Depth 10
