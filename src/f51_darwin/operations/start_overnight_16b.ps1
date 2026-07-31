param(
    [string]$Root = "",
    [string]$DatasetRoot = "",
    [string]$Config = "src\\configs\\darwin_x_1.6b_nitro.yaml",
    [string]$Checkpoint = "",
    [int]$StepsPerCycle = 2000,
    [int]$BlockSize = 64,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.00015,
    [int]$EvalEvery = 500,
    [int64]$MinimumModelParameters = 1700000000,
    [int64]$MinimumCheckpointBytes = 8000000000,
    [int]$MinimumFreeDiskGB = 100,
    [int]$MaximumIdleGpuMemoryMB = 2048,
    [switch]$Launch,
    [switch]$Canary,
    [int]$CanarySteps = 250,
    [int64]$CanaryHoldoutTokens = 1048576,
    [int]$PostRunObservationSeconds = 120,
    [string]$CanaryRoot = ""
)

$ErrorActionPreference = "Stop"

function Assert-File {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Write-JsonAtomic {
    param([string]$Path, [object]$Value, [int]$Depth = 8)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-PcieAspmState {
    $activeScheme = (& powercfg /GETACTIVESCHEME 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Unable to query the active power scheme: $activeScheme" }
    $query = (& powercfg /QUERY SCHEME_CURRENT SUB_PCIEXPRESS ASPM 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "Unable to query PCIe ASPM state: $query" }
    # powercfg localizes labels, but its last two hexadecimal values are the
    # current AC and DC indexes respectively.
    $indexes = [regex]::Matches($query, '0x[0-9a-fA-F]{8}')
    if ($indexes.Count -lt 2) { throw "Unable to parse PCIe ASPM AC index." }
    $acHex = $indexes[$indexes.Count - 2].Value
    $acIndex = [Convert]::ToInt32($acHex.Substring(2), 16)
    return [ordered]@{
        active_scheme = $activeScheme
        ac_index = $acIndex
        raw_query = $query.Trim()
        disable_command = "powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0"
        rollback_command = "powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM $acIndex; powercfg /SETACTIVE SCHEME_CURRENT"
    }
}

function Disable-PcieAspmOnAc {
    & powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to disable PCIe ASPM on AC." }
    & powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to reactivate the current power scheme." }
}

function Restore-PcieAspmOnAc {
    param([int]$AcIndex)
    & powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM $AcIndex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to restore PCIe ASPM on AC." }
    & powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to reactivate the current power scheme after ASPM restore." }
}

function Get-WinEventsOrEmpty {
    param([hashtable]$FilterHashtable, [int]$MaxEvents = 0)
    try {
        if ($MaxEvents -gt 0) {
            return @(Get-WinEvent -FilterHashtable $FilterHashtable -MaxEvents $MaxEvents -ErrorAction Stop)
        }
        return @(Get-WinEvent -FilterHashtable $FilterHashtable -ErrorAction Stop)
    } catch {
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {
            return @()
        }
        throw
    }
}

function Get-LastNvidiaEventRecordId {
    $events = @(Get-WinEventsOrEmpty -FilterHashtable @{
        LogName = 'System'; ProviderName = 'nvlddmkm'; Id = 13, 14, 153
    } -MaxEvents 1)
    return $(if ($events.Count -eq 0) { [int64]0 } else { [int64]$events[0].RecordId })
}

function Get-NvidiaEventsSince {
    param([int64]$AfterRecordId)
    return @(
        Get-WinEventsOrEmpty -FilterHashtable @{
            LogName = 'System'; ProviderName = 'nvlddmkm'; Id = 13, 14, 153
        } |
            Where-Object { [int64]$_.RecordId -gt $AfterRecordId } |
            ForEach-Object {
                [ordered]@{
                    id = [int]$_.Id
                    record_id = [int64]$_.RecordId
                    time_utc = $_.TimeCreated.ToUniversalTime().ToString('o')
                    message = $_.Message
                }
            }
    )
}

function Get-LiveKernel141Since {
    param([datetime]$Since)
    return @(
        Get-WinEventsOrEmpty -FilterHashtable @{
            LogName = 'Application'
            ProviderName = 'Windows Error Reporting'
            Id = 1001
            StartTime = $Since
        } |
            Where-Object {
                $_.Message -match 'LiveKernelEvent' -and
                $_.Message -match '(?m)^P1:\s*141\s*$'
            } |
            ForEach-Object {
                [ordered]@{
                    record_id = [int64]$_.RecordId
                    time_utc = $_.TimeCreated.ToUniversalTime().ToString('o')
                    message = $_.Message
                }
            }
    )
}

function Get-GpuSample {
    $rows = @(
        & "C:\Windows\System32\nvidia-smi.exe" `
            --query-gpu=index,name,pci.bus_id,utilization.gpu,memory.used,temperature.gpu,power.draw `
            --format=csv,noheader,nounits
    )
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi monitoring sample failed." }
    return @($rows | ForEach-Object {
        $parts = $_ -split ',\s*'
        [ordered]@{
            index = [int]$parts[0]
            name = $parts[1]
            pci_bus_id = $parts[2]
            utilization_gpu_percent = [double]$parts[3]
            memory_used_mb = [double]$parts[4]
            temperature_c = [double]$parts[5]
            power_draw_w = [double]$parts[6]
        }
    })
}

if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) }
$Root = (Resolve-Path -LiteralPath $Root).Path
$srcRoot = Join-Path $Root "src"
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$srcRoot;$env:PYTHONPATH" } else { $srcRoot }
$datasetRootWasExplicit = -not [string]::IsNullOrWhiteSpace($DatasetRoot)
if (-not $DatasetRoot) { $DatasetRoot = Join-Path $Root 'workspace' }
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$python = Join-Path $Root ".venv_nitro\Scripts\python.exe"
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
$tokenBin = Join-Path $DatasetRoot "01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast_v2.bin"
$tokenManifestPath = "$tokenBin.manifest.json"
$checkpointRoot = Join-Path $DatasetRoot "03_CHECKPOINTS"
$manifestRoot = Join-Path $DatasetRoot "04_MANIFESTOS"
$latestPointerPath = Join-Path $checkpointRoot "organism_latest.json"
$goldLineagePath = Join-Path $manifestRoot "gold_local_lineage.json"

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCommand) { throw "git executable is required on PATH" }
$gitExe = $gitCommand.Source
$sourceCommit = (& $gitExe -C $Root rev-parse HEAD 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $sourceCommit) { throw "Unable to resolve source commit." }
$trackedStatus = (& $gitExe -C $Root status --porcelain --untracked-files=no 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to verify tracked worktree state: $trackedStatus" }
if ($trackedStatus) { throw "Tracked worktree must be clean before readiness: $trackedStatus" }

Assert-File $python "Training Python"
Assert-File $configPath "Canonical 1.6B config"
Assert-File $tokenBin "External token corpus"
Assert-File $tokenManifestPath "Token manifest"
Assert-File $latestPointerPath "Checkpoint pointer"
Assert-File $goldLineagePath "Verified GOLD lineage manifest"

$goldLineage = Get-Content -LiteralPath $goldLineagePath -Raw | ConvertFrom-Json
if ([string]$goldLineage.status -ne "ok") {
    throw "GOLD lineage is not ready: status=$($goldLineage.status) blockers=$($goldLineage.blockers -join ',')"
}
if ([string]$goldLineage.source_commit -ne $sourceCommit) {
    throw "GOLD lineage source commit is stale: expected=$sourceCommit observed=$($goldLineage.source_commit)"
}

$existing = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match 'darwin_organism\.py\s+(run247|cycle.*--canary)' -and
            $_.ProcessId -ne $PID
        }
)
if ($existing.Count -gt 0) {
    throw "A Darwin training or canary process is already active (PID: $($existing.ProcessId -join ','))."
}

$nitroRuntimeText = & $python -m tools.check_nitro_runtime --root $Root --require-hardware 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "Nitro runtime provenance check failed: $nitroRuntimeText"
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
$tokenSha256 = (Get-FileHash -LiteralPath $tokenBin -Algorithm SHA256).Hash.ToLowerInvariant()
if ($tokenSha256 -ne ([string]$tokenManifest.sha256).ToLowerInvariant()) {
    throw "Token corpus SHA-256 mismatch: expected=$($tokenManifest.sha256) observed=$tokenSha256"
}

$modelText = & $python -m tools.inspect_darwin_x `
    --config $configPath --instantiate-meta 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "Model config inspection failed: $modelText"
}
$modelReport = $modelText | ConvertFrom-Json
$modelParameters = [int64]$modelReport.meta_instantiated_params
if ($modelParameters -lt $MinimumModelParameters -or $modelParameters -ge 2400000000) {
    throw "Config is outside the canonical local 1.6B/1.76B class: $modelParameters parameters."
}

$latestPointer = Get-Content -LiteralPath $latestPointerPath -Raw | ConvertFrom-Json
if (-not $Checkpoint) {
    $Checkpoint = [string]$latestPointer.path
}
$checkpointPath = if ([IO.Path]::IsPathRooted($Checkpoint)) {
    $Checkpoint
} else {
    Join-Path $checkpointRoot $Checkpoint
}
Assert-File $checkpointPath "Canonical 1.6B checkpoint"
$checkpointInfo = Get-Item -LiteralPath $checkpointPath
if ($checkpointInfo.Length -lt $MinimumCheckpointBytes) {
    throw "Checkpoint is too small or partial: bytes=$($checkpointInfo.Length)"
}
if ($latestPointer.path -eq $checkpointInfo.Name) {
    if ([int64]$latestPointer.size_bytes -ne $checkpointInfo.Length) {
        throw "Latest pointer size disagrees with checkpoint."
    }
}
$checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $latestPointer.sha256 -or $checkpointSha256 -ne ([string]$latestPointer.sha256).ToLowerInvariant()) {
    throw "Latest pointer SHA-256 disagrees with checkpoint."
}

$checkpointText = & $python -m scripts.inspect_organism_checkpoint `
    $checkpointPath --config $configPath --verify-identity 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint inspection failed: $checkpointText"
}
$checkpointReport = $checkpointText | ConvertFrom-Json
if ([int]$checkpointReport.checkpoint_version -lt 7) {
    throw "Overnight launch requires the migrated v7 checkpoint."
}
if (-not [bool]$checkpointReport.embedded_config_matches_file -or
    -not [bool]$checkpointReport.resume_shape_compatible) {
    throw "Checkpoint config or anatomy does not match the canonical 1.6B runtime."
}
if (-not [bool]$checkpointReport.identity_verified) {
    throw "Checkpoint content identity failed full recomputation."
}

$gpuRows = @(
    & "C:\Windows\System32\nvidia-smi.exe" `
        --query-gpu=index,name,pci.bus_id,memory.used,memory.total `
        --format=csv,noheader,nounits |
        ForEach-Object {
            $parts = $_ -split ',\s*'
            [pscustomobject]@{
                index = [int]$parts[0]
                name = $parts[1]
                pci_bus_id = $parts[2]
                memory_used_mb = [int]$parts[3]
                memory_total_mb = [int]$parts[4]
            }
        }
)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -lt 2) {
    throw "Two NVIDIA GPUs are required; detected $($gpuRows.Count)."
}
$busy = @($gpuRows | Where-Object { $_.memory_used_mb -gt $MaximumIdleGpuMemoryMB })
if ($busy.Count -gt 0) {
    throw "GPU memory is not idle enough for launch: $($busy | ConvertTo-Json -Compress)"
}

$drive = [IO.DriveInfo]::new((Get-Item -LiteralPath $checkpointRoot).PSDrive.Root)
$freeDiskGB = [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
if ($freeDiskGB -lt $MinimumFreeDiskGB) {
    throw "Insufficient disk headroom for unattended checkpoints: $freeDiskGB GB free."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$baseCheckpointSha256 = $null
$pcieAspmBefore = $null
$startingNvidiaRecordId = [int64]0
if ($Canary) {
    if ($CanarySteps -ne 250) { throw "Recovery canary is fixed at 250 steps." }
    if (-not $CanaryRoot) {
        $CanaryRoot = Join-Path $checkpointRoot "canary_recovery_$stamp"
    }
    $CanaryRoot = [IO.Path]::GetFullPath($CanaryRoot)
    $canonicalRootFull = [IO.Path]::GetFullPath($checkpointRoot)
    $canaryRootPrefix = $canonicalRootFull.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    if (-not $CanaryRoot.StartsWith($canaryRootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Canary root must be a child of the canonical checkpoint root."
    }
    $isolatedCanaryPrefix = $CanaryRoot.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    $baseCheckpointSha256 = $checkpointSha256
    $pcieAspmBefore = Get-PcieAspmState
    $startingNvidiaRecordId = Get-LastNvidiaEventRecordId
}

$readiness = [ordered]@{
    ready = $true
    checked_at = (Get-Date).ToUniversalTime().ToString("o")
    local_lineage = "F51-Darwin-X-1.6B-Nitro"
    source_commit = $sourceCommit
    tracked_worktree_clean = $true
    override_provenance = $(if ($datasetRootWasExplicit) { "DatasetRoot_parameter" } else { "project_workspace_default" })
    project_root = $Root
    dataset_root = $DatasetRoot
    config = $configPath
    config_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
    model_name = $modelReport.model_name
    model_parameters = $modelParameters
    checkpoint = $checkpointPath
    checkpoint_version = [int]$checkpointReport.checkpoint_version
    checkpoint_bytes = $checkpointInfo.Length
    checkpoint_sha256 = $checkpointSha256
    checkpoint_base_id = if ($checkpointReport.identity) { $checkpointReport.identity.declared } else { $checkpointReport.base_checkpoint_id }
    checkpoint_cycle = [int]$checkpointReport.training_state.cycle
    checkpoint_step = [int]$checkpointReport.training_state.step
    token_bin = $tokenBin
    token_bytes = $tokenInfo.Length
    token_count = [int64]$tokenManifest.tokens
    token_sha256_manifest = $tokenManifest.sha256
    token_sha256_observed = $tokenSha256
    tokenizer_identity = $goldLineage.tokenizer.identity
    gpus = $gpuRows
    free_disk_gb = $freeDiskGB
    preserve_all_checkpoints = $true
    stop_before_disk_exhaustion = $true
    organs = @("heartbeat", "ghost", "jepa", "curiosity", "spider", "mtp", "expert_pool", "legacy", "lineage")
    block_size = $BlockSize
    batch_size = $BatchSize
    steps_per_cycle = $StepsPerCycle
    eval_every = $EvalEvery
    learning_rate = $LearningRate
    mode = $(if ($Canary) { "recovery_canary" } else { "run247" })
    canary = $(if ($Canary) {
        [ordered]@{
            steps = $CanarySteps
            holdout_tokens = $CanaryHoldoutTokens
            holdout_batches = 16
            holdout_seed = 999
            isolated_checkpoint_root = $CanaryRoot
            base_checkpoint_sha256 = $baseCheckpointSha256
            starting_nvidia_event_record_id = $startingNvidiaRecordId
            pcie_aspm_before = $pcieAspmBefore
            planned_power_change = $pcieAspmBefore.disable_command
            post_run_observation_seconds = $PostRunObservationSeconds
            automatic_run247 = $false
            automatic_promotion = $false
        }
    } else { $null })
}

New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null
$readinessPath = Join-Path $manifestRoot "overnight_16b_readiness.json"
Write-JsonAtomic -Path $readinessPath -Value $readiness -Depth 8

if (-not $Launch) {
    $readiness | ConvertTo-Json -Depth 6
    if ($Canary) {
        Write-Host "CANARY DRY RUN OK. Review the manifest, then add -Launch for one isolated 250-step cycle."
    } else {
        Write-Host "DRY RUN OK. Add -Launch to start the canonical local 1.6B lineage."
    }
    exit 0
}

$launchCommit = (& $gitExe -C $Root rev-parse HEAD 2>&1 | Out-String).Trim()
$launchTrackedStatus = (& $gitExe -C $Root status --porcelain --untracked-files=no 2>&1 | Out-String).Trim()
if ($launchCommit -ne $readiness.source_commit -or $launchTrackedStatus) {
    throw "Launch inputs changed after readiness: commit=$launchCommit tracked=$launchTrackedStatus"
}
if ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -ne $readiness.config_sha256 -or
    (Get-FileHash -LiteralPath $tokenBin -Algorithm SHA256).Hash.ToLowerInvariant() -ne $tokenSha256 -or
    (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checkpointSha256) {
    throw "Launch artifact identities changed after readiness."
}

$runRoot = Join-Path $DatasetRoot "runtime\runs\overnight_16b"
$runDir = Join-Path $runRoot $stamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$stdout = Join-Path $runDir "train.stdout.log"
$stderr = Join-Path $runDir "train.stderr.log"
$pidPath = Join-Path $runDir "train.pid"
$launchManifest = Join-Path $runDir "launch_manifest.json"

# ── Canary path isolation ──
if ($Canary) {
    $metricsJsonl = Join-Path $runDir "metrics.jsonl"
    $runtimeMetadata = Join-Path $runDir "runtime_metadata.json"
    $arguments = @(
        "-u", "-m", "scripts.darwin_organism", "cycle", "--canary",
        "--config", $configPath,
        "--token-bin", $tokenBin,
        "--resume", $checkpointPath,
        "--cycles", "1",
        "--steps", [string]$CanarySteps,
        "--block-size", [string]$BlockSize,
        "--batch-size", [string]$BatchSize,
        "--lr", [string]$LearningRate,
        "--device", "cuda",
        "--eval-every", [string]$EvalEvery,
        "--checkpoint-root", $CanaryRoot,
        "--metrics-jsonl", $metricsJsonl,
        "--canary-metadata-json", $runtimeMetadata,
        "--run-id", $stamp,
        "--holdout-tokens", [string]$CanaryHoldoutTokens,
        "--holdout-batches", "16",
        "--holdout-seed", "999",
        "--require-clean-worktree",
        "--base-checkpoint-sha256", $baseCheckpointSha256
    )

    $gitCommit = $sourceCommit
    $nvidiaDriver = (
        & "C:\Windows\System32\nvidia-smi.exe" `
            --query-gpu=driver_version --format=csv,noheader 2>&1 |
            Select-Object -First 1 | Out-String
    ).Trim()
    $runtimeMetadataPayload = [ordered]@{
        git_commit = $gitCommit
        base_checkpoint_sha256 = $baseCheckpointSha256
        command = @($python) + $arguments
        config_identity = $readiness.config_sha256
        driver = $nvidiaDriver
        gpus = @($gpuRows)
        starting_nvidia_event_record_id = $startingNvidiaRecordId
        token_source = [ordered]@{
            path = $tokenBin
            bytes = $tokenInfo.Length
            tokens = [int64]$tokenManifest.tokens
            sha256_manifest = $tokenManifest.sha256
        }
    }
    Write-JsonAtomic -Path $runtimeMetadata -Value $runtimeMetadataPayload
} else {
    $arguments = @(
        "-u",
        "-m",
        "scripts.darwin_organism",
        "run247",
        "--config", $configPath,
        "--token-bin", $tokenBin,
        "--resume", $checkpointPath,
        "--steps", [string]$StepsPerCycle,
        "--block-size", [string]$BlockSize,
        "--batch-size", [string]$BatchSize,
        "--lr", [string]$LearningRate,
        "--device", "cuda",
        "--eval-every", [string]$EvalEvery
    )
}

$env:F51_DATASET_ROOT = $DatasetRoot
$pcieAspmChanged = $false
try {
if ($Canary) {
    # The only environmental mutation allowed by the recovery design. The
    # exact previous value and rollback command are already in the manifest.
    Disable-PcieAspmOnAc
    $pcieAspmChanged = $true
    $readiness.pcie_aspm_changed_at = (Get-Date).ToUniversalTime().ToString("o")
}
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
$readiness.run_dir = $runDir
$readiness.stdout = $stdout
$readiness.stderr = $stderr
$readiness | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchManifest -Encoding utf8
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
[ordered]@{
    run_dir = $runDir
    pid = $process.Id
    started_at = $readiness.started_at
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot "latest.json") -Encoding utf8

# ── Canary: monitor until exit, then observe ──
if ($Canary) {
    Write-Host "Canary 1.6B launched. PID=$($process.Id) - monitoring..."
    $launchTime = Get-Date
    $gpuSamples = [Collections.Generic.List[object]]::new()
    $monitorErrors = [Collections.Generic.List[object]]::new()
    while (-not $process.HasExited) {
        try {
            $gpuSamples.Add([ordered]@{
                timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
                rows = @(Get-GpuSample)
            })
        } catch {
            $monitorErrors.Add([ordered]@{
                timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
                source = 'gpu_sample'
                error = $_.Exception.Message
            })
        }
        Start-Sleep -Seconds 5
        $process.Refresh()
    }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    Write-Host "Canary process exited with code $exitCode. Observing for $PostRunObservationSeconds seconds..."
    Start-Sleep -Seconds $PostRunObservationSeconds
    try {
        $newNvidiaEvents = @(Get-NvidiaEventsSince -AfterRecordId $startingNvidiaRecordId)
    } catch {
        $newNvidiaEvents = @()
        $monitorErrors.Add([ordered]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
            source = 'nvidia_event_log'
            error = $_.Exception.Message
        })
    }
    try {
        $liveKernel141 = @(Get-LiveKernel141Since -Since $launchTime)
    } catch {
        $liveKernel141 = @()
        $monitorErrors.Add([ordered]@{
            timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
            source = 'live_kernel_event_log'
            error = $_.Exception.Message
        })
    }

    $peakTemperatures = @(
        foreach ($gpu in $gpuRows) {
            $temperatures = @(
                $gpuSamples | ForEach-Object { $_.rows } |
                    Where-Object { [int]$_.index -eq [int]$gpu.index } |
                    ForEach-Object { [double]$_.temperature_c }
            )
            if ($temperatures.Count -gt 0) {
                [double](($temperatures | Measure-Object -Maximum).Maximum)
            }
        }
    )

    $candidatePointerPath = Join-Path $CanaryRoot "organism_latest.json"
    $candidatePath = $null
    $candidateCheckpointSha256 = $null
    $checkpointVerified = $false
    $checkpointInspection = $null
    if ($exitCode -eq 0 -and (Test-Path -LiteralPath $candidatePointerPath -PathType Leaf)) {
        try {
            $candidatePointer = Get-Content -LiteralPath $candidatePointerPath -Raw | ConvertFrom-Json
            $candidatePath = if ([IO.Path]::IsPathRooted([string]$candidatePointer.path)) {
                [string]$candidatePointer.path
            } else {
                Join-Path $CanaryRoot ([string]$candidatePointer.path)
            }
            $candidatePath = [IO.Path]::GetFullPath($candidatePath)
            if (-not $candidatePath.StartsWith($isolatedCanaryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Candidate checkpoint escaped the isolated canary root.'
            }
            if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
                $candidateInfo = Get-Item -LiteralPath $candidatePath
                $candidateCheckpointSha256 = (
                    Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256
                ).Hash
                $inspectionText = & $python -m scripts.inspect_organism_checkpoint `
                    $candidatePath --config $configPath --verify-identity 2>&1 | Out-String
                $inspectionExitCode = $LASTEXITCODE
                try {
                    $checkpointInspection = $inspectionText | ConvertFrom-Json
                } catch {
                    throw "Candidate inspector returned invalid JSON: $inspectionText"
                }
                $pointerMatchesCheckpoint = (
                    [int64]$candidatePointer.size_bytes -eq [int64]$candidateInfo.Length -and
                    [int]$candidatePointer.cycle -eq [int]$checkpointInspection.training_state.cycle -and
                    [int]$candidatePointer.step -eq [int]$checkpointInspection.training_state.step -and
                    [string]$candidatePointer.base_checkpoint_id -eq [string]$checkpointInspection.base_checkpoint_id
                )
                $checkpointVerified = (
                    $inspectionExitCode -eq 0 -and
                    [bool]$checkpointInspection.strict_resume_compatible -and
                    [bool]$checkpointInspection.topology_manifest_valid -and
                    [bool]$checkpointInspection.optimizer_resume_compatible -and
                    [bool]$checkpointInspection.identity_verified -and
                    $pointerMatchesCheckpoint -and
                    [int]$checkpointInspection.training_state.cycle -eq ([int]$checkpointReport.training_state.cycle + 1) -and
                    [int]$checkpointInspection.training_state.step -gt [int]$checkpointReport.training_state.step
                )
            }
        } catch {
            $checkpointVerified = $false
            $checkpointInspection = [ordered]@{ error = $_.Exception.Message }
            $monitorErrors.Add([ordered]@{
                timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
                source = 'candidate_checkpoint_inspection'
                error = $_.Exception.Message
            })
        }
    }

    $metricRecords = @()
    $metricsTruncatedTail = $false
    if (Test-Path -LiteralPath $metricsJsonl -PathType Leaf) {
        $metricsBytes = [IO.File]::ReadAllBytes($metricsJsonl)
        $metricsTruncatedTail = ($metricsBytes.Length -gt 0 -and $metricsBytes[-1] -ne 10)
        $metricLines = @(Get-Content -LiteralPath $metricsJsonl)
        for ($lineIndex = 0; $lineIndex -lt $metricLines.Count; $lineIndex++) {
            $line = $metricLines[$lineIndex]
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $metricRecords += ($line | ConvertFrom-Json)
            } catch {
                if ($metricsTruncatedTail -and $lineIndex -eq ($metricLines.Count - 1)) {
                    break
                }
                $monitorErrors.Add([ordered]@{
                    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
                    source = 'metrics_jsonl'
                    error = "Invalid JSONL record at line $($lineIndex + 1): $($_.Exception.Message)"
                })
                break
            }
        }
    }
    $baselineRecord = $metricRecords | Where-Object { $_.phase -eq 'baseline' } | Select-Object -Last 1
    $candidateRecord = $metricRecords | Where-Object { $_.phase -eq 'candidate' } | Select-Object -Last 1
    $candidateMetrics = $candidateRecord.metrics
    $gateInputPath = Join-Path $runDir "canary_gate_input.json"
    $gateInput = [ordered]@{
        expected_steps = 250
        successful_updates = $(if ($null -eq $candidateMetrics) { 0 } else { [int]$candidateMetrics.successful_updates })
        skipped_updates = $(if ($null -eq $candidateMetrics) { 250 } else { [int]$candidateMetrics.skipped_updates })
        fresh = $(if ($null -eq $candidateMetrics) { @{} } else { $candidateMetrics.fresh })
        replay = $(if ($null -eq $candidateMetrics) { @{} } else { $candidateMetrics.replay })
        replay_fraction = $(if ($null -eq $candidateMetrics) { 0.0 } else { [double]$candidateMetrics.replay_fraction })
        base_heldout_loss = $(if ($null -eq $baselineRecord) { $null } else { $baselineRecord.heldout.lm_loss })
        candidate_heldout_loss = $(if ($null -eq $candidateRecord) { $null } else { $candidateRecord.heldout.lm_loss })
        peak_temperatures_c = @($peakTemperatures)
        new_nvidia_events = @($newNvidiaEvents)
        live_kernel_event_141 = ($liveKernel141.Count -gt 0)
        checkpoint_verified = $checkpointVerified
        external_benchmark_passed = $false
    }
    Write-JsonAtomic -Path $gateInputPath -Value $gateInput
    $decisionText = & $python -m f51_darwin.training_observability `
        --gate-json $gateInputPath 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { throw "Canary acceptance evaluation failed: $decisionText" }
    $decision = $decisionText | ConvertFrom-Json

    $finalReportPath = Join-Path $runDir "canary_final_report.json"
    $finalReport = [ordered]@{
        schema_version = 1
        run_id = $stamp
        process_exit_code = $exitCode
        launch_time_utc = $launchTime.ToUniversalTime().ToString('o')
        observation_completed_utc = (Get-Date).ToUniversalTime().ToString('o')
        post_run_observation_seconds = $PostRunObservationSeconds
        base_checkpoint = $checkpointPath
        base_checkpoint_sha256 = $baseCheckpointSha256
        candidate_checkpoint = $candidatePath
        candidate_checkpoint_sha256 = $candidateCheckpointSha256
        candidate_pointer = $candidatePointerPath
        checkpoint_verified = $checkpointVerified
        checkpoint_inspection = $checkpointInspection
        pcie_aspm_before = $pcieAspmBefore
        gpu_samples = @($gpuSamples)
        peak_temperatures_c = @($peakTemperatures)
        new_nvidia_events = @($newNvidiaEvents)
        live_kernel_141_events = @($liveKernel141)
        monitor_errors = @($monitorErrors)
        metrics_jsonl = $metricsJsonl
        metrics_truncated_tail = $metricsTruncatedTail
        gate_input = $gateInput
        decision = $decision
        awaiting_external_benchmark = $true
        canonical_pointer_untouched = $latestPointerPath
    }
    Write-JsonAtomic -Path $finalReportPath -Value $finalReport -Depth 12
    (Get-FileHash -LiteralPath $finalReportPath -Algorithm SHA256).Hash |
        Set-Content -LiteralPath "$finalReportPath.sha256" -Encoding ascii

    $blockingFailures = @($decision.failures | Where-Object { $_ -ne 'external_benchmark' })
    if ($monitorErrors.Count -gt 0) { $blockingFailures += 'monitoring_error' }
    if ($exitCode -ne 0 -or $blockingFailures.Count -gt 0) {
        throw "Canary failed closed. exit=$exitCode failures=$($blockingFailures -join ',') report=$finalReportPath"
    }
    Write-Host "Canary runtime gates passed; external benchmark remains mandatory. report=$finalReportPath"
} else {
    Write-Host "Overnight 1.6B launch started. PID=$($process.Id)"
}
Write-Host "stdout=$stdout"
Write-Host "stderr=$stderr"
} finally {
    if ($pcieAspmChanged -and $null -ne $pcieAspmBefore) {
        Restore-PcieAspmOnAc -AcIndex ([int]$pcieAspmBefore.ac_index)
    }
}
