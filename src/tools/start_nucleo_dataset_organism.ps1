param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [int]$IntervalSeconds = 900,
    [int]$MaxFilesPerCycle = 25,
    [int]$MaxScanFiles = 2000,
    [int64]$MaxBytesPerCycle = 67108864,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $Workspace "logs"
$stateDir = Join-Path $Workspace "state"
$pidFile = Join-Path $logDir "nucleo_dataset_organism.pid"
New-Item -ItemType Directory -Force -Path $logDir, $stateDir | Out-Null

if (Test-Path $pidFile) {
    $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($oldPid -match '^\d+$') {
        $oldProcess = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($oldProcess) {
            if (-not $Restart) {
                Write-Host "Already running: pid=$oldPid"
                exit 0
            }
            Write-Host "Stopping existing NUCLEO dataset organism: pid=$oldPid"
            Stop-Process -Id ([int]$oldPid) -Force
            Start-Sleep -Seconds 2
        }
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logDir "nucleo_dataset_organism_$stamp.out.log"
$stderrLog = Join-Path $logDir "nucleo_dataset_organism_$stamp.err.log"

$arguments = @(
    "src/tools/nucleo_dataset_organism.py",
    "--apply",
    "--workspace", $Workspace,
    "--max-files-per-cycle", [string]$MaxFilesPerCycle,
    "--max-scan-files", [string]$MaxScanFiles,
    "--max-bytes-per-cycle", [string]$MaxBytesPerCycle,
    "--batch-prefix", "world_seed",
    "--interval-seconds", [string]$IntervalSeconds,
    "--command-timeout-seconds", "900",
    "--local-tokenize-timeout-seconds", "7200",
    "--pid-file", $pidFile
)

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value ([string]$process.Id) -Encoding ascii
Write-Host "Started NUCLEO dataset organism: pid=$($process.Id)"
Write-Host "stdout=$stdoutLog"
Write-Host "stderr=$stderrLog"
