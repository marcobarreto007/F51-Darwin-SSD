param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [int]$IntervalSeconds = 300,
    [int]$StaleHeartbeatMinutes = 45
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $Workspace "logs"
$pidFile = Join-Path $logDir "nucleo_dataset_watchdog.pid"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $pidFile) {
    $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($oldPid -match '^\d+$') {
        $oldProcess = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($oldProcess) {
            Write-Host "Watchdog already running: pid=$oldPid"
            exit 0
        }
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logDir "nucleo_dataset_watchdog_$stamp.out.log"
$stderrLog = Join-Path $logDir "nucleo_dataset_watchdog_$stamp.err.log"

$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Root "src\\tools\\watchdog_nucleo_dataset_organism.ps1"),
    "-Root", $Root,
    "-Workspace", $Workspace,
    "-IntervalSeconds", [string]$IntervalSeconds,
    "-StaleHeartbeatMinutes", [string]$StaleHeartbeatMinutes
)

$process = Start-Process `
    -FilePath "powershell" `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value ([string]$process.Id) -Encoding ascii
Write-Host "Started NUCLEO dataset watchdog: pid=$($process.Id)"
Write-Host "stdout=$stdoutLog"
Write-Host "stderr=$stderrLog"
