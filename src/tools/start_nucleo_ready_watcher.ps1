param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $Workspace "logs"
$pidFile = Join-Path $logDir "nucleo_ready_watcher.pid"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $pidFile) {
    $oldPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($oldPid -match '^\d+$') {
        $oldProcess = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($oldProcess) {
            Write-Host "NUCLEO ready watcher already running: pid=$oldPid"
            exit 0
        }
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logDir "nucleo_ready_watcher_$stamp.out.log"
$stderrLog = Join-Path $logDir "nucleo_ready_watcher_$stamp.err.log"
$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Root "src\\tools\\watch_nucleo_ready.ps1"),
    "-Root", $Root,
    "-Workspace", $Workspace,
    "-IntervalSeconds", [string]$IntervalSeconds
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
Write-Host "Started NUCLEO ready watcher: pid=$($process.Id)"
Write-Host "stdout=$stdoutLog"
Write-Host "stderr=$stderrLog"
