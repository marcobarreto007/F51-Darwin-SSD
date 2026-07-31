param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [int]$IntervalSeconds = 300,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $Workspace "logs"
$stateDir = Join-Path $Workspace "state"
$pidFile = Join-Path $logDir "nucleo_ready_watcher.pid"
$logFile = Join-Path $logDir "nucleo_ready_watcher.log"
$readyPath = Join-Path $stateDir "nucleo_ready_gate.json"
New-Item -ItemType Directory -Force -Path $logDir, $stateDir | Out-Null
Set-Content -Path $pidFile -Value ([string]$PID) -Encoding ascii

function Write-ReadyLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -Path $logFile -Value "[$stamp] $Message" -Encoding utf8
}

Write-ReadyLog "ready watcher started interval=${IntervalSeconds}s"

do {
    $checkScript = Join-Path $Root "src\\tools\\check_nucleo_ready.ps1"
    $output = & powershell -ExecutionPolicy Bypass -File $checkScript -Json 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | Out-String).Trim()
    if ($text) {
        Set-Content -Path $readyPath -Value $text -Encoding utf8
    }
    if ($exitCode -eq 0) {
        Write-ReadyLog "PRONTO $text"
        break
    }
    Write-ReadyLog "waiting $text"

    if ($Once) {
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
