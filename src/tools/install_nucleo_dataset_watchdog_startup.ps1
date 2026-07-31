param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [int]$IntervalSeconds = 300,
    [int]$StaleHeartbeatMinutes = 45
)

$ErrorActionPreference = "Stop"

$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$launcherPath = Join-Path $startupDir "F51_NUCLEO_Dataset_Watchdog.cmd"
$watchdogStarter = Join-Path $Root "src\\tools\\start_nucleo_dataset_watchdog.ps1"

if (-not (Test-Path $watchdogStarter)) {
    throw "missing watchdog starter: $watchdogStarter"
}

New-Item -ItemType Directory -Force -Path $startupDir | Out-Null

$lines = @(
    "@echo off",
    "cd /d `"$Root`"",
    "start `"F51 NUCLEO Dataset Watchdog`" /min powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$watchdogStarter`" -Root `"$Root`" -Workspace `"$Workspace`" -IntervalSeconds $IntervalSeconds -StaleHeartbeatMinutes $StaleHeartbeatMinutes"
)

Set-Content -Path $launcherPath -Value $lines -Encoding ascii

[PSCustomObject]@{
    Ok = $true
    Launcher = $launcherPath
    Root = $Root
    Workspace = $Workspace
}
