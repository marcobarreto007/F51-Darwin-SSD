param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [int]$IntervalSeconds = 300,
    [int]$StaleHeartbeatMinutes = 45,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $Workspace "logs"
$stateDir = Join-Path $Workspace "state"
$pidFile = Join-Path $logDir "nucleo_dataset_organism.pid"
$watchdogLog = Join-Path $logDir "nucleo_dataset_organism_watchdog.log"
$heartbeatPath = Join-Path $stateDir "nucleo_dataset_organism_heartbeat.json"
New-Item -ItemType Directory -Force -Path $logDir, $stateDir | Out-Null

function Write-WatchdogLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -Path $watchdogLog -Value "[$stamp] $Message" -Encoding utf8
}

function Get-OrganismPid {
    if (-not (Test-Path $pidFile)) {
        return $null
    }
    $raw = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($raw -match '^\d+$') {
        return [int]$raw
    }
    return $null
}

function Test-OrganismAlive {
    $pidValue = Get-OrganismPid
    if ($null -eq $pidValue) {
        return $false
    }
    return $null -ne (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)
}

function Test-HeartbeatFresh {
    if (-not (Test-Path $heartbeatPath)) {
        return $false
    }
    $age = (Get-Date) - (Get-Item $heartbeatPath).LastWriteTime
    return $age.TotalMinutes -le $StaleHeartbeatMinutes
}

function Restart-Organism {
    Write-WatchdogLog "restart requested"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "src\\tools\\start_nucleo_dataset_organism.ps1") -Restart | ForEach-Object {
        Write-WatchdogLog $_
    }
}

Write-WatchdogLog "watchdog started interval=${IntervalSeconds}s stale_minutes=$StaleHeartbeatMinutes"

do {
    $alive = Test-OrganismAlive
    $fresh = Test-HeartbeatFresh
    if (-not $alive) {
        Write-WatchdogLog "organism not running"
        Restart-Organism
    } elseif (-not $fresh) {
        Write-WatchdogLog "heartbeat stale or missing"
        Restart-Organism
    } else {
        $pidValue = Get-OrganismPid
        Write-WatchdogLog "ok pid=$pidValue"
    }

    if ($Once) {
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
