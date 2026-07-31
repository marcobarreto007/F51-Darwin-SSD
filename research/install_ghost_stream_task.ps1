<#
.SYNOPSIS
  Registers the Windows Task Scheduler task "F51 Ghost Stream".

.DESCRIPTION
  Creates or replaces a scheduled task that launches
  research\ghost_stream_runner.ps1, which in turn runs
    python research\ghost_stream.py --loop --interval 300

  Triggers installed:
    - At logon of the current user
    - Daily at 00:00, repeating every 30 minutes for 1 day (watchdog)

  The wrapper itself loops every 5 minutes; the watchdog simply restarts
  the loop if it has died. A global mutex inside the wrapper guarantees
  we never run two loops in parallel.

  Runs only when the current user is logged on (no password stored).
  For unattended startup before logon, edit this script to use
  -RunLevel Highest with an AtStartup trigger and a stored credential.

.PARAMETER TaskName
  Scheduled task name. Default "F51 Ghost Stream".

.PARAMETER Interval
  Cycle interval in seconds passed to the wrapper. Default 300.

.PARAMETER Force
  Re-create the task even if it already exists. Default behavior.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File research\install_ghost_stream_task.ps1
#>

[CmdletBinding()]
param(
    [string]$TaskName = "F51 Ghost Stream",
    [int]$Interval = 300,
    [switch]$Force = $true
)

$ErrorActionPreference = "Stop"

# --- Locate project root -------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Runner = Join-Path $ProjectRoot "research\ghost_stream_runner.ps1"

if (-not (Test-Path $Runner)) {
    throw "Wrapper not found: $Runner"
}

# --- Resolve powershell.exe (handle Winged/Sysinternals variants) --------
$Pwsh = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $Pwsh) {
    throw "powershell.exe not found on PATH."
}

$User = "$env:USERDOMAIN\$env:USERNAME"

# --- Build task definition ----------------------------------------------
$ActionArgs = '-ExecutionPolicy Bypass -NoProfile -NonInteractive -File "{0}" -Interval {1}' -f $Runner, $Interval
$Action = New-ScheduledTaskAction `
    -Execute $Pwsh `
    -Argument $ActionArgs `
    -WorkingDirectory $ProjectRoot

# Trigger 1: at logon
$TriggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $User

# Trigger 2: watchdog — fire every 30 min so the loop restarts if it died.
# The wrapper's global mutex guarantees the second invocation exits cleanly
# instead of double-running python.
$TriggerWatchdog = New-ScheduledTaskTrigger -Once -At (Get-Date -Hour 0 -Minute 0 -Second 0) `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Days 365)

$Triggers = @($TriggerLogon, $TriggerWatchdog)

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId $User `
    -LogonType Interactive `
    -RunLevel Limited

# --- Register (replace if exists) ---------------------------------------
if ($Force) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Removing existing task '$TaskName'..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Triggers `
    -Settings $Settings `
    -Principal $Principal `
    -Description "F51 Darwin-SSD — Ghost Stream automatic ingestion (loop every $Interval s)" `
    -Force | Out-Null

# --- Final report --------------------------------------------------------
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " F51 Ghost Stream scheduled task installed" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Task name : $TaskName"
Write-Host " User      : $User"
Write-Host " Wrapper   : $Runner"
Write-Host " Interval  : ${Interval}s"
Write-Host " Triggers  : AtLogon + every 30 min (watchdog)"
Write-Host ""
Write-Host " Verify it is running:" -ForegroundColor Yellow
Write-Host "   schtasks /query /tn `"$TaskName`" /fo LIST /v"
Write-Host "   Get-Process python | Format-Table Id, StartTime, Path"
$todayLog = "logs\ghost_stream_$(Get-Date -Format yyyyMMdd).log"
Write-Host ("   Get-Content {0} -Tail 20" -f $todayLog)
Write-Host ""
Write-Host " Start now (manual):" -ForegroundColor Yellow
Write-Host "   Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host ""
Write-Host " Stop:" -ForegroundColor Yellow
Write-Host "   Stop-ScheduledTask -TaskName `"$TaskName`""
Write-Host "   (or) Get-Process python | Stop-Process -Force"
Write-Host ""
Write-Host " Uninstall:" -ForegroundColor Yellow
Write-Host "   powershell -File research\uninstall_ghost_stream_task.ps1"
Write-Host "================================================" -ForegroundColor Cyan
