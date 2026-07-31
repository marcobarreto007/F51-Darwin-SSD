<#
.SYNOPSIS
  Removes the "F51 Ghost Stream" scheduled task and (optionally) kills
  any still-running python ghost_stream process.

.PARAMETER TaskName
  Scheduled task name. Default "F51 Ghost Stream".

.PARAMETER KillProcess
  Also stop the running python ghost_stream loop.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File research\uninstall_ghost_stream_task.ps1
#>

[CmdletBinding()]
param(
    [string]$TaskName = "F51 Ghost Stream",
    [switch]$KillProcess
)

$ErrorActionPreference = "Stop"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing scheduled task '$TaskName'..." -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed." -ForegroundColor Green
} else {
    Write-Host "No scheduled task named '$TaskName'." -ForegroundColor DarkGray
}

if ($KillProcess) {
    Write-Host "Stopping any running ghost_stream python processes..." -ForegroundColor Yellow
    $procs = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "ghost_stream\.py" }
    if ($procs) {
        foreach ($p in $procs) {
            try {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
                Write-Host "  killed PID $($p.ProcessId)" -ForegroundColor Green
            } catch {
                Write-Host "  could not kill PID $($p.ProcessId): $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "  no ghost_stream.py process found." -ForegroundColor DarkGray
    }
}
