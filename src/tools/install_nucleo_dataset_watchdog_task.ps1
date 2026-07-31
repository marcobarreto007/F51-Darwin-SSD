param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory",
    [string]$TaskName = "F51 NUCLEO Dataset Watchdog",
    [int]$IntervalSeconds = 300,
    [int]$StaleHeartbeatMinutes = 45
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $Root "src\\tools\\start_nucleo_dataset_watchdog.ps1"
if (-not (Test-Path $scriptPath)) {
    throw "missing watchdog start script: $scriptPath"
}

$argument = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-Root", "`"$Root`"",
    "-Workspace", "`"$Workspace`"",
    "-IntervalSeconds", [string]$IntervalSeconds,
    "-StaleHeartbeatMinutes", [string]$StaleHeartbeatMinutes
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null

    $task = Get-ScheduledTask -TaskName $TaskName
    [PSCustomObject]@{
        Ok = $true
        TaskName = $task.TaskName
        State = $task.State
        Root = $Root
        Workspace = $Workspace
        Action = $argument
    }
} catch {
    [PSCustomObject]@{
        Ok = $false
        Error = $_.Exception.Message
        Fallback = "Run src\\tools\\install_nucleo_dataset_watchdog_startup.ps1 for per-user Startup persistence."
    }
    exit 2
}
