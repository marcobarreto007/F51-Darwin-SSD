<#
.SYNOPSIS
  F51 Ghost Stream — wrapper invoked by Windows Task Scheduler.

.DESCRIPTION
  Single-instance PowerShell wrapper around
    python research/ghost_stream.py --loop --interval 300

  - Acquires a global mutex so the Task Scheduler watchdog trigger and
    manual launches never double-run the loop.
  - Appends to logs\ghost_stream_YYYYMMDD.log (rotated daily by date).
  - Resolves python.exe explicitly (Task Scheduler does not always
    inherit the full interactive PATH).
  - Prunes logs older than 30 days on each start.

.PARAMETER Interval
  Seconds between ghost stream cycles. Default 300 (5 minutes).

.PARAMETER Source
  Optional comma-separated list of sources (fineweb,wikipedia,math,c4).
  If omitted, ghost_stream.py uses all configured sources.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File research\ghost_stream_runner.ps1
#>

[CmdletBinding()]
param(
    [int]$Interval = 300,
    [string]$Source = ""
)

$ErrorActionPreference = "Stop"

# --- Resolve project root (parent of src/scripts/) ---------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# --- Log helpers ---------------------------------------------------------
$LogsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}
$Today = Get-Date -Format "yyyyMMdd"
$LogFile = Join-Path $LogsDir "ghost_stream_$Today.log"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    try {
        Add-Content -Path $LogFile -Value $line -Encoding utf8 -ErrorAction Stop
    } catch {
        # File may be briefly locked by another writer; retry once after a
        # short backoff. If it still fails, give up silently — console still
        # sees the line.
        Start-Sleep -Milliseconds 50
        try { Add-Content -Path $LogFile -Value $line -Encoding utf8 -ErrorAction Stop } catch {}
    }
    Write-Host $line
}

# --- Single-instance mutex (global, kernel-level) ------------------------
$MutexName = "Global\F51GhostStream"
$Mutex = New-Object System.Threading.Mutex($false, $MutexName)
$acquired = $false
try {
    try {
        $acquired = $Mutex.WaitOne(0, $false)
    } catch [System.ApplicationException] {
        $acquired = $false
    } catch [System.Threading.AbandonedMutexException] {
        # Previous owner died while holding the mutex — ownership passes
        # to us. Treat as acquired but signal recovery.
        $acquired = $true
    }
    if (-not $acquired) {
        Write-Log "ghost_stream_runner: another instance already running (mutex $MutexName held). Exiting."
        exit 0
    }

    # --- Resolve python.exe -------------------------------------------------
    $PythonCandidates = @(
        (Get-Command python.exe -ErrorAction SilentlyContinue).Source,
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Python312\python.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
    if (-not $PythonCandidates) {
        Write-Log "ghost_stream_runner: FATAL python.exe not found."
        throw "python.exe not found. Install Python or add it to PATH."
    }
    $Python = $PythonCandidates | Select-Object -First 1

    Write-Log "ghost_stream_runner: START pid=$PID user=$env:USERNAME"
    Write-Log "ghost_stream_runner: python=$Python"
    Write-Log "ghost_stream_runner: project_root=$ProjectRoot"
    Write-Log "ghost_stream_runner: interval=${Interval}s source='$Source'"

    # --- Build argument list ------------------------------------------------
    $ArgList = @("-u", "research\ghost_stream.py", "--loop", "--interval", $Interval)
    if ($Source) {
        $ArgList += @("--source", $Source)
    }

    # --- Prune logs older than 30 days -------------------------------------
    Get-ChildItem -Path $LogsDir -Filter "ghost_stream_*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    # --- Launch python with stdout+stderr appended to log ------------------
    # NOTE: We deliberately use cmd.exe for the 2>&1 merge. In PowerShell 5.1,
    # `& python ... 2>&1` wraps each stderr line in an ErrorRecord and, under
    # $ErrorActionPreference = "Stop", turns non-fatal warnings (HuggingFace
    # "unauthenticated request" notice, etc.) into terminating errors. Running
    # through cmd.exe sidesteps the wrapping entirely.
    $exitCode = 0
    $CycleStart = Get-Date
    try {
        # Build a single quoted python argument string. Wrap any arg that
        # contains whitespace in double quotes ([char]34) so cmd.exe parses
        # them correctly. We avoid backtick escape (`") because backticks
        # inside single-quoted PS strings are literal, not escapes.
        $q = [char]34
        $pyArgs = ($ArgList | ForEach-Object {
            if ($_ -match '\s') { "$q$_$q" } else { "$_" }
        }) -join ' '

        $cmdLine = "$q$Python$q $pyArgs >> $q$LogFile$q 2>&1"

        # Live-tail the log so manual launches see streaming output. The job
        # runs in a child PowerShell process; we poll it while python runs.
        $tailJob = Start-Job -ScriptBlock {
            param($Path)
            if (Test-Path $Path) {
                Get-Content -Path $Path -Wait -Tail 0
            }
        } -ArgumentList $LogFile

        try {
            & cmd.exe /c $cmdLine
            $exitCode = $LASTEXITCODE
            if ($null -eq $exitCode) { $exitCode = 0 }
        } finally {
            Start-Sleep -Milliseconds 200
            Receive-Job $tailJob -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host $_ }
            Stop-Job $tailJob -ErrorAction SilentlyContinue
            Remove-Job $tailJob -Force -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Log "ghost_stream_runner: EXCEPTION $($_.Exception.Message)"
        $exitCode = -1
    }

    $Duration = (Get-Date) - $CycleStart
    Write-Log "ghost_stream_runner: END exit=$exitCode duration=$($Duration.ToString())"

    exit $exitCode
}
finally {
    if ($acquired) {
        try { $Mutex.ReleaseMutex() } catch {}
    }
    try { $Mutex.Dispose() } catch {}
}
