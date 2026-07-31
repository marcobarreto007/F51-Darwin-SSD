param(
    [string]$Hub = "root@70.30.158.46",
    [int]$Port = 56013,
    [int64]$ExpectedCheckpointBytes = 21185819745,
    [int]$IntervalSeconds = 60,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Invoke-Hub {
    param([Parameter(Mandatory = $true)][string]$Command)

    $quotedCommand = "'" + $Command.Replace("'", "'\''") + "'"
    $remoteCommand = "bash -lc $quotedCommand"
    $sshArgs = @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", [string]$Port,
        $Hub,
        $remoteCommand
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & ssh @sshArgs 2>$null
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Last-Matching-Line {
    param(
        [string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Pattern
    )

    $matches = @($Lines | Where-Object { $_ -match $Pattern })
    if ($matches.Count -eq 0) {
        return ""
    }
    return $matches[-1].Trim()
}

function Progress-Bar {
    param([double]$Percent)

    $safePercent = [Math]::Max(0, [Math]::Min(100, $Percent))
    $filled = [int][Math]::Floor($safePercent / 5)
    return ("#" * $filled) + ("." * (20 - $filled))
}

function Get-Nucleo-State {
    $tokenLines = @(Invoke-Hub "if [ -f /workspace/tokens/tokenize_v2.log ]; then tr '\r' '\n' < /workspace/tokens/tokenize_v2.log | grep -E '[0-9]+/[0-9]+ files|DONE|done' | tail -1; fi")
    $tokenLine = Last-Matching-Line -Lines $tokenLines -Pattern "([0-9]+/[0-9]+ files|DONE|done)"

    $tokenSizeLines = @(Invoke-Hub "stat -c%s /workspace/tokens/tokens_full.bin 2>/dev/null || true")
    $tokenSizeLine = Last-Matching-Line -Lines $tokenSizeLines -Pattern "^[0-9]+$"
    $tokenBytes = 0L
    if ($tokenSizeLine -match "^[0-9]+$") {
        $tokenBytes = [int64]$tokenSizeLine
    }

    $activeTokenizerLines = @(Invoke-Hub "ps -eo pid,ppid,cmd | grep '[t]okenize_v2.py' | grep -v 'Wait for tokenization' | wc -l")
    $activeTokenizerLine = Last-Matching-Line -Lines $activeTokenizerLines -Pattern "^[0-9]+$"
    $activeTokenizerCount = 0
    if ($activeTokenizerLine -match "^[0-9]+$") {
        $activeTokenizerCount = [int]$activeTokenizerLine
    }

    $ckptLines = @(Invoke-Hub "stat -c%s /workspace/checkpoints/step_0007900_moe.pt 2>/dev/null || true")
    $ckptBytesLine = Last-Matching-Line -Lines $ckptLines -Pattern "^[0-9]+$"
    $ckptBytes = 0L
    if ($ckptBytesLine -match "^[0-9]+$") {
        $ckptBytes = [int64]$ckptBytesLine
    }

    $diskLines = @(Invoke-Hub "df -h / | tail -1")
    $diskLine = Last-Matching-Line -Lines $diskLines -Pattern "\s+[0-9]+%\s+/"

    $procLines = @(Invoke-Hub "ps -eo cmd | grep -E 'tokenize_v2.py|scp |rsync ' | grep -v grep | grep -v 'Wait for tokenization' | wc -l")
    $procLine = Last-Matching-Line -Lines $procLines -Pattern "^[0-9]+$"

    $doneFiles = 0
    $totalFiles = 0
    $tokenPct = 0
    if ($tokenLine -match "([0-9]+)/([0-9]+) files") {
        $doneFiles = [int]$matches[1]
        $totalFiles = [int]$matches[2]
        if ($totalFiles -gt 0) {
            $tokenPct = [Math]::Round(($doneFiles / $totalFiles) * 100, 1)
        }
    }
    $tokenFileOk = ($tokenBytes -gt 0 -and ($tokenBytes % 4) -eq 0)
    $tokenCount = 0L
    if ($tokenFileOk) {
        $tokenCount = [int64]($tokenBytes / 4)
    }
    $tokenDone = (($totalFiles -gt 0 -and $doneFiles -ge $totalFiles) -or $tokenLine -match "DONE|done")
    if (-not $tokenDone -and $tokenFileOk -and $activeTokenizerCount -eq 0) {
        $tokenDone = $true
        $tokenPct = 100
        $tokenLine = "artifact_ready; last_log=$tokenLine; active_tokenizers=0; bytes=$tokenBytes; tokens=$tokenCount"
    }

    $ckptGB = [Math]::Round($ckptBytes / 1GB, 2)
    $expectedCheckpointGB = [Math]::Round($ExpectedCheckpointBytes / 1GB, 2)
    $ckptPct = [Math]::Round(($ckptBytes / $ExpectedCheckpointBytes) * 100, 1)

    [PSCustomObject]@{
        Time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        TokenLine = $tokenLine
        TokenPercent = $tokenPct
        TokenDone = $tokenDone
        CheckpointBytes = $ckptBytes
        CheckpointGB = $ckptGB
        ExpectedCheckpointGB = $expectedCheckpointGB
        CheckpointPercent = $ckptPct
        CheckpointDone = ($ckptBytes -ge $ExpectedCheckpointBytes)
        DiskLine = $diskLine
        RemoteProcessCount = $procLine
    }
}

function Write-State {
    param([Parameter(Mandatory = $true)]$State)

    Clear-Host
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "   F51 NUCLEO - $($State.Time)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "TOKENIZATION" -ForegroundColor Yellow
    if ($State.TokenLine) {
        Write-Host "   $($State.TokenLine)"
        Write-Host ("   [{0}] {1}%" -f (Progress-Bar $State.TokenPercent), $State.TokenPercent)
    } else {
        Write-Host "   No tokenization status found."
    }

    Write-Host ""
    Write-Host "CHECKPOINT" -ForegroundColor Yellow
    Write-Host ("   [{0}] {1}% ({2} / {3} GiB)" -f (Progress-Bar $State.CheckpointPercent), $State.CheckpointPercent, $State.CheckpointGB, $State.ExpectedCheckpointGB)
    Write-Host "   bytes=$($State.CheckpointBytes)"

    Write-Host ""
    Write-Host "HUB" -ForegroundColor Yellow
    Write-Host "   disk: $($State.DiskLine)"
    Write-Host "   matching remote processes: $($State.RemoteProcessCount)"

    Write-Host ""
    if ($State.TokenDone) {
        Write-Host "   OK tokenization complete" -ForegroundColor Green
    }
    if ($State.CheckpointDone) {
        Write-Host "   OK checkpoint complete" -ForegroundColor Green
    }
    if ($State.TokenDone -and $State.CheckpointDone) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "   NUCLEO READY FOR MERGE/SYNC GATES" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        try { [Console]::Beep(880, 300) } catch {}
        return $true
    }

    Write-Host ""
    Write-Host "   refresh interval: ${IntervalSeconds}s; Ctrl+C to stop" -ForegroundColor DarkGray
    return $false
}

do {
    try {
        $state = Get-Nucleo-State
        $complete = Write-State -State $state
    } catch {
        Write-Host "Monitor error: $($_.Exception.Message)" -ForegroundColor Red
        $complete = $false
        if ($Once) {
            exit 1
        }
    }

    if ($Once -or $complete) {
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
