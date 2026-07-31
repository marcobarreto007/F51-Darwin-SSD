param(
    [string]$RunDir,
    [int]$IntervalSeconds = 5
)

$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Resolve-LatestRunDir {
    param([string]$SearchRoot)

    $candidate = Get-ChildItem -Path $SearchRoot -Filter "dopamine_status.txt" -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $candidate) {
        return $null
    }
    return $candidate.DirectoryName
}

if (-not $RunDir) {
    $RunDir = Resolve-LatestRunDir -SearchRoot (Join-Path $ROOT "workspace")
    if (-not $RunDir) {
        Write-Host "Nenhum dopamine_status.txt encontrado em workspace/. Passe -RunDir explicitamente."
        exit 1
    }
}

Write-Host "Monitorando: $RunDir"
Write-Host "(Ctrl+C para sair)"
Write-Host ""

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $flag = Join-Path $ROOT "TRAINING_HALTED.flag"
    $halted = Test-Path -LiteralPath $flag

    $dopaminePath = Join-Path $RunDir "dopamine_status.txt"
    $status = if (Test-Path -LiteralPath $dopaminePath) {
        Get-Content -LiteralPath $dopaminePath -Tail 1
    } else {
        "(dopamine_status.txt ainda nao existe)"
    }

    $gpu = & nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null

    $procs = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match "darwin_organism" }
    )

    Write-Host "[$ts] halted=$halted  procs=$($procs.Count)"
    Write-Host "  status: $status"
    foreach ($line in $gpu) {
        Write-Host "  gpu: $line"
    }
    Write-Host ""

    Start-Sleep -Seconds $IntervalSeconds
}
