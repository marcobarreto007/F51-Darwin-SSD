# F51 Darwin-X 600M — Cloud Launch Script
# Modelo: 600M params, d_model=1408, L=12, E=6, context=4096
# Organismo completo: GABA, DAE, Causal Bus, Ghost, Curiosity, Spider
# Checkpoint isolado: workspace/03_CHECKPOINTS_600M/

param(
    [string]$Config = "configs/darwin_x_600m.yaml",
    [int]$StepsPerCycle = 500,
    [int]$BlockSize = 2048,
    [int]$BatchSize = 2,
    [float]$LearningRate = 1.5e-4,
    [int]$EvalEvery = 500,
    [int]$WarmupSteps = 100,
    [int]$AccumSteps = 4,
    [string]$Device = "cuda",
    [string]$CausalMode = "shadow",
    [switch]$FreshStart,
    [string]$Resume = $null
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  F51 DARWIN-X 600M — CLOUD LAUNCH" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Config:        $Config"
Write-Host "  Steps/Cycle:   $StepsPerCycle"
Write-Host "  Block Size:    $BlockSize"
Write-Host "  Batch Size:    $BatchSize"
Write-Host "  LR:            $LearningRate"
Write-Host "  Causal Mode:   $CausalMode"
Write-Host "  Warmup Steps:  $WarmupSteps"
Write-Host "  Accum Steps:   $AccumSteps"
Write-Host "============================================================"

$pythonArgs = @(
    "-u", "-m", "scripts.darwin_organism",
    "run247",
    "--config", $Config,
    "--causal-mode", $CausalMode,
    "--batch-size", $BatchSize,
    "--block-size", $BlockSize,
    "--steps", $StepsPerCycle,
    "--lr", $LearningRate,
    "--optimizer", "adamw",
    "--warmup-steps", $WarmupSteps,
    "--accum-steps", $AccumSteps,
    "--device", $Device,
    "--eval-every", $EvalEvery
)

if ($FreshStart) {
    $pythonArgs += "--fresh-start"
    Write-Host "  MODE: FRESH START (nova linhagem)" -ForegroundColor Yellow
}
elseif ($Resume) {
    $pythonArgs += "--resume"
    $pythonArgs += $Resume
    Write-Host "  MODE: RESUME de checkpoint" -ForegroundColor Green
}
else {
    Write-Host "  ERRO: --FreshStart ou --Resume obrigatorio" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Launching..." -ForegroundColor Green
& python $pythonArgs
