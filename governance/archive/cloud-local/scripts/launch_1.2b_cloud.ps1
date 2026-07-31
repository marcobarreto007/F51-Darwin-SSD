# F51 Darwin-X 1.2B — Cloud Launch Script
# Modelo: 1.2B params, d_model=1536, L=14, E=8, context=4096
# Organismo completo: GABA, DAE, Causal Bus, Ghost, Curiosity, Spider
# Checkpoint isolado: workspace/03_CHECKPOINTS_1.2B/
# Requer: GPU 40GB+ (A100 ou dual GPU)

param(
    [string]$Config = "configs/darwin_x_1.2b.yaml",
    [int]$StepsPerCycle = 500,
    [int]$BlockSize = 2048,
    [int]$BatchSize = 2,
    [float]$LearningRate = 1.0e-4,
    [int]$EvalEvery = 500,
    [int]$WarmupSteps = 500,
    [int]$AccumSteps = 8,
    [string]$Device = "cuda",
    [string]$CausalMode = "shadow",
    [switch]$FreshStart,
    [string]$Resume = $null
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  F51 DARWIN-X 1.2B — CLOUD LAUNCH" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Config:        $Config"
Write-Host "  Steps/Cycle:   $StepsPerCycle"
Write-Host "  Block Size:    $BlockSize"
Write-Host "  Batch Size:    $BatchSize"
Write-Host "  LR:            $LearningRate"
Write-Host "  Causal Mode:   $CausalMode"
Write-Host "  Warmup Steps:  $WarmupSteps"
Write-Host "  Accum Steps:   $AccumSteps (gradient accumulation)"
Write-Host "  VRAM est:      ~24GB (dual GPU) ou ~40GB (single A100)"
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
Write-Host "  Launching 1.2B organism..." -ForegroundColor Green
& python $pythonArgs
