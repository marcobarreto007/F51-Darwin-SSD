param(
    [string]$Root = "",
    [string]$DatasetRoot = "",
    [string]$Config = "src\\configs\\darwin_x_1.6b_nitro.yaml",
    [string]$Checkpoint = "",
    [int]$StepsPerCycle = 2000,
    [int]$BlockSize = 64,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.00015,
    [int]$EvalEvery = 500,
    [int64]$MinimumModelParameters = 1700000000,
    [int64]$MinimumCheckpointBytes = 8000000000,
    [int]$MinimumFreeDiskGB = 100,
    [int]$MaximumIdleGpuMemoryMB = 2048,
    [switch]$Launch,
    [switch]$Canary,
    [int]$CanarySteps = 250,
    [int64]$CanaryHoldoutTokens = 1048576,
    [int]$PostRunObservationSeconds = 120,
    [string]$CanaryRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }
$implementation = Join-Path $Root "src\\f51_darwin\\operations\start_overnight_16b.ps1"
if (-not (Test-Path -LiteralPath $implementation -PathType Leaf)) {
    throw "Overnight launcher implementation not found: $implementation"
}

$forward = @{}
foreach ($name in $PSBoundParameters.Keys) { $forward[$name] = $PSBoundParameters[$name] }
$forward["Root"] = $Root
& $implementation @forward
