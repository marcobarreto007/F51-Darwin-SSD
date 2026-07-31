[CmdletBinding()]
param(
    [switch]$Canary,
    [switch]$Calibrate,
    [switch]$PublishCandidate
)

$implementation = Join-Path $PSScriptRoot "..\f51_darwin\operations\start_smol_darwin_transplant.ps1"
& $implementation @PSBoundParameters
exit $LASTEXITCODE
