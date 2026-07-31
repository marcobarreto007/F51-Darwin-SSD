[CmdletBinding(DefaultParameterSetName = 'Plan')]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [Parameter(ParameterSetName = 'Apply')]
    [switch]$Apply,
    [Parameter(ParameterSetName = 'Rollback')]
    [switch]$Rollback,
    [Parameter(ParameterSetName = 'Rollback')]
    [string]$Manifest
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $project '.venv_nitro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "migration_python_missing: $python"
}

$arguments = @(
    '-m', 'f51_darwin.workspace_migration',
    '--project-root', $project
)
if ($Apply) {
    $arguments += '--apply'
}
if ($Rollback) {
    if (-not $Manifest) {
        throw 'rollback_manifest_required'
    }
    $arguments += @('--rollback', '--manifest', $Manifest)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "single_root_migration_failed: exit $LASTEXITCODE"
}
