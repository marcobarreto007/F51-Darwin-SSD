param(
    [string]$Root = "C:\Users\marco\Desktop\F51-Darwin-SSD",
    [string]$Workspace = "C:\Users\marco\Desktop\F51-Corpus-Factory"
)

$ErrorActionPreference = "Stop"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($previousPythonPath) { "$Root\src;$previousPythonPath" } else { "$Root\src" }

Push-Location $Root
try {
    python -m tools.nucleo_dataset_organism --status --workspace $Workspace
} finally {
    Pop-Location
}
