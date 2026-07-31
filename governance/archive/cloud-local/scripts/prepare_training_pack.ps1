# F51 Darwin-SSD — PREPARADOR DE TRAINING PACK
# Wrapper PowerShell para prepare_training_pack.py

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir

# Ativa venv se existir
$VenvPaths = @(
    "$ProjectRoot\.venv",
    "$ProjectRoot\.venv_nitro",
    "$ProjectRoot\venv"
)

$VenvActivated = $false
foreach ($VenvPath in $VenvPaths) {
    if (Test-Path "$VenvPath\Scripts\Activate.ps1") {
        & "$VenvPath\Scripts\Activate.ps1"
        $VenvActivated = $true
        Write-Host "✓ Venv ativado: $VenvPath" -ForegroundColor Green
        break
    }
}

if (-not $VenvActivated) {
    Write-Host "⚠ Nenhum venv encontrado, usando Python do sistema" -ForegroundColor Yellow
}

# Executa script Python
$PythonScript = "$ScriptDir\prepare_training_pack.py"
$Args = $args

& python $PythonScript $Args

exit $LASTEXITCODE
