#!/usr/bin/env pwsh
# F51 Darwin-SSD — Auto Deploy GPU
# Comprime, upload, descomprime e inicia treino automaticamente
# Uso: .\scripts\auto_deploy_gpu.ps1

param(
    [string]$GPUHost = "ssh1.vast.ai",
    [int]$GPUPort = 33348,
    [string]$CorpusPath = "data/corpus",
    [string]$ProjectPath = "F51-Darwin-SSD",
    [int]$Steps = 50000,
    [int]$BatchSize = 8,
    [int]$BlockSize = 512,
    [switch]$SkipCompress = $false,
    [switch]$SkipUpload = $false
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Color = "Green")
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message" -ForegroundColor $Color
}

function Test-Command {
    param([string]$Cmd)
    try {
        $null = Get-Command $Cmd -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Invoke-SSH {
    param([string]$Command)
    $sshCmd = "ssh -p $GPUPort -o StrictHostKeyChecking=no root@${GPUHost} `"$Command`""
    Invoke-Expression $sshCmd 2>&1 | Where-Object { $_ -notmatch "^Welcome" -and $_ -notmatch "^If" -and $_ -notmatch "^Have" }
}

function Test-GPUConnection {
    Write-Status "Testando conexão GPU..." "Cyan"
    $result = Invoke-SSH "echo 'GPU OK' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    if ($result -match "GPU OK") {
        Write-Status "GPU conectada: $result" "Green"
        return $true
    }
    Write-Status "FALHA: Não conectou à GPU" "Red"
    return $false
}

function Get-GPUInfo {
    Write-Status "Coletando info da GPU..." "Cyan"
    $info = @{
        Disk = (Invoke-SSH "df -h /workspace | grep -E '^/|overlay' | awk '{print `$4}'") -join ""
        Uptime = (Invoke-SSH "cat /proc/uptime | awk '{print `$1 / 60}'") -join ""
    }
    Write-Status "GPU: Disco livre=$($info.Disk), Uptime=$([math]::Round($info.Uptime)) min" "Green"
    return $info
}

function Compress-Corpus {
    param([string]$Source, [string]$Output)

    if ($SkipCompress) {
        Write-Status "SKIP: Compressão já desabilitada" "Yellow"
        return $Output
    }

    Write-Status "Iniciando compressão do corpus..." "Cyan"
    Write-Status "Source: $Source" "Gray"
    Write-Status "Output: $Output" "Gray"

    # Verificar tamanho do corpus
    $corpusSize = (Get-ChildItem -Path $Source -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Status "Tamanho do corpus: $([math]::Round($corpusSize, 2)) GB" "Cyan"

    # Comprimir usando tar via Git Bash
    $compressScript = @"
cd "$Source" && tar -czf `"$Output`" .
"@

    Write-Status "Comprimindo (isso pode demorar 20-60 min)..." "Yellow"
    $startTime = Get-Date

    # Usar Git Bash tar
    $bashExe = "C:\Program Files\Git\bin\bash.exe"
    if (Test-Path $bashExe) {
        & $bashExe -c "cd '$Source' && tar -czf '$Output' . 2>&1" | Out-Host
    } else {
        # Tentar usar bash do PATH
        bash -c "cd '$Source' && tar -czf '$Output' . 2>&1" | Out-Host
    }

    $elapsed = (Get-Date) - $startTime
    $finalSize = (Get-Item $Output).Length / 1GB

    Write-Status "Compressão concluída!" "Green"
    Write-Status "Tempo: $([math]::Round($elapsed.TotalMinutes, 1)) min" "Gray"
    Write-Status "Tamanho final: $([math]::Round($finalSize, 2)) GB ($([math]::Round((1 - $finalSize / $corpusSize) * 100, 1))% compressão)" "Green"

    return $Output
}

function Upload-ToGPU {
    param([string]$LocalFile, [string]$RemotePath)

    if ($SkipUpload) {
        Write-Status "SKIP: Upload já desabilitado" "Yellow"
        return $true
    }

    Write-Status "Iniciando upload para GPU..." "Cyan"
    Write-Status "Local: $LocalFile" "Gray"
    Write-Status "Remote: $RemotePath" "Gray"

    $fileSize = (Get-Item $LocalFile).Length / 1GB
    Write-Status "Tamanho: $([math]::Round($fileSize, 2)) GB" "Cyan"

    # Criar diretório remoto
    $remoteDir = Split-Path -Parent $RemotePath
    Invoke-SSH "mkdir -p $remoteDir"

    # Upload via SCP
    $scpCmd = "scp -P $GPUPort -o StrictHostKeyChecking=no `"$LocalFile`" root@${GPUHost}:`"$RemotePath`""
    Write-Status "Uploading (tempo estimado: $([math]::Round($fileSize * 2)) min)..." "Yellow"

    $startTime = Get-Date
    Invoke-Expression $scpCmd 2>&1 | Out-Host
    $elapsed = (Get-Date) - $startTime

    Write-Status "Upload concluído!" "Green"
    Write-Status "Tempo: $([math]::Round($elapsed.TotalMinutes, 1)) min" "Gray"

    return $true
}

function Deploy-Project {
    Write-Status "Fazendo deploy do projeto F51..." "Cyan"

    # Verificar se projeto existe na GPU
    $projectExists = Invoke-SSH "test -d /workspace/$ProjectPath && echo 'EXISTS'"
    if ($projectExists -match "EXISTS") {
        Write-Status "Projeto já existe na GPU" "Yellow"
    } else {
        Write-Status "Projeto não encontrado, fazendo upload..." "Yellow"
        # Criar tar do projeto (sem data/)
        $projectTar = "$env:TEMP\f51_project.tar"
        bash -c "cd '$PSScriptRoot/../..' && tar -cf '$projectTar' --exclude='data' --exclude='.git' --exclude='checkpoints' --exclude='runs' f51_darwin scripts tokenizer configs 2>&1" | Out-Null
        Write-Status "Projeto empacotado: $projectTar" "Gray"

        Upload-ToGPU -LocalFile $projectTar -RemotePath "/workspace/$ProjectPath.tar"
        Invoke-SSH "cd /workspace && tar -xf $ProjectPath.tar && rm $ProjectPath.tar"
        Write-Status "Projeto deployado!" "Green"

        Remove-Item $projectTar -Force -ErrorAction SilentlyContinue
    }
}

function Extract-CorpusOnGPU {
    param([string]$RemoteArchive, [string]$TargetDir)

    Write-Status "Descomprimindo corpus na GPU..." "Cyan"

    Invoke-SSH "mkdir -p $TargetDir"

    # Descomprimir
    $extractCmd = "cd $TargetDir && tar -xzf $RemoteArchive && ls -lh | wc -l"
    Write-Status "Extraindo (pode demorar)..." "Yellow"

    $result = Invoke-SSH $extractCmd
    Write-Status "Arquivos extraídos: $result" "Green"
}

function Start-Training {
    Write-Status "Iniciando treinamento..." "Cyan"

    $trainCmd = @"
cd /workspace/$ProjectPath && nohup python scripts/train_cloud.py --device cuda --batch-size $BatchSize --block-size $BlockSize --steps $Steps --save-every 1000 --eval-every 500 --checkpoint-dir checkpoints/cloud --corpus data/corpus --tokenizer tokenizer/f51_bpe > /workspace/train.log 2>&1 &
"@

    Invoke-SSH $trainCmd

    Start-Sleep -Seconds 3
    $pid = Invoke-SSH "pgrep -f train_cloud.py | head -1"

    Write-Status "Treinamento iniciado!" "Green"
    Write-Status "PID: $pid" "Gray"
    Write-Status "Log: ssh -p $GPUPort root@${GPUHost} 'tail -f /workspace/train.log'" "Cyan"
}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     F51 DARWIN-SSD — AUTO DEPLOY GPU                      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Verificações
Write-Status "Verificando pré-requisitos..." "Cyan"
if (-not (Test-Path $CorpusPath)) {
    Write-Status "ERRO: Corpus não encontrado em $CorpusPath" "Red"
    exit 1
}

if (Test-Command "vastai") {
    $balance = vastai show user 2>&1 | Select-String "Credit" | ForEach-Object { $_ -replace '\D', '' }
    if ($balance) {
        Write-Status "Saldo Vast.ai: $$balance" "Gray"
        $hours = [math]::Floor([double]$balance / 0.501)
        Write-Status "Tempo disponível: ~$hours horas" "Gray"
    }
}

# 1. Testar conexão GPU
if (-not (Test-GPUConnection)) {
    exit 1
}

# 2. Info GPU
$gpuInfo = Get-GPUInfo

# 3. Deploy projeto
Deploy-Project

# 4. Comprimir corpus (se necessário)
$archiveFile = "$env:TEMP\corpus_backup.tar.gz"
if (-not $SkipCompress -and -not (Test-Path $archiveFile)) {
    Compress-Corpus -Source $CorpusPath -Output $archiveFile
} elseif (Test-Path $archiveFile) {
    Write-Status "Arquivo comprimido já existe: $archiveFile" "Yellow"
    Write-Status "Tamanho: $([math]::Round((Get-Item $archiveFile).Length / 1GB, 2)) GB" "Gray"
}

# 5. Upload corpus
Upload-ToGPU -LocalFile $archiveFile -RemotePath "/workspace/corpus_backup.tar.gz"

# 6. Descomprimir na GPU
Extract-CorpusOnGPU -RemoteArchive "/workspace/corpus_backup.tar.gz" -TargetDir "/workspace/$ProjectPath/data"

# 7. Limpar arquivo local
if (Get-Confirm "Remover arquivo comprimido local?") {
    Remove-Item $archiveFile -Force
    Write-Status "Arquivo local removido" "Gray"
}

# 8. Iniciar treino
Start-Training

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║     DEPLOY CONCLUÍDO! TREINAMENTO RODANDO                 ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Status "Comandos úteis:" "Cyan"
Write-Host "  Ver log:     ssh -p $GPUPort root@${GPUHost} 'tail -f /workspace/train.log'" -ForegroundColor Gray
Write-Host "  Ver GPU:     ssh -p $GPUPort root@${GPUHost} 'nvidia-smi'" -ForegroundColor Gray
Write-Host "  Ver checkpoints: ssh -p $GPUPort root@${GPUHost} 'ls -lh /workspace/$ProjectPath/checkpoints/cloud/'" -ForegroundColor Gray
Write-Host ""
