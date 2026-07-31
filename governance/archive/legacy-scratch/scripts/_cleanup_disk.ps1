# HISTORICAL ONLY - DO NOT EXECUTE.
# F51 Darwin-SSD — Limpeza de disco (2026-07-16)
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$script:freed = 0.0

function Nuke-File {
    param([string]$Path, [string]$Label, [double]$GB)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        try {
            Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            $script:freed += $GB
            Write-Host "[OK] $Label — $GB GB" -ForegroundColor Green
        } catch {
            Write-Host "[ERRO] $Label — $_" -ForegroundColor Red
        }
    } else {
        Write-Host "[SKIP] Nao encontrado: $Label" -ForegroundColor Yellow
    }
}

function Nuke-DirContents {
    param([string]$DirPath, [string]$Label, [double]$GB)
    if (Test-Path -LiteralPath $DirPath -PathType Container) {
        try {
            $items = Get-ChildItem -LiteralPath $DirPath -ErrorAction Stop
            $count = ($items | Measure-Object).Count
            $items | Remove-Item -Recurse -Force -ErrorAction Stop
            $script:freed += $GB
            Write-Host "[OK] $Label ($count itens) — $GB GB" -ForegroundColor Green
        } catch {
            Write-Host "[ERRO] $Label — $_" -ForegroundColor Red
        }
    } else {
        Write-Host "[SKIP] Nao encontrado: $Label" -ForegroundColor Yellow
    }
}

Nuke-DirContents "C:\Users\marco\Desktop\F51-Corpus-Factory\batches" "Batches NUCLEO (upload_failed)" 16.9
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\03_CHECKPOINTS\organism_cycle_066.pt" "Checkpoint cycle_066" 10.13
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\03_CHECKPOINTS\organism_cycle_067.pt" "Checkpoint cycle_067" 10.13
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\03_CHECKPOINTS\organism_cycle_068.pt" "Checkpoint cycle_068" 10.13
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast.bin" "Corpus v1 (substituido por v2)" 67.58
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\00_CORPUS_PRINCIPAL_tokens_feast.bin.manifest.json" "Manifest corpus v1" 0.001
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\tokens_synthetic_v1.bin" "Sintetico v1" 0.35
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\tokens_synthetic_v1.bin.manifest.json" "Manifest sintetico v1" 0.001
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\tokens_unified_balanced.bin" "Balanced antigo" 0.37
Nuke-File "C:\Users\marco\Desktop\F51-Dataset-Organizado\01_TOKENIZADOS\tokens_unified_balanced.bin.manifest.json" "Manifest balanced" 0.001
Nuke-File "C:\Users\marco\Desktop\F51-Inference\checkpoint\organism_cycle_237.pt" "Checkpoint duplicado (inference)" 2.93
Nuke-DirContents "C:\Users\marco\Desktop\F51-Darwin-SSD\data\_MIGRATION_BACKUPS" "Migration backups (repo)" 0.01
