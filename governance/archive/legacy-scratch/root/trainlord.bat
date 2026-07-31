@echo off
REM ═══════════════════════════════════════════════════════
REM F51 TRAIN LORD — Checkpoint Horário + Auto-Restart
REM ═══════════════════════════════════════════════════════
REM Salva checkpoint a cada ~1 hora (500 steps)
REM Se morrer → renasce em 10s
REM Se memória acabar → reduz corpus e tenta de novo
REM ═══════════════════════════════════════════════════════
cd /d C:\Users\marco\Desktop\F51-Darwin-SSD

set ATTEMPT=0
set CORPUS_DIR=data/train_corpus

:loop
set /a ATTEMPT+=1
echo.
echo ══════════════════════════════════════════════════
echo  F51 TRAIN LORD — TENTATIVA %ATTEMPT%
echo  %date% %time%
echo  Corpus: %CORPUS_DIR%
echo  Checkpoint: a cada 500 steps (~1h em 5060 Ti)
echo  Auto-restart: SIM
echo ══════════════════════════════════════════════════

python -u scripts/train_cloud.py ^
  --device cuda ^
  --batch-size 8 ^
  --block-size 512 ^
  --steps 150000 ^
  --save-every 500 ^
  --eval-every 250 ^
  --lr 3e-4 ^
  --corpus %CORPUS_DIR% ^
  --tokenizer tokenizer/f51_bpe ^
  --checkpoint-dir checkpoints/trainlord ^
  --no-weights

echo.
echo ══════════════════════════════════════════════════
echo  TREINO MORREU — exit: %ERRORLEVEL%
echo  Causa provavel: memoria insuficiente para corpus
echo ══════════════════════════════════════════════════

REM Se foi MemoryError, tenta com corpus menor
if %ERRORLEVEL% NEQ 0 (
    if "%CORPUS_DIR%"=="data/train_corpus" (
        echo.
        echo ⚠️  Reduzindo corpus para evitar MemoryError...
        set CORPUS_DIR=data/corpus
        echo   Novo corpus: %CORPUS_DIR%
    )
)

echo.
echo 🔄 Renascendo em 10 segundos...
timeout /t 10 /nobreak >nul
goto loop
