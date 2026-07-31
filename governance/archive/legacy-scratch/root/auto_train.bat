@echo off
REM F51 Darwin — Auto-Restart Training Loop
REM Salva checkpoint a cada 300 steps (~12 min)
REM Se morrer, renasce automaticamente em 10s
cd /d C:\Users\marco\Desktop\F51-Darwin-SSD

set ATTEMPT=0

:loop
set /a ATTEMPT+=1
echo.
echo ============================================
echo  F51 DARWIN — TENTATIVA %ATTEMPT%
echo  %date% %time%
echo  Save: 300 steps  |  Auto-restart: ON
echo ============================================
echo.

python -u scripts/train_cloud.py ^
  --device cuda ^
  --batch-size 8 ^
  --block-size 512 ^
  --steps 150000 ^
  --save-every 300 ^
  --eval-every 150 ^
  --lr 3e-4 ^
  --corpus data/corpus ^
  --tokenizer tokenizer/f51_bpe ^
  --checkpoint-dir checkpoints/fresh ^
  --no-weights

echo.
echo ============================================
echo  TREINO MORREU — exit code: %ERRORLEVEL%
echo  Renascendo em 10 segundos...
echo ============================================
timeout /t 10 /nobreak >nul
goto loop
