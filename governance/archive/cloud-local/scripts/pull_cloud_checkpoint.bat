@echo off
REM ============================================
REM pull_cloud_checkpoint.bat
REM Baixa o checkpoint mais recente da nuvem.
REM NAO MEXE em nada rodando. So baixa.
REM ============================================
setlocal enabledelayedexpansion

set CLOUD_HOST=root@192.234.50.251
set CLOUD_PORT=1585
set CLOUD_DIR=/workspace/F51-Darwin-SSD/checkpoints/cloud
set LOCAL_DIR=checkpoints\cloud

if not exist "%LOCAL_DIR%" mkdir "%LOCAL_DIR%"

echo Buscando checkpoint mais recente na nuvem...

REM Pega o cloud_latest.json
ssh -o StrictHostKeyChecking=no -p %CLOUD_PORT% %CLOUD_HOST% "cat %CLOUD_DIR%/cloud_latest.json" > %TEMP%\cloud_latest.json 2>nul
if errorlevel 1 (
    echo ERRO: Nao foi possivel conectar na nuvem.
    exit /b 1
)

REM Extrai step e path
for /f "tokens=2 delims=:," %%a in ('findstr "step" %TEMP%\cloud_latest.json') do set CLOUD_STEP=%%a
for /f "tokens=2 delims=:" %%a in ('findstr "path" %TEMP%\cloud_latest.json') do set CLOUD_PATH=%%a

REM Limpa aspas e espacos
set CLOUD_STEP=%CLOUD_STEP:"=%
set CLOUD_STEP=%CLOUD_STEP: =%
set CLOUD_PATH=%CLOUD_PATH:"=%
set CLOUD_PATH=%CLOUD_PATH: =%

echo Nuvem: step=%CLOUD_STEP%

REM Formata step com 7 digitos
set STEP_PADDED=0000000%CLOUD_STEP%
set STEP_PADDED=%STEP_PADDED:~-7%

set LOCAL_FILE=%LOCAL_DIR%\cloud_step_%STEP_PADDED%.pt

if exist "%LOCAL_FILE%" (
    echo Este checkpoint ja existe localmente.
    echo Nada a fazer.
    exit /b 0
)

echo Baixando %CLOUD_PATH% ...
scp -o StrictHostKeyChecking=no -P %CLOUD_PORT% %CLOUD_HOST%:%CLOUD_PATH% %LOCAL_FILE%

if exist "%LOCAL_FILE%" (
    echo.
    echo OK! Checkpoint baixado: %LOCAL_FILE%
    echo Step: %CLOUD_STEP%
    echo.
    echo Para usar: python scripts/serve_f51.py --checkpoint %LOCAL_FILE%
) else (
    echo ERRO: Download falhou.
    exit /b 1
)
