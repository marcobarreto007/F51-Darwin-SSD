@echo off
REM F51 Darwin-SSD - Ghost Stream manual launcher
REM Opens a console window with the loop running. Useful for first-time
REM testing or for watching ingestion live. For unattended operation,
REM install the scheduled task instead:
REM   powershell -ExecutionPolicy Bypass -File scripts\install_ghost_stream_task.ps1

cd /d "%~dp0"

echo.
echo === F51 Ghost Stream - manual launch ===
echo Project: %CD%
echo Logs:    logs\ghost_stream_*.log
echo Press Ctrl+C to stop.
echo.

powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\ghost_stream_runner.ps1"
echo.
echo Runner exited with code %ERRORLEVEL%.
pause
