@echo off
rem fterm daily runner: panel sync (legacy db read-only) -> decision loop (paper matching).
rem Run after market close (~15:30). ASCII-only comments (cmd OEM codepage).
chcp 65001 >nul
cd /d D:\Final
.venv\Scripts\python.exe scripts\run_daily.py
echo.
echo ============ Press any key to close ============
pause >nul
