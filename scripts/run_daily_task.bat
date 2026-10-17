@echo off
rem fterm daily runner for scheduled task: run_daily_auto.py syncs panels from
rem the legacy A-share db (read-only) then aligns asof to the latest trading day.
rem No pause; log appended to data\logs. ASCII-only comments (cmd OEM codepage).
rem 2026-10-06 fix: prior version had literal CR bytes inside paths so the task
rem failed every night since 10-03; -u added for chronological log ordering.
cd /d D:\Final
if not exist data\logs mkdir data\logs
echo === %date% %time% fterm daily auto start === >> data\logs\run_daily.log
.venv\Scripts\python.exe -u -X utf8 scripts\run_daily_auto.py >> data\logs\run_daily.log 2>&1
echo [exit %errorlevel%] === %date% %time% === >> data\logs\run_daily.log
