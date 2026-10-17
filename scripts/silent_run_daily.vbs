' Silent wrapper for task fterm_daily (added 2026-09-30, per user directive "都改成静默运行").
' Runs the frozen bat with no visible console window, waits for it, propagates its exit code.
' Same pattern as the 13 ECO-W2/NEWECO tasks converted under W2-F67. ASCII only.
WScript.Quit CreateObject("WScript.Shell").Run("cmd.exe /c """"D:\Final\scripts\run_daily_task.bat""""", 0, True)
