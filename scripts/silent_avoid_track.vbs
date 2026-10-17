' Silent wrapper for the chain #31-C avoidance tracking job (added 2026-10-11).
' Runs the frozen command with no visible console window, waits for it, propagates its exit code.
' Same pattern as the 14 ECO-W2/NEWECO/fterm tasks converted under W2-F67.
' ASCII only -- VBScript reads .vbs as ANSI, so non-ASCII comments corrupt parsing.
' Note: inner paths are quoted with quadrupled quotes (""""), matching silent_run_daily.vbs.
WScript.Quit CreateObject("WScript.Shell").Run("cmd.exe /c ""cd /d C:\Users\27817\WorkBuddy\Worktrees\Final\master-d5a41331 && .venv312\Scripts\python.exe -X utf8 scripts\avoid_value_chain.py --track >> data\logs\avoid_track.log 2>&1""", 0, True)
