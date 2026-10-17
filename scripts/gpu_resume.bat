@echo off
rem ASCII only (cmd OEM codepage).
cd /d D:\Final
if not exist data\logs mkdir data\logs
echo === %date% %time% gpu resume start === >> data\logs\gpu_resume.log
.venv312\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()" || (echo CUDA not available - abort >> data\logs\gpu_resume.log & exit /b 1)
.venv312\Scripts\python.exe -c "import winreg; k=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\CI\Policy'); v=winreg.QueryValueEx(k,'VerifiedAndReputablePolicyState')[0]; exit(0 if v==0 else 1)" || (echo SAC still on - abort >> data\logs\gpu_resume.log & exit /b 1)
.venv312\Scripts\python.exe -X utf8 scripts\dlv2_chain.py >> data\logs\gpu_resume.log 2>&1
echo === %date% %time% dlv2 done === >> data\logs\gpu_resume.log
