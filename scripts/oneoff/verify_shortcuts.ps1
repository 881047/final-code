$ws = New-Object -ComObject WScript.Shell
Get-ChildItem "$env:USERPROFILE\Desktop" -Filter 'fterm*.lnk' | ForEach-Object {
    $s = $ws.CreateShortcut($_.FullName)
    Write-Host ($_.Name + ' -> ' + $s.TargetPath + ' | cwd=' + $s.WorkingDirectory + ' | icon=' + $s.IconLocation + ' | desc=' + $s.Description)
}
