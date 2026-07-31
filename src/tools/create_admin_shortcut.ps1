$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "PowerShell (Admin).lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-NoExit"
$shortcut.WorkingDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$shortcut.Save()

$shortcutBytes = [System.IO.File]::ReadAllBytes($shortcutPath)
$shortcutBytes[21] = $shortcutBytes[21] -bor 0x20
[System.IO.File]::WriteAllBytes($shortcutPath, $shortcutBytes)
Write-Host "Created elevated PowerShell shortcut: $shortcutPath"
