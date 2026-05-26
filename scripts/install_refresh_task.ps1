param(
    [string]$TaskName = "EuroLeague Data Refresh",
    [string]$Time = "09:00",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python = "python",
    [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $ProjectRoot "scripts\run_scheduled_refresh.ps1"
$argumentParts = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-ProjectRoot", "`"$ProjectRoot`"",
    "-Python", "`"$Python`""
)

if ($ExtraArgs.Trim().Length -gt 0) {
    $argumentParts += @("-ExtraArgs", "`"$ExtraArgs`"")
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($argumentParts -join " ")
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refresh EuroLeague data, ML features and injury availability outputs." `
    -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' for daily runs at $Time."
Write-Host "Logs will be written under: $(Join-Path $ProjectRoot "logs")"
