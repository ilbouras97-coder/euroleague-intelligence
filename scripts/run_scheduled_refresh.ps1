param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python = "python",
    [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"
$logDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "refresh-$timestamp.log"
$arguments = @("scripts\refresh_data.py")

if ($ExtraArgs.Trim().Length -gt 0) {
    $arguments += $ExtraArgs -split "\s+"
}

Push-Location $ProjectRoot
try {
    "[$(Get-Date -Format o)] Starting EuroLeague refresh" | Tee-Object -FilePath $logPath
    & $Python @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Refresh failed with exit code $LASTEXITCODE"
    }
    "[$(Get-Date -Format o)] Refresh finished successfully" | Tee-Object -FilePath $logPath -Append
}
finally {
    Pop-Location
}
