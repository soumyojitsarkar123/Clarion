$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $ScriptRoot ".run"
$PidFiles = @(
    (Join-Path $RunDir "backend.pid"),
    (Join-Path $RunDir "frontend.pid")
)

foreach ($pidFile in $PidFiles) {
    if (-not (Test-Path $pidFile)) {
        continue
    }

    $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pidValue -and (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $pidValue -Force
        Write-Host "Stopped process $pidValue"
    }

    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Write-Host "Clarion local services stopped."
