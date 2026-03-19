$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot
$FrontendDir = Join-Path $RepoRoot "frontend"
$RunDir = Join-Path $ScriptRoot ".run"
$FrontendPidFile = Join-Path $RunDir "frontend.pid"
$FrontendOutLog = Join-Path $RunDir "frontend.out.log"
$FrontendErrLog = Join-Path $RunDir "frontend.err.log"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Resolve-Python {
    $backendVenv = Join-Path $RepoRoot "Clarion-Backend\.venv\Scripts\python.exe"
    if (Test-Path $backendVenv) {
        return $backendVenv
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    throw "Python was not found. Create Clarion-Backend\.venv or install Python and add it to PATH."
}

if (Test-Path $FrontendPidFile) {
    $existingPid = (Get-Content $FrontendPidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-Host "Frontend is already running with PID $existingPid on http://127.0.0.1:8080"
        exit 0
    }
    Remove-Item $FrontendPidFile -Force -ErrorAction SilentlyContinue
}

$pythonExe = Resolve-Python
$arguments = @("-m", "http.server", "8080")

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $FrontendDir `
    -RedirectStandardOutput $FrontendOutLog `
    -RedirectStandardError $FrontendErrLog `
    -PassThru

$process.Id | Set-Content $FrontendPidFile

Write-Host "Frontend started."
Write-Host "PID: $($process.Id)"
Write-Host "URL: http://127.0.0.1:8080"
Write-Host "Logs: $FrontendOutLog"
