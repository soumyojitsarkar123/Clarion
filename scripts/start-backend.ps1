$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot
$BackendDir = Join-Path $RepoRoot "Clarion-Backend"
$RunDir = Join-Path $ScriptRoot ".run"
$BackendPidFile = Join-Path $RunDir "backend.pid"
$BackendOutLog = Join-Path $RunDir "backend.out.log"
$BackendErrLog = Join-Path $RunDir "backend.err.log"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Resolve-Python {
    $venvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    throw "Python was not found. Create Clarion-Backend\.venv or install Python and add it to PATH."
}

if (Test-Path $BackendPidFile) {
    $existingPid = (Get-Content $BackendPidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-Host "Backend is already running with PID $existingPid on http://127.0.0.1:8000"
        exit 0
    }
    Remove-Item $BackendPidFile -Force -ErrorAction SilentlyContinue
}

$pythonExe = Resolve-Python
$arguments = @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000")

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $BackendOutLog `
    -RedirectStandardError $BackendErrLog `
    -PassThru

$process.Id | Set-Content $BackendPidFile

Write-Host "Backend started."
Write-Host "PID: $($process.Id)"
Write-Host "URL: http://127.0.0.1:8000"
Write-Host "Logs: $BackendOutLog"
