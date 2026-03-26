$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot
$BackendDir = Join-Path $RepoRoot "Clarion-Backend"
$RunDir = Join-Path $ScriptRoot ".run"
$BackendPidFile = Join-Path $RunDir "backend.pid"
$BackendOutLog = Join-Path $RunDir "backend.out.log"
$BackendErrLog = Join-Path $RunDir "backend.err.log"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Wait-ForHttpReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)]$Process,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            return $false
        }

        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
        }

        Start-Sleep -Seconds 2
    }

    return $false
}

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
        try {
            $probe = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 3
            if ($probe.StatusCode -eq 200) {
                Write-Host "Backend is already running with PID $existingPid on http://127.0.0.1:8000"
                exit 0
            }
        } catch {
        }

        Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $BackendPidFile -Force -ErrorAction SilentlyContinue
}

$pythonExe = Resolve-Python
$command = "Set-Location '$BackendDir'; & '$pythonExe' -m uvicorn main:app --host 127.0.0.1 --port 8000"

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
    -PassThru

if (-not (Wait-ForHttpReady -Url "http://127.0.0.1:8000/health" -Process $process)) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $BackendPidFile -Force -ErrorAction SilentlyContinue
    throw "Backend process started but http://127.0.0.1:8000/health did not become ready in time."
}

$process.Id | Set-Content $BackendPidFile

Write-Host "Backend started."
Write-Host "PID: $($process.Id)"
Write-Host "URL: http://127.0.0.1:8000"
Write-Host "A backend PowerShell window was opened for live logs."
