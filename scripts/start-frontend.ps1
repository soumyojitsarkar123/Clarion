$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot
$FrontendDir = Join-Path $RepoRoot "frontend"
$RunDir = Join-Path $ScriptRoot ".run"
$FrontendPidFile = Join-Path $RunDir "frontend.pid"
$FrontendOutLog = Join-Path $RunDir "frontend.out.log"
$FrontendErrLog = Join-Path $RunDir "frontend.err.log"

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
        try {
            $probe = Invoke-WebRequest -Uri "http://127.0.0.1:8080/" -UseBasicParsing -TimeoutSec 3
            if ($probe.StatusCode -eq 200) {
                Write-Host "Frontend is already running with PID $existingPid on http://127.0.0.1:8080"
                exit 0
            }
        } catch {
        }

        Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $FrontendPidFile -Force -ErrorAction SilentlyContinue
}

$pythonExe = Resolve-Python
$command = "Set-Location '$FrontendDir'; & '$pythonExe' -m http.server 8080"

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
    -PassThru

if (-not (Wait-ForHttpReady -Url "http://127.0.0.1:8080/" -Process $process)) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $FrontendPidFile -Force -ErrorAction SilentlyContinue
    throw "Frontend process started but http://127.0.0.1:8080/ did not become ready in time."
}

$process.Id | Set-Content $FrontendPidFile

Write-Host "Frontend started."
Write-Host "PID: $($process.Id)"
Write-Host "URL: http://127.0.0.1:8080"
Write-Host "A frontend PowerShell window was opened for live logs."
