$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Wait-ForHttpReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
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

& (Join-Path $ScriptRoot "start-backend.ps1")
& (Join-Path $ScriptRoot "start-frontend.ps1")

if (-not (Wait-ForHttpReady -Url "http://127.0.0.1:8000/health")) {
    throw "Backend did not remain reachable at http://127.0.0.1:8000/health after start-dev."
}

if (-not (Wait-ForHttpReady -Url "http://127.0.0.1:8080/")) {
    throw "Frontend did not remain reachable at http://127.0.0.1:8080/ after start-dev."
}

Write-Host ""
Write-Host "Clarion is starting."
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:8080"
Write-Host ""
Write-Host "To stop both services:"
Write-Host ".\scripts\stop-dev.ps1"
