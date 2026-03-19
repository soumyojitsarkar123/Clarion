$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $ScriptRoot "start-backend.ps1")
& (Join-Path $ScriptRoot "start-frontend.ps1")

Write-Host ""
Write-Host "Clarion is starting."
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:8080"
Write-Host ""
Write-Host "To stop both services:"
Write-Host ".\scripts\stop-dev.ps1"
