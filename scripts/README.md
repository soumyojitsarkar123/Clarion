# Scripts

PowerShell helpers for local Clarion development on Windows.

## Available Scripts

- `start-backend.ps1` - starts the FastAPI backend on `http://127.0.0.1:8000`
- `start-frontend.ps1` - serves the frontend on `http://127.0.0.1:8080`
- `start-dev.ps1` - starts both backend and frontend
- `stop-dev.ps1` - stops processes started by the scripts above

## Usage

From the repository root in Windows PowerShell:

```powershell
.\scripts\start-dev.ps1
```

To stop the local processes:

```powershell
.\scripts\stop-dev.ps1
```

Runtime logs and PID files are written to `scripts/.run/`, which is kept local and git-ignored.
