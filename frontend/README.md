# Clarion Frontend

This directory contains the static browser UI for uploads, reports, and knowledge graph visualization.

## Run Locally

From the repository root on Windows PowerShell:

```powershell
.\scripts\start-frontend.ps1
```

This opens a dedicated PowerShell window for the frontend server.

You can also run it manually:

```bash
cd frontend
python -m http.server 8080
```

Then open:

- `http://127.0.0.1:8080`

The frontend expects the backend API at:

- `http://127.0.0.1:8000`

If the UI is opened via `file://`, uploads and API calls will not work correctly.
