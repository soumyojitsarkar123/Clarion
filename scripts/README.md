# Clarion Development & Automation Scripts

This directory contains development automation scripts, process lifecyle management tools, and live end-to-end smoke tests for Clarion.

---

## Script Inventory

| Script | Platform | Description |
|---|---|---|
| `start-backend.ps1` | Windows PowerShell | Starts the FastAPI backend in a dedicated window on `http://127.0.0.1:8000` |
| `start-frontend.ps1` | Windows PowerShell | Starts the static frontend web server in a dedicated window on `http://127.0.0.1:8080` |
| `start-dev.ps1` | Windows PowerShell | Launches both backend and frontend development processes simultaneously |
| `stop-dev.ps1` | Windows PowerShell | Gracefully terminates background processes managed by `start-dev.ps1` |
| `e2e_smoke_test.py` | Cross-Platform (Python) | Comprehensive live integration test against a running backend API |

---

## Windows Development Workflow

### Starting Development Servers

From the repository root in PowerShell:

```powershell
.\scripts\start-dev.ps1
```

- Backend API: `http://127.0.0.1:8000` (docs at `http://127.0.0.1:8000/docs`)
- Frontend UI: `http://127.0.0.1:8080`

### Stopping Development Servers

```powershell
.\scripts\stop-dev.ps1
```

Process identifiers (PIDs) are stored in `scripts/.run/`, which is managed locally and git-ignored.

---

## Linux / macOS / WSL Development Workflow

On Unix-like platforms or WSL, development servers can be run in separate terminals or terminal multiplexers (e.g. `tmux`):

### Terminal 1: Backend API
```bash
cd Clarion-Backend
source .venv/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: Frontend Server
```bash
cd frontend
python3 -m http.server 8080 --bind 127.0.0.1
```

---

## End-to-End Live Smoke Test (`e2e_smoke_test.py`)

The script `e2e_smoke_test.py` performs a live, programmatic health and pipeline integration test against an active Clarion backend instance.

### Test Operations:
1. Verifies `/` and `/system-status` connectivity.
2. Synthesizes a well-formed DOCX document in-memory using `python-docx`.
3. Uploads the document via `POST /upload`.
4. Executes full end-to-end analysis synchronously via `POST /analyze/{document_id}`.
5. Verifies all downstream endpoints:
   - `GET /knowledge-map/{document_id}`
   - `GET /summary/{document_id}`
   - `GET /graph/{document_id}`
   - `POST /query/{document_id}` (RAG retrieval and question answering)
   - `GET /dataset/relations`
6. Cleans up test document artifacts via `DELETE /upload/{document_id}`.

### Execution:

Ensure the backend is running on `http://127.0.0.1:8000`, then run:

**Linux / macOS / WSL:**
```bash
.venv_linux/bin/python scripts/e2e_smoke_test.py
```

**Windows:**
```powershell
Clarion-Backend\.venv\Scripts\python.exe scripts\e2e_smoke_test.py
```

> **Note on RAG Query Timing:** When a live local Ollama model is connected, the RAG query step may take 10–60 seconds depending on local CPU/GPU inference latency. When running under deterministic fallback mode, execution completes in under 5 seconds.
