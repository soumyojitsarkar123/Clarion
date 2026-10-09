# Clarion Frontend

The Clarion frontend is a lightweight, dependency-free web workspace for document uploading, analysis monitoring, interactive knowledge graph visualization (via Cytoscape.js), structured summary inspection, and RAG query exploration.

---

## Architecture & Integration

The frontend operates as a static client application that interfaces directly with the FastAPI backend:

- **Backend API Base**: `http://127.0.0.1:8000`
- **Default Frontend Port**: `http://127.0.0.1:8080` (or `http://127.0.0.1:3000`)
- **Key Modules**:
  - `index.html`: Workspace layout and UI component structure
  - `app.js`: Client-side state management, polling logic, Cytoscape graph rendering, and API communication
  - `styles.css`: Dark-mode styling, responsive panels, and graph container layouts

> **Note on Protocol:** The frontend must be served over HTTP (e.g. `http://127.0.0.1:...`). Opening `index.html` directly using the `file://` protocol will cause browser CORS restrictions to block uploads and API communication.

---

## Running Locally

### Option 1: Automated Script (Windows PowerShell)

From the repository root:

```powershell
.\scripts\start-frontend.ps1
```

This launches a dedicated PowerShell window running the local static file server on `http://127.0.0.1:8080`.

### Option 2: Manual Python HTTP Server (Cross-Platform)

From the `frontend/` directory:

```bash
cd frontend
python3 -m http.server 8080 --bind 127.0.0.1
```

> **Port Conflict Note:** If the human annotation server is currently active on port 8080, run the frontend on an alternate port (such as 3000):
> ```bash
> python3 -m http.server 3000 --bind 127.0.0.1
> ```
> Then open `http://127.0.0.1:3000` in your web browser.

---

## Features & UI Workflow

1. **Document Upload**:
   - Drag-and-drop or select PDF and DOCX files.
   - Uploads to `POST /upload`, registering document metadata and initializing chunking.
2. **Analysis Execution & Progress Tracking**:
   - Triggers `POST /analyze/{document_id}`.
   - Continuously polls `GET /status/{document_id}` with stage progress percentage until completion.
3. **Interactive Knowledge Graph**:
   - Fetches Cytoscape JSON from `GET /graph/{document_id}`.
   - Visualizes concept nodes, relationship types (prerequisite, definition, explanation, etc.), and directed edges.
   - Supports zoom, pan, node selection, and cluster exploration.
4. **Structured Summaries & Grounding**:
   - Displays hierarchical summaries retrieved from `GET /summary/{document_id}`.
5. **RAG Question Answering**:
   - Submits queries to `POST /query/{document_id}`.
   - Renders grounded answers alongside retrieved chunk context citations.

---

## Verification & Code Quality

Validate the JavaScript syntax without external build dependencies:

```bash
node --check app.js
```
