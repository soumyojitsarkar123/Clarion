# Clarion Backend

FastAPI backend for document ingestion, chunking, embeddings, knowledge-map generation, summaries, dataset capture, and graph export.

## Current Backend Reality

This backend is provider-aware, but it is not purely LLM-driven.

What the code currently does by default:

- extracts text from uploaded PDF/DOCX files
- chunks text by document structure
- creates embeddings with SentenceTransformers
- stores/retrieves vectors with FAISS
- builds knowledge maps using deterministic heuristic extraction and co-occurrence relations
- generates document-grounded summaries using deterministic fallback logic
- stores graph exports, summaries, and relation datasets in SQLite/JSON artifacts

That is important for anyone reproducing results: the codebase currently favors graceful deterministic behavior over blocking on live model inference.

## Runtime Model Configuration

Defaults come from [utils/config.py](utils/config.py).

| Component | Setting | Current default |
|---|---|---|
| Embedding model | `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-en-v1.5` |
| Embedding device | `EMBEDDING_DEVICE` | `auto` |
| LLM provider | `LLM_PROVIDER` | `ollama` |
| Ollama model | `OLLAMA_MODEL` | `qwen3.5:4b` |
| Ollama base URL | `OLLAMA_API_BASE` | `http://localhost:11434/v1` |
| Chunk size | `CHUNK_SIZE` | `512` |
| Chunk overlap | `CHUNK_OVERLAP` | `50` |
| SQLite timeout | `SQLITE_TIMEOUT_SECONDS` | `30.0` |

Supported LLM providers in code:

- `ollama`
- `openai`
- `deepseek`
- `gemini`

Important notes:

- `OPENAI_API_KEY` is optional, not required by default.
- Ollama is the default provider, but the backend still runs useful analysis without a successful live generation call.
- Embedding device `auto` is resolved at startup to `cuda`, `mps`, or `cpu`.

## Deterministic Vs Live-Provider Behavior

### Deterministic-first paths

- knowledge-map extraction in [knowledge_map_service.py](services/knowledge_map_service.py)
- structured summary generation in [summary_service.py](services/summary_service.py)
- relation dataset snapshot export in [relation_dataset_service.py](services/relation_dataset_service.py)

### Provider-aware paths

- LLM availability and provider configuration in [knowledge_map_service.py](services/knowledge_map_service.py)
- query/response generation routes and services
- system health/status checks

### Why this matters

If someone assumes every run is driven by live Ollama/OpenAI output, they will misunderstand the observed results. The current backend intentionally falls back to deterministic processing when the provider is unavailable, slow, or unusable.

## Pydantic Schema Inventory

The backend schema layer is in [models](models):

- [document.py](models/document.py): uploaded document metadata and status
- [chunk.py](models/chunk.py): chunk records and section metadata
- [embedding.py](models/embedding.py): vector payload schemas
- [knowledge_map.py](models/knowledge_map.py): concepts, relations, topics
- [graph.py](models/graph.py): graph export schemas and node/edge typing
- [retrieval.py](models/retrieval.py): retrieval result payloads
- [summary.py](models/summary.py): structured summaries and sections
- [response.py](models/response.py): API response wrappers

If you are changing the API, keep these schemas and the router responses aligned.

## Storage Layout

When the backend is started from `Clarion-Backend/`, the main storage root is:

- [data](data)

Important files/directories:

- app database: [clarion.db](data/clarion.db)
- relation dataset database: [relation_dataset.db](data/relation_dataset.db)
- graph exports: [graphs](data/graphs)
- readable dataset snapshots: [datasets](data/datasets)
- logs: [logs](logs)

The workspace also mirrors readable dataset snapshots to:

- [../data/datasets](../data/datasets)

## API Surface

Routers currently mounted in [main.py](main.py):

- `/upload`
- `/analyze`
- `/knowledge-map`
- `/query`
- `/summary`
- `/status`
- `/dataset`
- `/graph`
- `/logs`
- `/system`

Examples:

- `POST /upload`
- `POST /analyze/{document_id}`
- `GET /knowledge-map/{document_id}`
- `GET /summary/{document_id}`
- `GET /graph/{document_id}`
- `GET /dataset/relations`
- `GET /dataset/relations/stats`
- `GET /system-status`

## Setup

### 1. Create the environment

```powershell
cd Clarion-Backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Optional `.env`

Create `Clarion-Backend/.env` only if you need to override defaults.

Example:

```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_API_BASE=http://localhost:11434/v1

EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=auto

CHUNK_SIZE=512
CHUNK_OVERLAP=50
SQLITE_TIMEOUT_SECONDS=30.0
```

### 3. Run the API

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Windows helper scripts also exist in [../scripts](../scripts), but the command above is the cleanest documented path.

### 4. Verify The Backend

- API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Reproducibility Checklist

If reproducibility matters, record these for every run:

1. Working directory used to launch the backend.
2. Exact values of `LLM_PROVIDER`, provider model, embedding model, and embedding device.
3. Whether the run used live provider responses or deterministic fallback behavior.
4. Whether embeddings loaded successfully or the deterministic embedding fallback was used.
5. `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MIN_CHUNK_SIZE`.
6. Input file checksum or exact source document copy.

Important implementation notes:

- Chunking is word-based in [chunking_service.py](services/chunking_service.py), not tokenizer-based.
- Running the backend from the wrong directory can create a different `data/` tree and make artifacts appear “missing”.
- Graph routes may return either Cytoscape-style exports or legacy node-link JSON depending on what artifacts exist, so downstream consumers should be tolerant of both.

## Dataset Export

For a single research snapshot:

```powershell
cd Clarion-Backend
python export_dataset.py
```

That writes:

- [research_dataset_export.json](data/research_dataset_export.json)

During normal analysis runs, readable per-document relation snapshots are also written under:

- [data/datasets](data/datasets)

## Local Development

Windows PowerShell helpers are available in [../scripts](../scripts) if you want one-command startup for local work.
