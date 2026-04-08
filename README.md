# Clarion

Clarion is a local-first document analysis workspace with a FastAPI backend and a static frontend for uploads, summaries, datasets, and knowledge graph visualization.

## What The Current System Actually Does

The current pipeline is:

1. Upload PDF or DOCX documents.
2. Extract raw text and split it into structure-aware chunks.
3. Create embeddings with SentenceTransformers and index them with FAISS.
4. Build a knowledge map from extracted text.
5. Save summaries, graph artifacts, and relation datasets for later inspection.
6. Serve the results in the browser UI.

Important implementation note:

- The current mapping and summary flow is primarily deterministic and heuristic-first.
- Live LLM availability is checked, but user-facing summaries and core knowledge-map generation are designed to keep working even when the live model is slow or unavailable.
- That means the codebase is not an LLM-only extraction pipeline, and the docs need to reflect that clearly.

## Model Stack

### Embeddings

- Default embedding model: `BAAI/bge-large-en-v1.5`
- Library: `sentence-transformers`
- Vector index: `faiss-cpu`
- Device setting: `EMBEDDING_DEVICE=auto`
- Resolved device is chosen at runtime: `cuda`, `mps`, or `cpu`

### LLM Providers

Supported providers in code:

- `ollama`
- `openai`
- `deepseek`
- `gemini`

Default provider/runtime:

- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=qwen3.5:4b`
- `OLLAMA_API_BASE=http://localhost:11434/v1`

Important behavior:

- Ollama is the default, but it is not required for every part of the pipeline to run.
- If the live provider is unavailable, Clarion falls back to deterministic behavior instead of failing the whole analysis.

## Data Models In The Repository

There are two different meanings of “models” in this repo, and mixing them up causes confusion:

### Runtime ML Models

- Embedding model: `BAAI/bge-large-en-v1.5`
- LLM model/provider: configurable by environment, defaulting to Ollama `qwen3.5:4b`

### Pydantic Schema Models

Backend schema definitions live in [models](Clarion-Backend/models):

- [document.py](Clarion-Backend/models/document.py)
- [chunk.py](Clarion-Backend/models/chunk.py)
- [embedding.py](Clarion-Backend/models/embedding.py)
- [knowledge_map.py](Clarion-Backend/models/knowledge_map.py)
- [graph.py](Clarion-Backend/models/graph.py)
- [retrieval.py](Clarion-Backend/models/retrieval.py)
- [summary.py](Clarion-Backend/models/summary.py)
- [response.py](Clarion-Backend/models/response.py)

## Repository Layout

```text
Clarion/
├── Clarion-Backend/          # FastAPI backend, storage, graph/data services
├── frontend/                 # Static browser UI
├── scripts/                  # Windows PowerShell launch helpers
└── data/                     # Repo-level dataset mirror used by the workspace UI
```

## Quick Start

### 1. Backend

```powershell
cd Clarion-Backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

```powershell
cd frontend
python -m http.server 8080
```

### 3. Verify The Run

- Backend docs: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8080`

### Windows Helper Scripts

PowerShell helpers exist under [scripts](scripts), but the manual commands above are the primary documented run path.

## Reproducibility Notes

This is the part most likely to drift if it is not documented precisely.

### 1. Run From The Correct Working Directory

Backend paths such as `data/`, `logs/`, and `data/datasets/` are resolved relative to the backend working directory.

For consistent results, start the backend from `Clarion-Backend/`.

If you run the backend from a different working directory, you can accidentally create a second `data/` tree and think files are missing.

### 2. Pin The Runtime Configuration

For reproducible runs, record at least:

- `LLM_PROVIDER`
- `OLLAMA_MODEL` or other provider model
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_DEVICE`
- resolved embedding device
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `SQLITE_TIMEOUT_SECONDS`

### 3. Chunking Is Word-Based

`CHUNK_SIZE` and `CHUNK_OVERLAP` are currently applied to word slices in [chunking_service.py](Clarion-Backend/services/chunking_service.py), not tokenizer-level tokens. Any README claiming token-based chunking would be inaccurate.

### 4. Fallback Behavior Changes Outputs

Clarion intentionally keeps running when some model services are unavailable.

That means two runs on the same file can differ if:

- the live LLM provider is available in one run but not another
- the embedding model loads in one run but falls back in another
- the document text extraction quality changes

For experiments, record whether the run used:

- live embeddings vs deterministic fallback embeddings
- live provider responses vs deterministic summary/mapping fallback logic

### 5. Storage Outputs

Key artifacts are stored under [Clarion-Backend/data](Clarion-Backend/data):

- main app DB: [clarion.db](Clarion-Backend/data/clarion.db)
- relation dataset DB: [relation_dataset.db](Clarion-Backend/data/relation_dataset.db)
- graph exports: [graphs](Clarion-Backend/data/graphs)
- readable relation snapshots: [datasets](Clarion-Backend/data/datasets)

The workspace also mirrors readable dataset snapshots to the repo-level [data/datasets](data/datasets) folder for easier inspection from the UI/workspace.

## More Specific Docs

- Backend details: [Clarion-Backend/README.md](Clarion-Backend/README.md)
- Frontend details: [frontend/README.md](frontend/README.md)
- Script usage: [scripts/README.md](scripts/README.md)
