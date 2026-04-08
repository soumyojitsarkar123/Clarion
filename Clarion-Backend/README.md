# Clarion — Backend

FastAPI backend for document ingestion, chunking, embeddings, knowledge-map generation, summaries, relation dataset capture, and graph export.

---

## What This Is

Clarion is a research-oriented document intelligence pipeline. It accepts PDF/DOCX uploads and produces:

- **Chunk-based embeddings** using SentenceTransformers + FAISS
- **Knowledge maps** via heuristic concept and co-occurrence relation extraction
- **Structured summaries** grounded in extracted document content
- **Relation datasets** exported as CSV/JSON for downstream research use
- **Graph exports** in Cytoscape-compatible JSON format

The backend is provider-aware but does not require a live LLM to function. Core extraction and analysis pipelines use deterministic heuristics. LLM calls (via Ollama by default) are used where available for enrichment tasks and will gracefully fall back when unavailable.

---

## Runtime Configuration

Defaults are defined in [`utils/config.py`](utils/config.py).

| Component | Setting | Default |
|---|---|---|
| Embedding model | `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-en-v1.5` |
| Embedding device | `EMBEDDING_DEVICE` | `auto` (resolves to cuda / mps / cpu) |
| LLM provider | `LLM_PROVIDER` | `ollama` |
| Ollama model | `OLLAMA_MODEL` | `qwen3.5:4b` |
| Ollama base URL | `OLLAMA_API_BASE` | `http://localhost:11434/v1` |
| Chunk size | `CHUNK_SIZE` | `512` |
| Chunk overlap | `CHUNK_OVERLAP` | `50` |
| SQLite timeout | `SQLITE_TIMEOUT_SECONDS` | `30.0` |

Supported LLM providers:

- `ollama` (default, local)
- `openai`
- `deepseek`
- `gemini`

`OPENAI_API_KEY` and other provider keys are optional and only required if you switch providers.

---

## Deterministic vs. Live-Provider Behaviour

### Deterministic paths (always run)

- Text extraction and chunking — [`services/chunking_service.py`](services/chunking_service.py)
- Knowledge-map extraction — [`services/knowledge_map_service.py`](services/knowledge_map_service.py)
- Relation dataset capture — [`services/relation_dataset_service.py`](services/relation_dataset_service.py)
- Summary generation fallback — [`services/summary_service.py`](services/summary_service.py)

### Provider-aware paths (require a live LLM)

- LLM-enriched relation confidence scoring
- Query answering via RAG
- Dataset refinement and quality scoring

> If a live provider is unavailable, the backend logs a warning and continues with deterministic output. Results will differ depending on whether LLM enrichment ran successfully.

---

## Data Models

Schema layer is in [`models/`](models/):

| File | Contents |
|---|---|
| [`document.py`](models/document.py) | Document metadata and processing status |
| [`chunk.py`](models/chunk.py) | Chunk records and section metadata |
| [`embedding.py`](models/embedding.py) | Vector payload schemas |
| [`knowledge_map.py`](models/knowledge_map.py) | Concepts, relations, topics |
| [`graph.py`](models/graph.py) | Graph export schemas and node/edge typing |
| [`retrieval.py`](models/retrieval.py) | Retrieval result payloads |
| [`summary.py`](models/summary.py) | Structured summaries and sections |
| [`response.py`](models/response.py) | API response wrappers |

---

## Storage Layout

All runtime artifacts are written relative to `Clarion-Backend/data/`:

| Path | Contents |
|---|---|
| `data/clarion.db` | Main SQLite database (documents, chunks, jobs, samples) |
| `data/relation_dataset.db` | Relation dataset records |
| `data/graphs/` | Per-document Cytoscape graph exports (JSON) |
| `data/datasets/` | Per-document relation snapshots (CSV/JSON) |
| `data/vectorstore/` | FAISS vector index files |

Readable dataset snapshots are also mirrored to `../data/datasets/`.

Logs are written to `logs/clarion.log` at runtime (not committed to version control).

---

## API Surface

Routers mounted in [`main.py`](main.py):

| Prefix | Purpose |
|---|---|
| `/upload` | Upload PDF/DOCX documents |
| `/analyze` | Trigger full analysis pipeline |
| `/knowledge-map` | Retrieve knowledge maps |
| `/query` | RAG-based document querying |
| `/summary` | Retrieve document summaries |
| `/status` | Processing job status |
| `/dataset` | Access relation datasets |
| `/dataset/factory` | Dataset generation and export |
| `/graph` | Graph exports |
| `/logs` | Runtime log access |
| `/system` | System status and provider info |

Key endpoints:

```
POST   /upload
POST   /analyze/{document_id}
GET    /knowledge-map/{document_id}
GET    /summary/{document_id}
GET    /graph/{document_id}
GET    /dataset/relations
GET    /dataset/relations/stats
GET    /system-status
```

Interactive API docs available at [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) when running locally.

---

## Setup

### 1. Create the environment

```powershell
cd Clarion-Backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

Create `Clarion-Backend/.env` only if you need to override defaults:

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

### 3. Run

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Windows helper scripts are also available in [`../scripts`](../scripts).

### 4. Verify

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- System status: [http://127.0.0.1:8000/system-status](http://127.0.0.1:8000/system-status)

---

## Reproducibility Notes

When recording results, capture:

1. Exact values of `LLM_PROVIDER`, model name, embedding model, and device.
2. Whether runs used live LLM responses or deterministic fallback.
3. Whether embeddings loaded from an existing FAISS index or were re-computed.
4. `CHUNK_SIZE`, `CHUNK_OVERLAP`.
5. Input file checksum or exact source document.

Key implementation notes:

- Chunking is **word-based**, not tokenizer-based ([`services/chunking_service.py`](services/chunking_service.py)).
- Running the backend from the wrong directory will create a misaligned `data/` tree.
- Graph routes return Cytoscape-style JSON. Downstream consumers should handle this format.
