# Clarion Backend - Agent Commands Reference

## Running the Application

```bash
cd D:\Clarion\Clarion-Backend
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Testing Imports

```bash
python -c "from main import app; print('OK')"
python -c "from services.background_service import BackgroundService; print('OK')"
python -c "from routers.status import router; print('OK')"
```

## Running Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## API Testing

```bash
# Health check
curl http://localhost:8000/health

# Queue stats
curl http://localhost:8000/status/queue/stats

# List active jobs
curl http://localhost:8000/status/jobs/active

# Get document status
curl http://localhost:8000/status/{document_id}

# Dataset stats
curl http://localhost:8000/dataset/relations/stats
```

## Key Files

- `main.py` - Application entry point
- `services/background_service.py` - Background processing orchestration
- `services/processing_pipeline.py` - Document processing pipeline
- `routers/status.py` - Status and monitoring endpoints
- `routers/analyze.py` - Document analysis endpoint
- `models/document.py` - Document models

## Database

SQLite databases at `data/`:

- `data/clarion.db` - Main database
- `data/relation_dataset.db` - Relation dataset records

Tables in `data/clarion.db`:

- `documents` - Document metadata
- `processing_jobs` - Job tracking
- `chunks` - Text chunks
- `embeddings` - Embedding metadata
- `knowledge_maps` - Extracted knowledge maps
- `summaries` - Structured summaries
- `analysis_reports` - Pipeline reports
- `rag_audits` - Retrieval audit records

## Data Directories

- `data/graphs/` - NetworkX graph exports (JSON + Cytoscape JSON)
- `data/vectorstore/` - FAISS indexes
- `data/datasets/` - Relation dataset snapshots
- `logs/` - Application logs

Readable dataset snapshots are also mirrored to `../data/datasets/`.

## Linting

The project uses standard Python type hints. To check:

```bash
python -m py_compile services/background_service.py
python -m py_compile services/processing_pipeline.py
python -m py_compile routers/status.py
```

## Research Modules

The relation-validation experiment code lives under `core/`:

- `core/experiments/relation_validator.py` - Relation validity classifier
- `core/experiments/graph_features.py` - Graph-based feature extraction
- `core/benchmarking/` - Benchmark harness, splitting, and reporting

These modules require `pandas` and `scikit-learn`; they are not part of the
serving API surface.

## Documentation

- `README.md` - Application overview and setup
- `../README.md` - Repository overview
- `../frontend/README.md` - Frontend details
- `../scripts/README.md` - PowerShell helper scripts

## Provider Configuration (Local Ollama)

| Provider | ENV Variable | API Base |
|----------|--------------|----------|
| Ollama | `OLLAMA_MODEL` | `http://localhost:11434/v1` |
