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

## Ollama Local Models

For experiments without API keys:

```bash
# Start Ollama
ollama serve

# Run experiments with Ollama
python experiments/run_with_ollama.py --task stats
python experiments/run_with_ollama.py --task list
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark --runs 10
python experiments/run_with_ollama.py --model deepseek-coder:6.7b --task analyze --file doc.pdf
python experiments/run_with_ollama.py --task compare
```

## Testing Imports

```bash
python -c "from main import app; print('OK')"
python -c "from services.background_service import BackgroundService; print('OK')"
python -c "from routers.status import router; print('OK')"
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

SQLite database at: `data/clarion.db`

Tables:
- `documents` - Document metadata
- `processing_jobs` - Job tracking

## Data Directories

- `data/documents/` - Original documents
- `data/chunks/` - Processed text chunks
- `data/vectorstore/` - FAISS indexes
- `data/graphs/` - NetworkX graphs
- `data/knowledge_maps/` - Extracted knowledge maps
- `logs/` - Application logs

## Linting

The project uses standard Python type hints. To check:
```bash
python -m py_compile services/background_service.py
python -m py_compile services/processing_pipeline.py
python -m py_compile routers/status.py
```

## Experiments

```bash
# Check dataset stats
python experiments/run_with_ollama.py --task stats

# List available models
python experiments/run_with_ollama.py --task list

# Run benchmark with specific model
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark --runs 10

# Compare models (all available)
python experiments/run_with_ollama.py --task compare

# Interactive labeling
python experiments/run_with_ollama.py --task label

# Run ablation from benchmarks
python experiments/run_ablation_from_benchmarks.py --out-dir data/reports/ablation

# Package research artifacts
python experiments/package_research_artifacts.py --out-root data/reports/packages
```

## Document Upload CLI

Lightweight tool for uploading and analyzing documents without frontend:

```bash
# Upload single document (triggers analysis automatically)
python experiments/doc_cli.py upload document.pdf

# Upload and wait for completion
python experiments/doc_cli.py upload document.pdf --wait

# Batch upload multiple files
python experiments/doc_cli.py batch "docs/*.pdf"

# Check dataset statistics
python experiments/doc_cli.py stats

# Upload and check dataset growth
python experiments/doc_cli.py check document.pdf

# Wait for document to complete
python experiments/doc_cli.py wait document_id

# Get document status
python experiments/doc_cli.py status document_id
```

## Documentation

- `docs/EXPERIMENT_WORKFLOW.md` - Step-by-step experiment execution guide
- `README.md` - Application overview and setup

## Available Models (Installed)

| Model | Tag | Size | Description |
|-------|-----|------|-------------|
| Qwen 3 | `qwen3:latest` | 5.2 GB | Qwen 3 8B model |
| DeepSeek Coder | `deepseek-coder:6.7b` | 3.8 GB | DeepSeek Coder 6.7B |
| Gemma 3 | `gemma3:4b` | 3.3 GB | Gemma 3 4B model |

## Provider Configuration (Local Ollama)

| Provider | ENV Variable | API Base |
|----------|--------------|----------|
| Ollama | `OLLAMA_MODEL` | `http://localhost:11434/v1` |
