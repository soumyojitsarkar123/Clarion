# Experiment Workflow Guide

This guide provides step-by-step instructions for running research experiments using locally available Ollama models without requiring external API keys.

## Available Models (Installed)

| Model | Tag | Size |
|-------|-----|------|
| Qwen 3 | `qwen3:latest` | 5.2 GB |
| DeepSeek Coder | `deepseek-coder:6.7b` | 3.8 GB |

---

## Prerequisites

### 1. Install and Start Ollama

```bash
# Start Ollama server (runs on http://localhost:11434 by default)
ollama serve
```

### 2. Configure Environment

Create/update `.env` file:

```bash
# Ollama Configuration (local models)
OLLAMA_MODEL=qwen3:latest
OLLAMA_API_BASE=http://localhost:11434/v1

# Embedding (local)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

### 3. Verify Setup

```bash
cd D:\Clarion\Clarion-Backend
python experiments/run_with_ollama.py --task stats
python experiments/run_with_ollama.py --task list
```

---

## Workflow 1: Document Analysis

### Step 1: Start the Backend

```bash
cd D:\Clarion\Clarion-Backend
python main.py
```

### Step 2: Upload and Analyze Document

```bash
# Upload document
curl -X POST "http://localhost:8000/upload" -F "file=@your_document.pdf"

# Analyze document (uses configured Ollama model)
curl -X POST "http://localhost:8000/analyze/{document_id}"
```

---

## Workflow 2: Generate Relation Dataset

Relations are automatically logged during document analysis.

```bash
# Check dataset stats
curl "http://localhost:8000/dataset/relations/stats"

# Export dataset
curl "http://localhost:8000/dataset/relations/export?format=json" -o relations.json
```

### Recommended Documents for Dataset Growth

| Document Type | Expected Relations |
|---------------|-------------------|
| Textbook chapters (CS, math) | 20-50 per chapter |
| Technical documentation | 15-30 per doc |
| Research papers | 10-25 per paper |

**Target**: 100-200 labeled relations for stable benchmarking.

---

## Workflow 3: Manual Labeling

### Interactive Script

```bash
python experiments/run_with_ollama.py --task label
```

### Via API

```bash
# Get unlabeled records
curl "http://localhost:8000/dataset/relations?is_valid=null&limit=50"

# Label a record
curl -X PATCH "http://localhost:8000/dataset/relations/validate" \
  -H "Content-Type: application/json" \
  -d '{"record_id": "xxx", "is_valid": true}'
```

---

## Workflow 4: Run Experiments

### Minimum Sample Requirements

| Experiment | Min Samples | Recommended |
|------------|-------------|-------------|
| Ablation study | 50 | 100+ |
| Embedding comparison | 30 | 80+ |

### Run Benchmark

```bash
# With Qwen 3
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark --runs 10

# With DeepSeek Coder
python experiments/run_with_ollama.py --model deepseek-coder:6.7b --task benchmark --runs 10
```

### Compare Models

```bash
python experiments/run_with_ollama.py --task compare
```

### Run Ablation Study

```bash
python experiments/run_ablation_from_benchmarks.py --out-dir data/reports/ablation
```

---

## Workflow 5: Package Research Artifacts

```bash
# Package all artifacts
python experiments/package_research_artifacts.py
```

---

## Switching Models

```bash
# Command line
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark

# Environment variable (PowerShell)
$env:OLLAMA_MODEL="deepseek-coder:6.7b"
python main.py

# Edit .env file
# OLLAMA_MODEL=deepseek-coder:6.7b
```

---

## Recommended Experiment Order

1. **Data Collection** - Process 5-10 documents, generate 100+ relations
2. **Baseline Benchmark** - Run with default model (qwen3:latest)
3. **Model Comparison** - Compare Qwen vs DeepSeek
4. **Ablation Study** - Run with `--no-graph-features`
5. **Package Artifacts** - Generate tables and plots

---

## Quick Reference

```bash
# Start Ollama
ollama serve

# Start backend
python main.py

# Check stats
python experiments/run_with_ollama.py --task stats

# List models
python experiments/run_with_ollama.py --task list

# Run benchmark
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark

# Compare models
python experiments/run_with_ollama.py --task compare

# Interactive labeling
python experiments/run_with_ollama.py --task label
```
