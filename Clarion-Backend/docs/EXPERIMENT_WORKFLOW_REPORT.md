# Experiment Workflow Implementation Report

**Date**: 2026-02-22  
**Focus**: Switch from implementation to experiment workflow execution

---

## Executive Summary

Successfully implemented a complete experiment workflow for the Clarion Backend system, enabling research-grade evaluation using locally available Ollama models. The implementation provides step-by-step workflows, helper scripts, and documentation for running experiments without external API keys.

---

## 1. Implemented Components

### 1.1 Helper Script: `experiments/run_with_ollama.py`

A unified CLI tool for all experiment tasks:

| Task | Command | Description |
|------|---------|-------------|
| Stats | `--task stats` | Show dataset statistics |
| List | `--task list` | List available Ollama models |
| Benchmark | `--task benchmark` | Run benchmark experiments |
| Analyze | `--task analyze` | Analyze documents |
| Label | `--task label` | Interactive manual labeling |
| Compare | `--task compare` | Compare multiple models |

**Usage Examples**:
```bash
# Check dataset stats
python experiments/run_with_ollama.py --task stats

# List models
python experiments/run_with_ollama.py --task list

# Run benchmark with specific model
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark --runs 10

# Compare all models
python experiments/run_with_ollama.py --task compare

# Interactive labeling
python experiments/run_with_ollama.py --task label
```

### 1.2 Documentation: `docs/EXPERIMENT_WORKFLOW.md`

Comprehensive step-by-step guide covering:
- Prerequisites and setup
- Document analysis workflow
- Relation dataset generation
- Manual labeling
- Running experiments
- Packaging research artifacts
- Model switching

### 1.3 Environment Configuration

Updated `.env.example` with Ollama settings:
```bash
OLLAMA_MODEL=qwen3:latest
OLLAMA_API_BASE=http://localhost:11434/v1
```

Model switching via:
- Command line: `--model <model_name>`
- Environment: `$env:OLLAMA_MODEL="gemma3:4b"`
- `.env` file: `OLLAMA_MODEL=gemma3:4b`

---

## 2. Installed Models

Verified available Ollama models:

| Model | Tag | Size |
|-------|-----|------|
| Qwen 3 | `qwen3:latest` | 4.9 GB |
| DeepSeek Coder | `deepseek-coder:6.7b` | 3.6 GB |
| Gemma 3 | `gemma3:4b` | 3.1 GB |

---

## 3. Step-by-Step Workflows

### 3.1 Document Analysis (CLI)

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
```

Or via API:

```bash
# 1. Start backend
cd D:\Clarion\Clarion-Backend
python main.py

# 2. Upload and analyze
curl -X POST "http://localhost:8000/upload" -F "file=@document.pdf"
curl -X POST "http://localhost:8000/analyze/{document_id}"
```

### 3.2 Relation Dataset Generation

Relations are automatically logged during document analysis.

```bash
# Check stats
python experiments/run_with_ollama.py --task stats

# Export for review
curl "http://localhost:8000/dataset/relations/export?format=json" -o relations.json
```

### 3.3 Manual Labeling

```bash
# Interactive labeling
python experiments/run_with_ollama.py --task label
```

Labeling Guidelines:
| Label | Criteria |
|-------|----------|
| `is_valid: true` | Relation is factually correct and meaningful |
| `is_valid: false` | Relation is hallucinated or incorrect |

### 3.4 Run Experiments

```bash
# Benchmark with Qwen
python experiments/run_with_ollama.py --model qwen3:latest --task benchmark --runs 10

# Benchmark with DeepSeek
python experiments/run_with_ollama.py --model deepseek-coder:6.7b --task benchmark --runs 10

# Compare models
python experiments/run_with_ollama.py --task compare
```

### 3.5 Ablation Study

```bash
python experiments/run_ablation_from_benchmarks.py --out-dir data/reports/ablation
```

### 3.6 Embedding Comparison

```bash
python -m core.experiments.embedding_comparison
```

### 3.7 Package Research Artifacts

```bash
python experiments/package_research_artifacts.py
```

---

## 4. Recommendations

### 4.1 Documents for Dataset Growth

| Document Type | Expected Relations | Value |
|---------------|-------------------|-------|
| Textbook chapters (CS, math) | 20-50 per chapter | High |
| Technical documentation | 15-30 per doc | Medium-High |
| Research papers | 10-25 per paper | High |
| Wikipedia articles (structured) | 10-20 per article | Medium |

### 4.2 Minimum Labeled Samples

| Experiment | Min Samples | Recommended |
|------------|-------------|-------------|
| Ablation study | 50 | 100+ |
| Embedding comparison | 30 | 80+ |
| Hallucination reduction | 40 | 100+ |

**Current Status**: 8 labeled records (need 30+ for experiments)

### 4.3 Best Experiment Order

1. **Data Collection** (Priority: High)
   - Process 5-10 documents
   - Generate 100+ relation records
   - Label 80%+ of records

2. **Baseline Benchmark** (Priority: High)
   - Run with default model (qwen3:latest)
   - Verify pipeline works

3. **Model Comparison** (Priority: Medium)
   - Compare qwen3 vs deepseek-coder vs gemma3
   - Document performance differences

4. **Ablation Study** (Priority: Medium)
   - Run with `--no-graph-features`
   - Compare against baseline

5. **Hallucination Analysis** (Priority: Medium)
   - Analyze false accept rate reduction
   - Focus on validation model effectiveness

6. **Embedding Comparison** (Priority: Low)
   - Test different embedding models
   - Optional optimization

7. **Package Artifacts** (Priority: High)
   - Generate tables and plots
   - Package for review

---

## 5. Switching LLM Providers

### 5.1 Local Ollama Models

```bash
# Via command line
python experiments/run_with_ollama.py --model gemma3:4b --task benchmark

# Via environment (PowerShell)
$env:OLLAMA_MODEL="deepseek-coder:6.7b"
python main.py
```

### 5.2 API Providers (Optional Later)

Not currently implemented. To add later:
```bash
# Would require API keys
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
```

---

## 6. Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `experiments/run_with_ollama.py` | Created | Unified experiment CLI |
| `docs/EXPERIMENT_WORKFLOW.md` | Created | Step-by-step guide |
| `AGENTS.md` | Updated | Quick reference |
| `.env.example` | Updated | Ollama configuration |
| `core/llm/base.py` | Updated | Added OLLAMA provider type |
| `core/llm/factory.py` | Updated | Ollama support in factory |

---

## 7. Current System Status

- **Dataset**: 8 labeled relations (5 types)
- **Models Available**: 3 (qwen3, deepseek-coder, gemma3)
- **Experiment Commands**: Ready to use
- **Documentation**: Complete

---

## 8. Next Steps

1. Process more documents to reach 100+ relations
2. Run initial benchmark with available models
3. Perform model comparison
4. Generate research artifacts

---

**Status**: Ready for experiment execution
