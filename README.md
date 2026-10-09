# Clarion: Intelligent Document Understanding, Knowledge Graph Generation, and Dataset Construction System

Clarion is a local-first research platform and document intelligence system combining structure-aware document parsing, dense vector retrieval, deterministic concept extraction, relation dataset construction, and a leak-free benchmarking and human ground-truth annotation suite.

---

## System Overview & Architecture

Clarion provides two interconnected pipelines:

1. **Document Intelligence Platform**: Ingests unstructured academic and technical documents (PDF, DOCX), performs structure-aware chunking, builds dense semantic vector indices, and generates grounded knowledge graphs and structured summaries.
2. **Empirical Research & Benchmark Harness**: Evaluates closed-domain knowledge graph relation validation with strict mathematical target-leakage prevention, document-isolated splits, statistical probability calibration, and an integrated human ground-truth annotation and adjudication system.

```text
DOCUMENT INGESTION & KNOWLEDGE MAPPING
[PDF / DOCX] ──► Text Extraction ──► Structure-Aware Chunking (Word Slices)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          Dense Embeddings (BGE-Large)               Lexical Concept Extraction
                       │                                           │
                       ▼                                           ▼
              FAISS Vector Index                     Co-Occurrence & Triples
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                               Knowledge Graph & Summaries
                                             │
                                             ▼
                                Candidate Relation Dataset
                                   (relation_dataset.db)

RESEARCH BENCHMARK & GROUND-TRUTH PIPELINE
[Candidate Relations] ──► Annotation Queue (Batch A & Overlap Batch B)
                                      │
                                      ▼
                      Local Web / CLI Annotation Console
                                      │
                                      ▼
                      Inter-Annotator Agreement (Kappa)
                                      │
                                      ▼
                       Disagreement Adjudication
                                      │
                                      ▼
                       Gold Dataset V1 Contract (Eligible & Provenance)
                                      │
                                      ▼
                  Leak-Free Benchmark Harness (5 Seeds)
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
Document-Isolated Splits                                Context-Manager Graph
(Zero Context Bleed)                                   Target-Leakage Removal
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      ▼
                       Multi-Signal Model & Baselines
                                      │
                                      ▼
                      Probability Calibration (Platt / Isotonic)
                                      │
                                      ▼
                      Percentile Bootstrap CI (B = 2,000)
```

---

## Core Principles & Scientific Integrity

- **Heuristic-First & Provider-Resilient**: Core extraction, chunking, graph generation, and validation run deterministically. When an LLM provider (Ollama, OpenAI, DeepSeek, Gemini) is connected, it enriches confidence scores and generative summaries. If the provider is offline, the entire pipeline remains functional via deterministic fallbacks.
- **Scientifically Honest Evidence Verification**: The evidence verification module implements directional lexical pattern matching and sentence-level evidence extraction. It does **not** employ a trained Natural Language Inference (NLI) model or claim deep semantic entailment.
- **Mathematical Target-Leakage Elimination**: Graph topological features (shortest path length, centrality, degree counts) are computed with candidate edges temporarily excised from the graph via a deterministic context manager, preventing trivial self-prediction.
- **Strict Gating for Real vs. Synthetic Benchmarks**: Synthetic regression fixtures verify algorithmic correctness and absence of leakage. Real-world empirical benchmark claims remain hard-blocked until human gold labels are completed and validated.

---

## Runtime Models & Schemas

### 1. Machine Learning & Embedding Models
- **Dense Embedding Model**: `BAAI/bge-large-en-v1.5` via `sentence-transformers`
- **Vector Index**: `faiss-cpu` local vector store
- **Compute Device**: Auto-resolved (`cuda`, `mps`, or `cpu`) via `EMBEDDING_DEVICE=auto`
- **LLM Provider Options**: `ollama` (default: `qwen3.5:4b`), `openai`, `deepseek`, `gemini`

### 2. Pydantic Schemas ([`Clarion-Backend/models/`](Clarion-Backend/models/))
- [`document.py`](Clarion-Backend/models/document.py): Ingestion metadata and job state tracking
- [`chunk.py`](Clarion-Backend/models/chunk.py): Structure-aware text segments and heading metadata
- [`embedding.py`](Clarion-Backend/models/embedding.py): Vector representations and retrieval queries
- [`knowledge_map.py`](Clarion-Backend/models/knowledge_map.py): Concepts, relation triples, and topic trees
- [`graph.py`](Clarion-Backend/models/graph.py): Cytoscape graph serialization formats
- [`ground_truth.py`](Clarion-Backend/models/ground_truth.py): Closed-domain ground-truth annotations, error taxonomies, and gold dataset records
- [`retrieval.py`](Clarion-Backend/models/retrieval.py): Similarity search results and scoring
- [`summary.py`](Clarion-Backend/models/summary.py): Hierarchical extractive and generative summaries
- [`response.py`](Clarion-Backend/models/response.py): Typed API response envelopes

---

## Repository Layout

```text
Clarion/
├── Clarion-Backend/          # FastAPI backend, storage, evaluation, and benchmark harness
│   ├── core/
│   │   ├── benchmarking/     # Canonical benchmark harness, splitters, manifests, annotation tools
│   │   ├── evaluation/       # Multi-signal validation, lexical evidence verifier, calibration
│   │   ├── experiments/      # Graph topological features, relation candidate extractors
│   │   └── llm/              # Provider adapters (Ollama, OpenAI, DeepSeek, Gemini)
│   ├── data/                 # Backend runtime storage (clarion.db, relation_dataset.db, queues)
│   ├── graph/                # Graph construction, hierarchy extraction, Cytoscape exporters
│   ├── models/               # Pydantic data schemas
│   ├── reports/              # Engineering validation and research protocol documentation
│   ├── routers/              # REST API endpoint handlers
│   ├── services/             # Pipeline orchestration, chunking, retrieval, embeddings
│   ├── tests/                # Comprehensive unit and integration test suite (120 tests)
│   └── utils/                # Configuration, logging, SQLite helpers
├── frontend/                 # Static web workspace for uploads, graph viewing, and query exploration
├── scripts/                  # Development scripts and end-to-end smoke test runner
├── data/                     # Top-level shared benchmarks and dataset mirrors
│   ├── benchmarks/           # Canonical benchmark results and benchmark registry
│   └── datasets/             # Human-readable exported dataset snapshots
└── appendix_report.md        # Technical architecture report and reference snippets
```

---

## Quick Start

### 1. Environment Setup

**Linux / macOS:**
```bash
cd Clarion-Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Windows (PowerShell):**
```powershell
cd Clarion-Backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Start the Backend API

```bash
# Run from Clarion-Backend directory
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- System Status: `http://127.0.0.1:8000/system-status`

### 3. Start the Frontend Workspace

In a separate terminal:

```bash
cd frontend
python -m http.server 3000
```

Open: `http://127.0.0.1:3000` (connects to backend at `http://127.0.0.1:8000`).

### 4. Run the Test Suite

```bash
cd Clarion-Backend
pytest tests/ -v
```

All 120 tests execute deterministically in sub-second to minute intervals without requiring external GPUs or live network connections.

---

## Human Annotation & Research Benchmarking

Clarion features a specialized CLI for closed-domain relation annotation and benchmarking.

### Launch Annotation Console (Local Web Interface)

- **Annotator A (Primary Queue — 200 Candidates)**:
  ```bash
  PYTHONPATH=Clarion-Backend python Clarion-Backend/core/benchmarking/annotation_tool.py serve-ui \
    --file Clarion-Backend/data/ground_truth/annotation_queue_batch1.json \
    --port 8080
  ```
  Access: `http://127.0.0.1:8080`

- **Annotator B (Independent Overlap Queue — 50 Candidates)**:
  ```bash
  PYTHONPATH=Clarion-Backend python Clarion-Backend/core/benchmarking/annotation_tool.py serve-ui \
    --file Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json \
    --port 8081
  ```
  Access: `http://127.0.0.1:8081`

### Compute Inter-Annotator Agreement & Adjudication

```bash
# Validate annotation queues
python Clarion-Backend/core/benchmarking/annotation_tool.py validate \
  --file Clarion-Backend/data/ground_truth/annotation_queue_batch1.json

# Calculate Cohen's & Fleiss' Kappa
python Clarion-Backend/core/benchmarking/annotation_tool.py agreement \
  --file-a Clarion-Backend/data/ground_truth/annotation_queue_batch1.json \
  --file-b Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json

# Extract disagreements for committee review
python Clarion-Backend/core/benchmarking/annotation_tool.py adjudicate \
  --file-a Clarion-Backend/data/ground_truth/annotation_queue_batch1.json \
  --file-b Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json \
  --output Clarion-Backend/data/ground_truth/adjudication_queue_batch1.json
```

### Execute Canonical Synthetic Benchmark

```bash
python Clarion-Backend/run_canonical_benchmark.py --runs 5 --seed 42
```

Outputs are automatically validated and recorded against `data/benchmarks/CANONICAL_BENCHMARK_REGISTRY.json`.

---

## Reproducibility Guidelines

For scientific reporting, ensure the following parameters are explicitly logged:

1. **Working Directory**: Always execute commands from `Clarion-Backend/` to maintain relative database path alignment.
2. **Configuration Pinning**: Document `EMBEDDING_MODEL_NAME`, resolved compute device, `CHUNK_SIZE` (in words), `CHUNK_OVERLAP`, and `SQLITE_TIMEOUT_SECONDS`.
3. **Execution Mode**: Explicitly disclose whether an evaluation ran in `SYNTHETIC_REGRESSION` mode or against human-adjudicated gold datasets.
4. **Resampling Parameters**: Document bootstrap resample iterations ($B=2000$) and random seeds ([42, 43, 44, 45, 46]).

---

## Documentation Links

- **Backend Architecture & API**: [Clarion-Backend/README.md](Clarion-Backend/README.md)
- **Frontend Workspace**: [frontend/README.md](frontend/README.md)
- **Script Helpers & Smoke Tests**: [scripts/README.md](scripts/README.md)
- **Contributing Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Engineering Validation Report**: [Clarion-Backend/reports/ENGINEERING_VALIDATION_REPORT.md](Clarion-Backend/reports/ENGINEERING_VALIDATION_REPORT.md)
- **Research Status & Annotation Protocol**: [Clarion-Backend/reports/RESEARCH_EXPERIMENT_REPORT.md](Clarion-Backend/reports/RESEARCH_EXPERIMENT_REPORT.md)
