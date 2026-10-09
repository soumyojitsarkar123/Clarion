# Clarion — Backend & Research Engineering Engine

FastAPI backend service and research benchmarking engine for document ingestion, semantic chunking, embedding generation, knowledge graph construction, relation dataset capture, statistical calibration, and closed-domain ground-truth adjudication.

---

## Architecture & System Capabilities

Clarion is a dual-function system designed for both document intelligence applications and rigorous empirical research:

1. **Document Ingestion & Semantic Chunking**: Accepts PDF and DOCX files, executing structure-aware word-slice chunking that preserves document hierarchy and headings without splitting across section boundaries.
2. **Dense Vector Indexing**: Computes dense text embeddings (`BAAI/bge-large-en-v1.5`) via `sentence-transformers` and builds local FAISS vector indices for similarity retrieval.
3. **Knowledge Graph & Mapping**: Extracts grounded concepts, relation tuples, and topic hierarchies using deterministic lexical extraction with optional LLM enrichment.
4. **Structured Grounding & Summaries**: Assembles hierarchical document summaries directly grounded in chunk context.
5. **Research Evaluation & Benchmarking**: Provides a leak-free benchmarking harness supporting multiple baselines, multi-signal validation, graph structural feature extraction, statistical calibration, and paired document-cluster bootstrap confidence intervals.
6. **Human Annotation & Adjudication Tooling**: Features a zero-dependency, local web-based and terminal-based annotation suite with atomic writes, automated backups, immutable candidate protection, inter-annotator agreement metrics, and gold dataset promotion contracts.

The backend is provider-aware but operates fully deterministically by default. Core extraction and evaluation algorithms run without requiring an active external LLM connection.

---

## Runtime Configuration

Configuration settings are managed via Pydantic Settings in [`utils/config.py`](utils/config.py). Overrides can be supplied via environment variables or a `.env` file.

| Component | Setting | Default | Description |
|---|---|---|---|
| Embedding model | `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-en-v1.5` | HuggingFace model path for dense embeddings |
| Embedding device | `EMBEDDING_DEVICE` | `auto` | Target device (`cuda`, `mps`, or `cpu`) |
| LLM provider | `LLM_PROVIDER` | `ollama` | Provider backend (`ollama`, `openai`, `deepseek`, `gemini`) |
| Ollama model | `OLLAMA_MODEL` | `qwen3.5:4b` | Default model tag when Ollama is active |
| Ollama base URL | `OLLAMA_API_BASE` | `http://localhost:11434/v1` | Endpoint for Ollama API |
| Chunk size | `CHUNK_SIZE` | `512` | Window size in words |
| Chunk overlap | `CHUNK_OVERLAP` | `50` | Word overlap between adjacent slices |
| SQLite timeout | `SQLITE_TIMEOUT_SECONDS` | `30.0` | Maximum lock wait time for SQLite connections |

---

## Deterministic Execution vs. Live LLM Enrichment

Clarion implements a dual-mode execution strategy:

### Deterministic Components (Always Active)
- Structure-aware document text extraction & word chunking — [`services/chunking_service.py`](services/chunking_service.py)
- Heuristic knowledge-map concept and co-occurrence extraction — [`services/knowledge_map_service.py`](services/knowledge_map_service.py)
- Relation candidate dataset logging & SQLite persistence — [`services/relation_dataset_service.py`](services/relation_dataset_service.py)
- Lexical evidence support & sentence grounding — [`core/evaluation/evidence_verifier.py`](core/evaluation/evidence_verifier.py)
- Extractive summary generation fallback — [`services/summary_service.py`](services/summary_service.py)
- Research benchmark harness & splitters — [`core/benchmarking/`](core/benchmarking/)

### Live Enrichment Components (Require Active Provider)
- Subjective LLM confidence scoring on candidate relations
- Natural language query answering via Retrieval-Augmented Generation (RAG)
- Generative summary synthesis and topic enrichment

> If a configured LLM provider is unreachable, the system logs an informational warning and falls back to deterministic extraction without halting execution.

---

## Data Models

Pydantic schemas reside in [`models/`](models/):

| Schema File | Core Models | Description |
|---|---|---|
| [`document.py`](models/document.py) | `Document`, `DocumentCreate`, `DocumentStatus` | Document metadata and ingestion tracking |
| [`chunk.py`](models/chunk.py) | `Chunk`, `ChunkCreate` | Structure-aware text segments and heading metadata |
| [`embedding.py`](models/embedding.py) | `EmbeddingItem`, `VectorSearchQuery` | Vector embeddings and similarity search schemas |
| [`knowledge_map.py`](models/knowledge_map.py) | `Concept`, `Relation`, `KnowledgeMap` | Concepts, relational triples, and topic nodes |
| [`graph.py`](models/graph.py) | `GraphData`, `NodeData`, `EdgeData` | Cytoscape-compatible graph representation |
| [`ground_truth.py`](models/ground_truth.py) | `GroundTruthRecord`, `GroundTruthDataset`, `ValidityLabel`, `ErrorCategory` | Closed-domain relation annotations, validation categories, and gold dataset contracts |
| [`retrieval.py`](models/retrieval.py) | `RetrievalResult`, `RetrievalContext` | FAISS similarity search payloads |
| [`summary.py`](models/summary.py) | `DocumentSummary`, `SummarySection` | Structured hierarchical summaries |
| [`response.py`](models/response.py) | `StandardResponse`, `HealthResponse` | Unified API response envelopes |

---

## Storage & Artifact Layout

All runtime and research artifacts are resolved relative to `Clarion-Backend/data/`:

| Directory / File | Description |
|---|---|
| `data/clarion.db` | Primary application database (documents, chunk text, job records) |
| `data/relation_dataset.db` | Candidate relation database with feature scores and metadata |
| `data/ground_truth/` | Annotation queues (`annotation_queue_batch1.json`, overlap queues), annotation protocol, and synthetic benchmark fixtures |
| `data/benchmarks/` | Canonical benchmark artifacts and registry (`CANONICAL_BENCHMARK_REGISTRY.json`) |
| `data/graphs/` | Per-document Cytoscape-compatible JSON graph exports |
| `data/datasets/` | Document-level relation snapshots (CSV/JSON) |
| `data/vectorstore/` | Serialized FAISS index files |
| `logs/clarion.log` | Runtime execution log (git-ignored) |

---

## API Surface

Routers are mounted in [`main.py`](main.py):

| Route Prefix | Primary Functionality |
|---|---|
| `/upload` | PDF/DOCX upload, parsing, and document registration |
| `/analyze` | Trigger full processing pipeline (chunking, embedding, mapping) |
| `/knowledge-map` | Retrieve extracted concepts, relations, and topic hierarchy |
| `/query` | RAG-based search and grounded question answering |
| `/summary` | Retrieve structured extractive or generative summaries |
| `/status` | Job queue tracking, cancellation, and recovery |
| `/dataset` | Query, filter, and export relation candidate datasets |
| `/dataset-factory` | Batch training sample generation and export |
| `/graph` | Cytoscape-formatted graph topology export |
| `/logs` | Access live execution log streams |
| `/system` | Diagnostic health checks, storage stats, and RAG audits |

Interactive documentation is available at `http://127.0.0.1:8000/docs` (Swagger UI) and `http://127.0.0.1:8000/redoc`.

---

## Research & Benchmarking Suite

Clarion includes a standalone research CLI for canonical synthetic regression benchmarking and human ground-truth annotation.

### 1. Canonical Benchmark Execution

Run the deterministic 5-seed benchmark harness across all baseline models, calibrators, and ablation configurations:

```bash
# From Clarion-Backend directory
python run_canonical_benchmark.py --runs 5 --seed 42
```

Options:
- `--runs`: Number of consecutive seeds (default: 5, seeds [42, 43, 44, 45, 46])
- `--seed`: Base random seed (default: 42)
- `--graph-features`: Include graph structural topological features
- `--dataset`: Path to fixture or dataset (`.json` or `.db`)

> **Scientific Integrity Notice:** Running against `synthetic_benchmark_fixture.json` operates in `SYNTHETIC_REGRESSION` mode. This verifies pipeline correctness, absence of leakage, and numerical reproducibility. Real-data research execution is hard-blocked until human gold labels are supplied.

### 2. Human Annotation & Adjudication CLI

The annotation workflow is managed via [`core/benchmarking/annotation_tool.py`](core/benchmarking/annotation_tool.py):

```bash
# Launch multi-threaded web annotation UI for primary batch (port 8080)
python core/benchmarking/annotation_tool.py serve-ui \
  --file data/ground_truth/annotation_queue_batch1.json \
  --port 8080

# Launch multi-threaded web annotation UI for independent overlap batch (port 8081)
python core/benchmarking/annotation_tool.py serve-ui \
  --file data/ground_truth/annotation_queue_batch1_overlap_b.json \
  --port 8081

# Launch interactive terminal session
python core/benchmarking/annotation_tool.py annotate \
  --file data/ground_truth/annotation_queue_batch1.json \
  --annotator-id AnnA

# Validate queue schema and protocol integrity
python core/benchmarking/annotation_tool.py validate \
  --file data/ground_truth/annotation_queue_batch1.json

# Compute inter-annotator agreement (Cohen's Kappa & Fleiss' Kappa)
python core/benchmarking/annotation_tool.py agreement \
  --file-a data/ground_truth/annotation_queue_batch1.json \
  --file-b data/ground_truth/annotation_queue_batch1_overlap_b.json

# Extract disagreements for adjudication committee review
python core/benchmarking/annotation_tool.py adjudicate \
  --file-a data/ground_truth/annotation_queue_batch1.json \
  --file-b data/ground_truth/annotation_queue_batch1_overlap_b.json \
  --output data/ground_truth/adjudication_queue_batch1.json

# Assemble verified Gold Dataset V1
python core/benchmarking/annotation_tool.py assemble-gold \
  --primary data/ground_truth/annotation_queue_batch1.json \
  --secondary data/ground_truth/annotation_queue_batch1_overlap_b.json \
  --output-gold data/ground_truth/gold_dataset_v1.json
```

---

## Scientific Rigor & Methodological Guarantees

### 1. Mathematical Target-Leakage Prevention
When computing structural graph features (shortest path length, centrality, degree counts) for candidate edge $(A, B)$, the candidate edge, its reciprocal $(B, A)$, and self-loops $(A, A)$ are temporarily removed from knowledge graph $G$ using a deterministic Python context manager (`GraphFeatureExtractor._without_candidate_edges`). This guarantees that topological distance metrics reflect alternative connectivity rather than the candidate edge itself.

### 2. Document-Level Data Isolation
Splits are strictly partitioned by source document identifier ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset$, $\mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$, $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$). Intra-document sentence pairs are never split across training and evaluation sets. `SplitResult.assert_no_document_leakage()` enforces this invariant before fitting.

### 3. Lexical Evidence Grounding vs. Trained NLI
The evidence verifier implements directional pattern matching and sentence-level evidence extraction. It does **not** employ a trained Natural Language Inference (NLI) model or assert deep semantic entailment. All reported scores reflect explicit lexical support.

### 4. Non-Leaking Calibration
Probability calibrators (Platt scaling, Isotonic regression, temperature scaling) are fitted strictly on validation split probabilities and evaluated on independent test splits. Calibration metrics include Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.

### 5. Percentile Bootstrap Confidence Intervals
Confidence intervals for all benchmark metrics are estimated using paired document-cluster percentile bootstrap resampling ($B = 2,000$ iterations). Bounds are strictly guaranteed to lie within the empirical support range $[\min(v), \max(v)]$.

---

## Installation & Development Setup

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

### 2. Run Backend Service

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 3. Run Test Suite

```bash
pytest tests/ -v
```

All 120 tests run in complete isolation using temporary test databases and deterministic mock paths without requiring network access or external GPU dependencies.
