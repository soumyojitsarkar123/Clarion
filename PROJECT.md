# Project: Clarion — Document Intelligence & Research-Product UI

## Architecture
- **System**: Clarion Document Understanding & Knowledge Mapping System
- **Backend**: FastAPI (`Clarion-Backend/`), SQLite (`data/clarion.db`), FAISS vector store, SentenceTransformers (`BAAI/bge-large-en-v1.5`), NetworkX, Ollama LLM integration
- **Frontend**: Zero-build single-page web client (`frontend/index.html`, `frontend/styles.css`, `frontend/app.js`), Cytoscape.js visualization, REST API + WebSockets
- **Research Evaluation Isolation**: Quarantined benchmark registry (`data/benchmarks/`), Ground Truth & Annotation queues (`Clarion-Backend/data/ground_truth/`). Strictly separated from production/development runs.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Baseline Telemetry Capture | Measure unoptimized AI_Book.pdf baseline metrics into docs/AI_BOOK_BASELINE.md | M0 | ORIGINAL_REQUEST § 2 |
| 2 | Query Test FAISS Dimension Fix | Fix test_api.py unmocked generate_embeddings_async and 384 vs 1024 dim mismatch | M0 | Explorer 1 & 3 Handoffs |
| 3 | Chunking Engine Overhaul | Prevent running header/footer false sections, enforce min chunk size, eliminate micro-chunks | M1 | ORIGINAL_REQUEST § 3 |
| 4 | Distributed Content Sampling | Stratified/distributed sampling across document; detect front matter, suppress contributor rosters | M1 | ORIGINAL_REQUEST § 4 |
| 5 | Concept Extraction Refinement | Discover multi-word technical concepts/noun phrases; suppress unigrams "Some", "These", "Book", "Editor", etc. | M1 | ORIGINAL_REQUEST § 5 |
| 6 | Candidate Semantic Relations | Extract candidate relations with source chunk grounding; explicitly label as candidate relations | M1 | ORIGINAL_REQUEST § 5, 11 |
| 7 | Multi-Class Document Classification | Multi-signal classifier: textbook/reference, academic paper, report, exam, tech docs, notes, unknown | M1 | ORIGINAL_REQUEST § 6 |
| 8 | Grounded Summary Generation | Grounded multi-section summary; eliminate acknowledgments/contributor dominance | M1 | ORIGINAL_REQUEST § 10 |
| 9 | Honest LLM Runtime States | Support LLM_AVAILABLE, LLM_DEGRADED, LLM_UNAVAILABLE, FALLBACK_USED; COMPLETED, DEGRADED, FAILED | M1 | ORIGINAL_REQUEST § 7, 13 |
| 10 | BGE-Large Performance & CPU Optimization | Preserve BAAI/bge-large-en-v1.5; model reuse, batching, thread pool config, GPU check | M1 | ORIGINAL_REQUEST § 8 |
| 11 | Computation Reuse & Cache Keying | Robust hashing (document_id + chunk_hash + config + model_name) for embeddings & chunking | M1 | ORIGINAL_REQUEST § 9 |
| 12 | Honest Status Header | Display Document Name, Analysis Mode, Active Model, Processing Time, and Honest Status | M2 | ORIGINAL_REQUEST § 12, 13 |
| 13 | Document Overview Metric Deck | 8 grounded metrics: Words, Pages, Sections, Chunks, Concepts, Candidate Relations, Nodes, Edges | M2 | ORIGINAL_REQUEST § 12 |
| 14 | Key Insights Display | 3–5 grounded document finding cards with topic tags | M2 | ORIGINAL_REQUEST § 12 |
| 15 | Knowledge Map Interaction | Cytoscape graph with node search, zoom, fit, relation filtering, and selected-node inspector drawer | M2 | ORIGINAL_REQUEST § 12 |
| 16 | Top Concepts Table | Dedicated concept table with frequency/occurrence count, supporting sections, and centrality | M2 | ORIGINAL_REQUEST § 12 |
| 17 | Candidate Relations Panel | Interactive list: Concept A → Relation → Concept B with source section/chunk citations | M2 | ORIGINAL_REQUEST § 11, 12 |
| 18 | Document Structure Outline | Sections/chapters/hierarchy from /summary/{document_id} exposed in UI | M2 | ORIGINAL_REQUEST § 12 |
| 19 | Ask The Document (RAG Interface) | Interactive query box calling POST /query/{document_id} with answer & grounded source chunks | M2 | ORIGINAL_REQUEST § 12 |
| 20 | Technical Details Panel | Expandable panel with stage durations (chunking, embedding, LLM, graph, total), device, model | M2 | ORIGINAL_REQUEST § 12 |
| 21 | Processing Timeline (Duration-Aware) | Stage, Status, Duration replacing generic progress bars | M2 | ORIGINAL_REQUEST § 14 |
| 22 | Section 15 Regression Test Suite | 11 test categories for chunking, sampling, concepts, classification, LLM states, fallback | M3 | ORIGINAL_REQUEST § 15 |
| 23 | Dual-Track E2E Test Suite | Requirement-driven Tiers 1-4 test suite; publish TEST_READY.md | M3 | Project Pattern Dual Track |
| 24 | AI_Book.pdf Clean Re-Analysis | End-to-end run on clean AI_Book.pdf; Before vs After empirical comparison | M4 | ORIGINAL_REQUEST § 16 |
| 25 | Research Evaluation Quarantining | Verify zero mutation to benchmark registries, gold contracts, and annotation queues | M4 | ORIGINAL_REQUEST § 17 |
| 26 | Final Review & Master Report | Full pytest, backend startup, frontend startup, Ollama test, git diff, report sections A-M | M4 | ORIGINAL_REQUEST § 18, 19 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M0 | Baseline Telemetry & Test Fix | Measure unoptimized AI_Book.pdf baseline into `docs/AI_BOOK_BASELINE.md`; fix `test_api.py::test_query_after_analysis` | none | DONE (`docs/AI_BOOK_BASELINE.md`, 120/121 tests pass, 1 skip) |
| M1 | Backend Document Intelligence (Workstream A) | Overhaul chunking, sampling, concept extraction, classification, LLM states, BGE-large CPU optimization, and caching | M0 | IN_PROGRESS (conv d4ea8529-297d-42d8-b41e-2aff77e2d732) |
| M2 | Research-Product UI Redesign (Workstream B) | Redesign `index.html`, `styles.css`, `app.js` with RAG Q&A, structure, honest status, timeline, overview, inspector | M0 | PLANNED |
| M3 | Regression Suite & E2E Testing (Workstream C) | Add § 15 regression tests (11 categories) and Tiers 1-4 E2E test suite; publish `TEST_READY.md` | M1, M2 | PLANNED |
| M4 | Final AI_Book E2E Verification & Review | Clean AI_Book run, before/after comparison, research integrity check, Section 18 review, master report | M1, M2, M3 | PLANNED |

## Interface Contracts
### Backend ↔ Frontend
- `GET /system/pipeline-status?document_id={id}`:
  ```json
  {
    "document_id": "str",
    "status": "running" | "completed" | "degraded" | "failed",
    "analysis_mode": "llm" | "deterministic_fallback",
    "active_model": "str",
    "total_duration_sec": 0.0,
    "stages": {
      "ingestion": { "status": "completed", "duration_sec": 0.0 },
      "chunking": { "status": "completed", "duration_sec": 0.0, "chunk_count": 0 },
      "embedding": { "status": "completed", "duration_sec": 0.0, "throughput_chunks_sec": 0.0, "device": "cpu" },
      "llm_analysis": { "status": "completed" | "fallback", "duration_sec": 0.0, "model": "str" },
      "graph_building": { "status": "completed", "duration_sec": 0.0, "node_count": 0, "edge_count": 0 }
    }
  }
  ```
- `GET /summary/{document_id}`:
  ```json
  {
    "document_id": "str",
    "title": "str",
    "document_type": "textbook" | "academic_paper" | "report" | "assessment" | "technical_documentation" | "notes" | "unknown",
    "overall_summary": "str",
    "sections": [
      {
        "id": "str",
        "title": "str",
        "level": 1,
        "summary": "str",
        "key_points": ["str"],
        "related_concepts": ["str"]
      }
    ]
  }
  ```
- `GET /knowledge-map/{document_id}`:
  ```json
  {
    "document_id": "str",
    "concepts": [
      { "id": "str", "name": "str", "frequency": 0, "supporting_sections": ["str"], "definition": "str" }
    ],
    "relations": [
      { "id": "str", "from_concept": "str", "to_concept": "str", "relation_type": "str", "is_candidate": true, "source_chunk_id": "str", "confidence": 0.0 }
    ]
  }
  ```
- `POST /query/{document_id}`:
  - Input: `{"query": "str", "top_k": 5}`
  - Output: `{"query": "str", "response": "str", "results": [{"chunk_id": "str", "content": "str", "score": 0.0, "section_title": "str"}]}`

## Code Layout
- Exclusive File Boundaries:
  - **Milestone 0 (Baseline & Test Fix)**:
    - `scripts/measure_ai_book_baseline.py` (new)
    - `docs/AI_BOOK_BASELINE.md` (new)
    - `Clarion-Backend/tests/test_api.py` (test fix only)
  - **Milestone 1 (Workstream A - Backend)**:
    - `Clarion-Backend/services/chunking_service.py`
    - `Clarion-Backend/services/knowledge_map_service.py`
    - `Clarion-Backend/services/summary_service.py`
    - `Clarion-Backend/services/embedding_service.py`
    - `Clarion-Backend/services/processing_pipeline.py`
    - `Clarion-Backend/core/llm/`
    - `Clarion-Backend/routers/status.py`
    - `Clarion-Backend/routers/system.py`
  - **Milestone 2 (Workstream B - Frontend)**:
    - `frontend/index.html`
    - `frontend/styles.css`
    - `frontend/app.js`
  - **Milestone 3 (Workstream C - QA & Tests)**:
    - `Clarion-Backend/tests/test_regression_section15.py` (new)
    - `tests/e2e/` (new E2E suite)
    - `TEST_READY.md` (new)
  - **Milestone 4 (Final E2E Verification)**:
    - `docs/AI_BOOK_BEFORE_AFTER_COMPARISON.md`
    - `docs/FINAL_COMPREHENSIVE_REPORT.md`
  - **STRICTLY PROTECTED (READ-ONLY, NO MODIFICATION)**:
    - `data/benchmarks/*`
    - `Clarion-Backend/data/ground_truth/*`
