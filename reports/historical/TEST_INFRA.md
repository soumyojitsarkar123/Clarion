# E2E Test Infra: Clarion Final Remediation & Publication Readiness

## 1. Test Philosophy
- **Opaque-Box & Requirement-Driven**: The test suite is derived strictly from user requirements in `ORIGINAL_REQUEST.md` (R1-R5) and user-visible APIs/contracts. Tests verify system behavior through public HTTP interfaces (`TestClient`), CLI scripts, and non-interactive DOM/JavaScript contract evaluators without coupling to private implementation classes.
- **Progressive Testability**: Verification mechanisms do not depend on features more complex than what is being tested. Earliest tiers provide clear pass/fail signals even before downstream features are fully wired.
- **Strict Environment & Database Isolation**: Tests operate in temporary isolated directories (`/tmp/clarion-e2e-XXXXXX`). The production database at `/mnt/d/Clarion/Clarion-Backend/data/clarion.db` (34,533,376 bytes) is monitored before and after every test session, with hard assertions guaranteeing zero modification.
- **Methodology**: Systematic 4-tier testing combining Category-Partition, Boundary Value Analysis (BVA), Pairwise Combinatorial Testing, and Real-World Workload Scenarios.

---

## 2. Feature Inventory Mapping
Mapped directly from `ORIGINAL_REQUEST.md` (R1-R5) and `PROJECT.md`:

| # | Feature | Category | Source Requirement | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|----------|-------------------|:------:|:------:|:------:|:------:|
| F1 | Genuine Stage Metrics Passthrough | Timeline & Telemetry | R1, R3 | 5 | 5 | ✓ | ✓ |
| F2 | Eliminate Synthetic `llm_analysis` Row | Timeline & Telemetry | R1, R3 | 5 | 5 | ✓ | ✓ |
| F3 | Authoritative 7-Stage Sequence & Telemetry Schema | Timeline & Telemetry | R1, R3 | 5 | 5 | ✓ | ✓ |
| F4 | Sub-second Duration Formatting (`ms`/`s`, no `0.00s`) | Frontend Timeline | R1 | 5 | 5 | ✓ | ✓ |
| F5 | Semantic Disambiguation Labels & Tooltips | Frontend Timeline | R1 | 5 | 5 | ✓ | ✓ |
| F6 | Frontend Timeline Conformance & Invariants | Frontend Timeline | R1, R3 | 5 | 5 | ✓ | ✓ |
| F7 | Per-Chunk / Small-Batch Dataset Generation | Dataset Factory | R2 | 5 | 5 | ✓ | ✓ |
| F8 | Full Provenance Preservation (doc/chunk/snippet) | Dataset Factory | R2 | 5 | 5 | ✓ | ✓ |
| F9 | Ollama Reasoning Suppression (`think: false`) | Ollama Integration | R2 | 5 | 5 | ✓ | ✓ |
| F10 | SQLite Checkpointing & Resumption | Dataset Factory | R2 | 5 | 5 | ✓ | ✓ |
| F11 | Bounded Retries, Backoff & JSON Schema Validation | Dataset Factory | R2 | 5 | 5 | ✓ | ✓ |
| F12 | Research Integrity & Gold Export Gate (HTTP 409) | Export Gate | R2 | 5 | 5 | ✓ | ✓ |
| F13 | Detailed Structured Attempt Telemetry | Observability | R2 | 5 | 5 | ✓ | ✓ |
| F14 | Baseline Backend Regression Preservation (235 tests) | Regression Suite | R4 | 5 | 5 | ✓ | ✓ |
| F15 | Production Database Isolation & Safety | Regression Safety | R4 | 5 | 5 | ✓ | ✓ |
| F16 | Live vs Mocked Model Distinction & Transparency | Model Verification | R4 | 5 | 5 | ✓ | ✓ |
| F17 | Markdown Deliverables & Honest Publication Classifications | Deliverables | R5 | 5 | 5 | ✓ | ✓ |

Total Identified Features: 17

---

## 3. Test Architecture & Directory Layout

### 3.1 Directory Structure
```
tests/e2e/
├── __init__.py
├── conftest.py                           # Root E2E fixtures: DB isolation, TestClient, production DB safety guard
├── run_e2e_suite.py                      # Standalone CLI runner with tier filtering & summary report
├── test_helpers.py                       # Shared test utilities (payload generators, DOM inspectors)
├── tier1_feature_coverage/               # Tier 1: 85 test cases (5 per feature F1-F17)
│   ├── test_t1_f01_genuine_metrics.py
│   ├── test_t1_f02_no_synthetic_row.py
│   ├── test_t1_f03_telemetry_schema.py
│   ├── test_t1_f04_duration_formatting.py
│   ├── test_t1_f05_semantic_labels.py
│   ├── test_t1_f06_frontend_invariants.py
│   ├── test_t1_f07_chunk_batching.py
│   ├── test_t1_f08_provenance.py
│   ├── test_t1_f09_ollama_parameters.py
│   ├── test_t1_f10_checkpoints.py
│   ├── test_t1_f11_retries_validation.py
│   ├── test_t1_f12_export_gate.py
│   ├── test_t1_f13_attempt_telemetry.py
│   ├── test_t1_f14_baseline_regression.py
│   ├── test_t1_f15_database_safety.py
│   ├── test_t1_f16_model_transparency.py
│   └── test_t1_f17_deliverables_readiness.py
├── tier2_boundary_corner/                # Tier 2: 85 boundary & corner cases (5 per feature F1-F17)
│   ├── test_t2_b01_metrics_boundaries.py
│   ├── test_t2_b02_synthetic_row_guards.py
│   ├── test_t2_b03_schema_corrupted.py
│   ├── test_t2_b04_subsecond_limits.py
│   ├── test_t2_b05_disambiguation_limits.py
│   ├── test_t2_b06_frontend_dom_limits.py
│   ├── test_t2_b07_chunk_limits.py
│   ├── test_t2_b08_provenance_boundaries.py
│   ├── test_t2_b09_reasoning_edge_cases.py
│   ├── test_t2_b10_checkpoint_recovery.py
│   ├── test_t2_b11_retry_exhaustion.py
│   ├── test_t2_b12_export_gate_corner.py
│   ├── test_t2_b13_telemetry_missing.py
│   ├── test_t2_b14_regression_boundaries.py
│   ├── test_t2_b15_db_concurrency_locks.py
│   ├── test_t2_b16_mock_leakage_guards.py
│   └── test_t2_b17_deliverable_validation.py
├── tier3_cross_feature/                  # Tier 3: 17 pairwise integration combinations
│   ├── test_t3_x01_ingest_metrics_passthrough.py
│   ├── test_t3_x02_chunking_provenance.py
│   ├── test_t3_x03_mapping_graph_disambiguation.py
│   ├── test_t3_x04_ollama_checkpoint_resume.py
│   ├── test_t3_x05_export_gate_telemetry.py
│   ├── test_t3_x06_db_safety_regression.py
│   ├── test_t3_x07_frontend_timeline_cold_recovery.py
│   ├── test_t3_x08_format_duration_pipeline_poll.py
│   ├── test_t3_x09_retry_backoff_checkpoint_sync.py
│   ├── test_t3_x10_model_switch_reasoning_flags.py
│   ├── test_t3_x11_gold_export_partial_state.py
│   ├── test_t3_x12_statistics_attempt_correlation.py
│   ├── test_t3_x13_degraded_pipeline_ui_conformance.py
│   ├── test_t3_x14_large_document_chunk_split_export.py
│   ├── test_t3_x15_rag_audit_evaluation_alignment.py
│   ├── test_t3_x16_multi_tenant_isolation_sessions.py
│   └── test_t3_x17_deliverables_git_integrity.py
└── tier4_real_world_scenarios/           # Tier 4: 9 realistic application scenarios
    ├── test_t4_s01_academic_paper_pipeline.py
    ├── test_t4_s02_technical_slides_pipeline.py
    ├── test_t4_s03_textbook_chapter_pipeline.py
    ├── test_t4_s04_dataset_factory_resilience.py
    ├── test_t4_s05_gold_dataset_export_audit.py
    ├── test_t4_s06_concurrent_uploads.py
    ├── test_t4_s07_cold_recovery_pipeline.py
    ├── test_t4_s08_adversarial_corrupt_ingest.py
    └── test_t4_s09_full_publication_audit.py
scripts/
└── run_all_e2e_tests.sh                  # Standalone execution script
```

### 3.2 Test Runner Invocation & Pass/Fail Semantics
- Python Runtime: `/mnt/d/Clarion/.venv_linux/bin/pytest`
- Direct pytest invocation:
  ```bash
  PYTHONPATH=Clarion-Backend /mnt/d/Clarion/.venv_linux/bin/pytest tests/e2e/ -v
  ```
- Tier-specific execution:
  ```bash
  PYTHONPATH=Clarion-Backend /mnt/d/Clarion/.venv_linux/bin/pytest tests/e2e/tier1_feature_coverage/ -v
  PYTHONPATH=Clarion-Backend /mnt/d/Clarion/.venv_linux/bin/pytest tests/e2e/tier2_boundary_corner/ -v
  PYTHONPATH=Clarion-Backend /mnt/d/Clarion/.venv_linux/bin/pytest tests/e2e/tier3_cross_feature/ -v
  PYTHONPATH=Clarion-Backend /mnt/d/Clarion/.venv_linux/bin/pytest tests/e2e/tier4_real_world_scenarios/ -v
  ```
- Unified Runner Script:
  ```bash
  bash scripts/run_all_e2e_tests.sh
  ```
- Pass/Fail Semantics:
  - Exit code `0`: All tests passed without regressions or production database mutations.
  - Non-zero exit code: Any assertion failure, unhandled exception, or database mutation triggers immediate failure.

---

## 4. Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity | Validation Objective |
|---|----------|--------------------|------------|----------------------|
| 1 | Academic Paper Pipeline (BSM 301 Mathematics) | F1, F3, F4, F5, F6, F14, F15 | High | Multi-page PDF ingestion through 7 stages, verifying metrics and UI formatting |
| 2 | Technical Lecture Slides (Thevenin Theorem) | F1, F2, F3, F5, F8, F15 | Medium | Structured presentation ingestion, concept mapping, and relation extraction |
| 3 | Textbook Chapter (DAA Graph Algorithms) | F1, F3, F7, F8, F10, F15 | High | Deep semantic chunking, provenance tracking, and knowledge graph construction |
| 4 | Dataset Factory Resilience Under Disruption | F7, F9, F10, F11, F13 | High | Per-chunk generation with simulated intermittent failure, backoff, and checkpoint resume |
| 5 | Research Gold Export Audit | F8, F9, F11, F12, F13 | High | Strict validation under `require_llm_for_export=True`, refusing partial exports (409) and verifying zero heuristic fallback |
| 6 | Concurrent Multi-Document Ingestion | F1, F3, F10, F14, F15 | High | Concurrent processing of multiple documents without cross-document state leakage or lock contention |
| 7 | Cold-System Restart Recovery | F1, F3, F6, F10, F15 | Medium | Persistence check in SQLite, recovering full stage telemetry after server restart |
| 8 | Adversarial Corrupt File Ingestion | F1, F3, F6, F14, F15 | Medium | Graceful rejection of corrupted files with informative stage error telemetry |
| 9 | Full Publication Deliverables Audit | F14, F15, F16, F17 | Medium | Automated audit of all 8 markdown deliverables for completeness and honest status tags |

---

## 5. Coverage Thresholds
- **Tier 1 (Feature Coverage)**: Exactly 85 test cases (5 test cases per feature across F1 to F17).
- **Tier 2 (Boundary & Corner Cases)**: Exactly 85 test cases (5 test cases per feature across F1 to F17).
- **Tier 3 (Cross-Feature Combinations)**: 17 pairwise integration test cases.
- **Tier 4 (Real-World Scenarios)**: 9 realistic application scenario test cases.
- **Total Minimum Suite**: **196 test cases**.
- **Database Safety Invariant**: 100% of test runs must leave `/mnt/d/Clarion/Clarion-Backend/data/clarion.db` unmodified.
