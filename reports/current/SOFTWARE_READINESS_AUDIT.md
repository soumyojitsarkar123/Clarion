# Software readiness audit

**Date:** 2026-11-11. **Status:** PARTIALLY VERIFIED. This audit reflects source inspection and the current targeted token-budget run; it is not a security certification or release approval.

| Area | Actual implementation and contract | Readiness / known limit |
|---|---|---|
| PDF/DOCX ingestion | `utils/file_handler.py`; PDF pages extracted with PyPDF2 then joined and normalized; DOCX via python-docx. | Text extraction works on recorded corpus; page coordinate map absent, scanned OCR absent, fidelity not corpus-validated. |
| Normalization | Regex repairs, camel/acronym splitting, whitespace collapse in `normalize_extracted_text`. | Transform changes offsets and has no reversible map. |
| Chunking | `services/chunking_service.py`; 512 words, 50 overlap, minimum 100; section-aware, word slicing and undersized merging. | Defaults preserved. Word threshold is not tokenizer limit; merged/reindexed `position_index` is not a reliable original source coordinate; no current alternative strategy benchmark. |
| Embeddings | `services/embedding_service.py`; BGE configured; runtime max length, fast tokenizer offsets, special-token count, exact chunk-local spans; isolated SQLite/FAISS run. | BGE path was previously exercised; `check_token_length` may estimate when tokenizer absent. Model-consistency metadata/version migration not fully established. |
| Vector persistence | SQLite embedding/segment metadata plus FAISS vector store; retrieval maps vector segment IDs to parent chunks. | Count/order aligned in captured isolated run. No model digest/index manifest guarantee found. |
| Entity/relation extraction | `services/knowledge_map_service.py`, evidence verifier and fallback heuristics. | Candidate relations and heuristic outputs are not human-validated gold; relation DB labels absent. |
| Graph construction | Knowledge-map and graph modules persist concepts, relations and graph views. | Structural output is implemented; semantic accuracy and source-page evidence not established. |
| Retrieval | `RetrievalService` dense FAISS search; score transformation currently `1/(1+distance)`; no verified BM25/hybrid/reranker path in the active query flow. | Score is ranking transformation, not calibrated probability. No relevance benchmark. |
| Context/generation | Query router builds one prompt and calls `LLMInterface`; Ollama native path uses `num_ctx`, `num_predict`; fallback is deterministic. | New budget service now packs complete ranked evidence and captures usage/truncation. Qwen preflight is estimated unless matching tokenizer configured; LLM call paths outside query and dataset-factory are only partially integrated. |
| Dataset generation | `DatasetFactoryService`; per-chunk generation, schema validation, attempts/checkpoints/retries, strict LLM export policy. | Prior smoke and mocked tests exist. Token estimates in this module are char-based and are not yet replaced by central service; gold quality remains unverified. |
| Annotation | UI, candidate DB, protocol/template. | 1,562 canonical candidates are unlabeled; no IAA. |
| Observability | `pipeline_observability.py`; seven canonical stages and null unavailable metrics. | Prior contract verified; token metrics currently returned on query, not yet persisted across all stages. |
| API/frontend | FastAPI routers; frontend status/query flows; prior UI tests use stub API. | Targeted query response contract passed. Live backend browser flow not verified in this assignment. |
| Security/data | SQLite, upload validation, CORS/settings, structured errors and logs. | No full threat model or adversarial security review. Do not claim certification. |
| Recovery/migrations | SQLite schemas initialized by service code and migration/recovery notes. | No production migration executed; no claim of safe general migration framework. |

## Concrete changes in this pass

Added `services/token_budget_service.py`, configurable context/output/safety limits, query evidence packing with duplicate and omission reporting, optional exact local tokenizer support, provider usage and truncation metadata, and focused tests. Runtime context remains 4,096; Ollama output cap remains 800; cloud provider output cap remains configured up to 2,000. BGE, chunk defaults, databases, and existing model choices were not changed.

## Release-level gaps

Source-page provenance; exact generation-model tokenizer access; token accounting across dataset factory and all stage telemetry; complete retrieval/graph/human evaluation; controlled model/resource benchmarks; live browser E2E; and a full security/recovery gate remain outstanding. See `SOFTWARE_RELEASE_READINESS.md`.
