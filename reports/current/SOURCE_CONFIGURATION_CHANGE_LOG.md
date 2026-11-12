# Source and configuration change log

## 2026-11-11 — token budgeting

- Added `Clarion-Backend/services/token_budget_service.py`: exact counting through explicitly configured local tokenizer; labeled UTF-8 byte estimate fallback; fixed-prompt budget; ranked whole-evidence packing; duplicate and omission reasons; no truncation of evidence strings.
- Added `LLM_CONTEXT_WINDOW`, `LLM_RESERVED_OUTPUT_TOKENS`, `LLM_TOKEN_SAFETY_TOKENS`, and optional cache-only `LLM_TOKENIZER_NAME` to settings and `.env.example`.
- Updated active `LLMInterface` query generation to preflight the configured context, send configured Ollama `num_ctx`, preserve its prior 800 output cap, capture provider counts and finish reason, and reject partial output when provider reports length stop.
- Updated `/query/{document_id}` to pack complete ranked passages and return omitted evidence IDs/reasons and explicit token metric provenance.
- Added five token-budget tests and assertions for the query response contract.
- No changes to embedding model, chunk size/overlap/minimum, production databases, retrieval ranking, or index files.

## Verification

Final full suite: 309 passed, 2 skipped, 0 failed (84.23s). A first run had a live Ollama timeout; the isolated test then passed warm and the final suite passed. Exact log: `research_artifacts/raw/software_readiness_backend_full_suite.log`.

## Explicitly incomplete

The central token service is not wired into `DatasetFactoryService` and every `BaseLLMProvider` implementation. Qwen exact tokenizer was not loaded; preflight remains an estimate by default. Context-template overhead is covered only by configurable safety reserve. No production schema or data migration was run.
